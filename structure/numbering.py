from __future__ import annotations

import logging
import re
from typing import Optional

import ast
import pandas as pd
from lxml import etree as ET

from DaadkrachtBatch import post_numbering_cleanup  # reuse existing helper

logger = logging.getLogger(__name__)


def correct_numbering(
    df_in: pd.DataFrame,
    *,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    logger.info("correct_numbering aangeroepen.")
    df = df_in.copy()

    required_defaults = {
        "textpartformat_u": "Nee",
        "textpartformat_b": "Nee",
        "textpartformat_i": "Nee",
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    list_columns = ["textpartformats_u", "textpartformats_b", "textpartformats_i", "textparts"]
    for col in list_columns:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]

    for col in ["indented", "numbered"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0

    if "niveau" in df.columns:
        df["niveau"] = pd.to_numeric(df["niveau"], errors="coerce")
    else:
        df["niveau"] = pd.Series([pd.NA] * len(df))

    df["niveau_bron"] = df["niveau"].copy()
    df["indented_bron"] = df["indented"].copy()

    if "opsommingstype" not in df.columns:
        df["opsommingstype"] = None

    if "bron_numId" not in df.columns:
        df["bron_numId"] = df.get("numId")

    niveau_input = df["niveau"].copy()

    lijst_teller = 0
    nieuwe_lijst = "Ja"
    num_id_mapping = {}
    next_num_id = 1

    for index, row in df.iterrows():
        profile = row.get("opmaakprofiel")
        if profile == "OPOndertekening":
            continue
        if (row.get("tekst") in ["b e s l u i t:", "b e s l u i t :", "b   e   s   l   u   i   t   :"]) and profile == "OPAanhef":
            continue

        if profile in ["OPArtikelTitel", "OPParagraafTitel", "OPHoofdstukTitel"]:
            lijst_teller += 1
            nieuwe_lijst = "Ja"
            continue

        text = str(row.get("tekst", "")).replace("\xa0", " ").lstrip()
        is_opsomming = _to_int(row.get("numbered", 0), 0) == 1
        opsommingstype = None

        if not is_opsomming:
            latin_letters = "A-Za-zÀ-ÖØ-öø-ÿ"
            digit_boundary = r"(?:\s+|(?=[^0-9]))"
            letter_boundary = rf"(?:\s+|(?=[{latin_letters}]))"

            patterns = {
                "bullet": r"^(?:\-|\•|°)\s*",
                "kleine letter": rf"^(?:[a-z]\s+|[a-z][\.\)]{letter_boundary})",
                "nummer_o": rf"^(?:\d+[oO°º]\s+|\d+[oO°º][\.\)]{letter_boundary})",
                "nummer": rf"^(?:\d+\s+|\d+[\.\)]{digit_boundary})",
                "hoofdletter": rf"^(?:[A-Z]\s+|[A-Z][\.\)]{letter_boundary})",
                "romeins cijfer klein": rf"^(?:(ix|iv|v?i{{0,3}})\s+|(ix|iv|v?i{{0,3}})[\.\)]{letter_boundary})",
                "romeins cijfer groot": rf"^(?:(IX|IV|V?I{{0,3}})\s+|(IX|IV|V?I{{0,3}})[\.\)]{letter_boundary})",
            }
            for type_key, pattern in patterns.items():
                if re.match(pattern, text):
                    new_text_raw = re.sub(pattern, "", text)
                    new_text = new_text_raw.lstrip()
                    trim_len = len(text) - len(new_text)
                    df.at[index, "tekst"] = new_text
                    df.at[index, "numbered"] = 1
                    _strip_textparts_prefix(df, index, trim_len)
                    opsommingstype = type_key
                    is_opsomming = True
                    break

        if opsommingstype or is_opsomming:
            if nieuwe_lijst == "Ja":
                lijst_teller += 1
                nieuwe_lijst = "Nee"
                huidige_numId = next_num_id
                next_num_id += 1
            else:
                huidige_numId = num_id_mapping.get(lijst_teller, next_num_id)

            if lijst_teller not in num_id_mapping:
                num_id_mapping[lijst_teller] = huidige_numId
            df.at[index, "numId"] = huidige_numId

            if not opsommingstype:
                opsommingstype = _opsommingstype_from_props(row.get("num_properties"), niveau=_to_int(row.get("niveau", 0), 0))

            if opsommingstype:
                df.at[index, "opsommingstype"] = opsommingstype

            input_lvl = _to_int_nullable(niveau_input.iat[index])
            ind = _to_int(row.get("indented", 0), 0)
            if input_lvl is not None:
                niveau = input_lvl
            else:
                if row["opmaakprofiel"] == "OPLid" and index > 0 and df.at[index - 1, "opmaakprofiel"] == "OPArtikelTitel":
                    niveau = 0
                else:
                    niveau = ind if ind > 0 else 0

                STABILIZE = {"OPLid", "Lijstalinea"}
                if index > 0:
                    prev_prof = df.at[index - 1, "opmaakprofiel"]
                    prev_ind = _to_int(df.at[index - 1, "indented"], 0)
                    if prev_prof == row["opmaakprofiel"] and prev_prof in STABILIZE and prev_ind == ind:
                        prev_lvl = _to_int(df.at[index - 1, "niveau"], 0)
                        prev_input_lvl = _to_int_nullable(niveau_input.iat[index - 1])
                        if prev_input_lvl is None or prev_input_lvl == prev_lvl:
                            niveau = prev_lvl

            if niveau > 0 and _looks_like_visual_level0(row):
                niveau = 0
                df.at[index, "indented"] = 0
            df.at[index, "niveau"] = int(niveau)
        else:
            if profile not in {"OPLid", "Lijstalinea"}:
                nieuwe_lijst = "Ja"

    mask = df["opsommingstype"].isna()
    if "num_properties" in df.columns:
        df.loc[mask, "opsommingstype"] = df.loc[mask, "num_properties"].apply(
            lambda props: _opsommingstype_from_props(props, niveau=None)
        )

    case1_targets = _detect_case1_merge_targets(df)
    corrected = post_numbering_cleanup(df)
    if case1_targets:
        if "fout" not in corrected.columns:
            corrected["fout"] = 0
        if "regel" not in corrected.columns:
            corrected["regel"] = ""
        target_mask = corrected["volgnummer"].isin(case1_targets)
        corrected.loc[target_mask, "fout"] = corrected.loc[target_mask, "fout"].clip(lower=2)
        corrected.loc[target_mask, "regel"] = corrected.loc[target_mask, "regel"].where(
            corrected.loc[target_mask, "regel"].astype(bool),
            "Deze zin is samengevoegd met 1 of meerdere vervolgzinnen uit het brondocument.",
        )
    corrected = _promote_letter_after_number(corrected)
    mask_after = corrected["opsommingstype"].isna()
    if mask_after.any() and "num_properties" in corrected.columns:
        corrected.loc[mask_after, "opsommingstype"] = corrected.loc[
            mask_after, "num_properties"
        ].apply(lambda props: _opsommingstype_from_props(props, niveau=None))

    corrected = _infer_levels_for_unstyled_lists(corrected)
    corrected = _promote_letter_after_number(corrected)
    corrected = _restore_levels_by_bron(corrected)
    corrected["niveau"] = pd.to_numeric(corrected["niveau"], errors="coerce").fillna(0).astype(int)

    if output_csv:
        corrected.to_csv(output_csv, sep=";", index=False)

    return corrected


OPS_NUMFMT_MAP = {
    "nummer": "decimal",
    "kleine letter": "lowerLetter",
    "bullet": "bullet",
    "hoofdletter": "upperLetter",
    "romeins cijfer klein": "lowerRoman",
    "romeins cijfer groot": "upperRoman",
    "nummer_o": "decimal",
}

BULLET_SYMBOLS = {"•", "◦", "∙", "·", "●", "○", "‣", "⁃", "-", "–", "—", "▪", "▫", "■", "□"}


def voeg_entries_toe_from_dataframe(numbering_xml_tree, read_new_numbering_df):
    root = numbering_xml_tree.getroot()
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    w15_namespace = "http://schemas.microsoft.com/office/word/2012/wordml"
    ET.register_namespace("w", namespace)
    ET.register_namespace("w15", w15_namespace)

    grouped_df = read_new_numbering_df[read_new_numbering_df["numbered"] == 1].groupby("numId")
    num_elements = []

    for num_id, group in grouped_df:
        abstract_num_id = str(num_id)
        abstract_num = ET.Element(
            f"{{{namespace}}}abstractNum",
            attrib={
                f"{{{namespace}}}abstractNumId": abstract_num_id,
                f"{{{w15_namespace}}}restartNumberingAfterBreak": "0",
            },
        )
        ET.SubElement(abstract_num, f"{{{namespace}}}multiLevelType", attrib={f"{{{namespace}}}val": "hybridMultilevel"})

        for _, row in group.drop_duplicates(subset=["niveau"]).iterrows():
            niveau = int(row["niveau"])
            raw_ops = row.get("opsommingstype")
            opsommingstype, bullet_symbol = _normalize_opsommingstype(raw_ops)
            num_fmt_val = OPS_NUMFMT_MAP.get(opsommingstype, "decimal")

            lvl_elem = ET.SubElement(abstract_num, f"{{{namespace}}}lvl", attrib={f"{{{namespace}}}ilvl": str(niveau)})
            ET.SubElement(lvl_elem, f"{{{namespace}}}start", attrib={f"{{{namespace}}}val": "1"})
            ET.SubElement(lvl_elem, f"{{{namespace}}}numFmt", attrib={f"{{{namespace}}}val": num_fmt_val})

            if opsommingstype == "bullet":
                lvl_text = (
                    bullet_symbol
                    or _extract_bullet_from_props(row.get("num_properties"))
                    or "\uf0b7"
                )
            elif opsommingstype == "nummer_o":
                lvl_text = f"%{niveau + 1}°."
            else:
                lvl_text = f"%{niveau + 1}."
            ET.SubElement(lvl_elem, f"{{{namespace}}}lvlText", attrib={f"{{{namespace}}}val": lvl_text})

            left_indent = 720 + (niveau * 720)
            pPr = ET.SubElement(lvl_elem, f"{{{namespace}}}pPr")
            ET.SubElement(
                pPr,
                f"{{{namespace}}}ind",
                attrib={f"{{{namespace}}}left": str(left_indent), f"{{{namespace}}}hanging": "360"},
            )
            rPr = ET.SubElement(lvl_elem, f"{{{namespace}}}rPr")
            if opsommingstype == "bullet":
                ET.SubElement(
                    rPr,
                    f"{{{namespace}}}rFonts",
                    attrib={
                        f"{{{namespace}}}ascii": "Symbol",
                        f"{{{namespace}}}hAnsi": "Symbol",
                        f"{{{namespace}}}hint": "default",
                    },
                )

        root.append(abstract_num)
        num = ET.Element(f"{{{namespace}}}num", attrib={f"{{{namespace}}}numId": abstract_num_id})
        ET.SubElement(num, f"{{{namespace}}}abstractNumId", attrib={f"{{{namespace}}}val": abstract_num_id})
        lvl_override = ET.SubElement(num, f"{{{namespace}}}lvlOverride", attrib={f"{{{namespace}}}ilvl": "0"})
        ET.SubElement(lvl_override, f"{{{namespace}}}startOverride", attrib={f"{{{namespace}}}val": "1"})
        num_elements.append(num)

    for num_element in num_elements:
        root.append(num_element)

    return numbering_xml_tree


def _normalize_opsommingstype(value) -> tuple[str, Optional[str]]:
    text = str(value or "").strip()
    if not text:
        return "", None
    lowered = text.lower()
    if lowered == "bullet":
        return "bullet", None
    if text in BULLET_SYMBOLS:
        return "bullet", text
    if len(text) == 1 and not text.isalnum():
        return "bullet", text
    if lowered in OPS_NUMFMT_MAP:
        return lowered, None
    return text, None


def _extract_bullet_from_props(num_props) -> Optional[str]:
    if isinstance(num_props, str):
        try:
            num_props = ast.literal_eval(num_props)
        except Exception:
            num_props = None
    if isinstance(num_props, dict):
        raw = num_props.get("lvlText")
        if isinstance(raw, str) and raw.strip():
            text = raw.strip()
            if text in BULLET_SYMBOLS or (len(text) == 1 and not text.isalnum()):
                return text
            return text
    return None


def _to_int(x, default=0):
    try:
        v = pd.to_numeric([x], errors="coerce")[0]
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def _to_int_nullable(x):
    try:
        v = pd.to_numeric([x], errors="coerce")[0]
        if pd.isna(v):
            return None
        return int(v)
    except Exception:
        return None


def _opsommingstype_from_props(num_props, niveau: Optional[int] = None) -> Optional[str]:
    if isinstance(num_props, str):
        try:
            num_props = ast.literal_eval(num_props)
        except Exception:
            return None
    if not isinstance(num_props, dict):
        return None
    num_fmt = num_props.get("numFmt")
    if not num_fmt:
        return None
    level = niveau
    if level is None:
        try:
            level = int(num_props.get("ilvl"))
        except Exception:
            level = 0

    num_fmt = str(num_fmt)
    if num_fmt in {"lowerLetter", "cardinalText"}:
        return "kleine letter"
    if num_fmt == "upperLetter":
        return "hoofdletter"
    if num_fmt == "lowerRoman":
        return "romeins cijfer klein"
    if num_fmt == "upperRoman":
        return "romeins cijfer groot"
    if num_fmt == "bullet":
        return "bullet"
    if num_fmt in {"decimal", "decimalZero"}:
        return "nummer"
    return num_fmt


def _infer_levels_for_unstyled_lists(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _safe_int(value, default=0):
        try:
            v = pd.to_numeric([value], errors="coerce")[0]
            if pd.isna(v):
                return default
            return int(v)
        except Exception:
            return default

    previous_level = 0
    previous_type = None
    current_list_id = None
    force_level0 = False
    expected_number = None
    list_started = False
    previous_indent = 0
    previous_num_signature: Optional[tuple] = None

    for idx in df.index:
        row_data = df.loc[idx]
        prof = row_data["opmaakprofiel"]
        if prof not in {"OPLid", "Lijstalinea"}:
            previous_type = None
            current_list_id = None
            force_level0 = False
            expected_number = None
            continue

        if _safe_int(df.at[idx, "numbered"], 0) != 1:
            # Geen onderdeel van een lijst: sla over maar behoud context
            continue

        ops = str(df.at[idx, "opsommingstype"] or "").lower()
        if ops not in {"nummer", "kleine letter", "hoofdletter", "bullet", "romeins cijfer klein", "romeins cijfer groot"}:
            previous_type = None
            current_list_id = None
            force_level0 = False
            expected_number = None
            continue

        list_id = row_data.get("numId")
        if list_id != current_list_id:
            current_list_id = list_id
            force_level0 = False
            expected_number = None
            previous_level = 0
            list_started = False
            previous_indent = 0
            previous_num_signature = None

        current_level = _safe_int(row_data.get("niveau"), 0)
        if not list_started or (current_level > 0 and _looks_like_visual_level0(df.loc[idx])):
            current_level = 0
            df.at[idx, "niveau"] = 0
        indent_raw = _safe_int(row_data.get("indented"), 0)
        if not list_started:
            previous_indent = indent_raw
        indent_level = 1 if indent_raw > 0 else 0
        base_level = current_level if current_level > 0 else 0

        leading_number = _leading_numeric_value(df, idx)
        if leading_number is not None:
            if expected_number is None:
                force_level0 = indent_raw == 0
                expected_number = leading_number + 1
            else:
                if leading_number == expected_number:
                    expected_number = leading_number + 1
                else:
                    force_level0 = False
                    expected_number = leading_number + 1
        else:
            force_level0 = False
            expected_number = None

        def _reset_indents():
            if "indented" in df.columns:
                df.at[idx, "indented"] = 0
            for col in ("tab_stops", "has_tab_runs", "first_line_indent", "hanging_indent"):
                if col in df.columns:
                    if col == "tab_stops":
                        df.at[idx, col] = "[]"
                    else:
                        df.at[idx, col] = 0

        if ops == "nummer":
            if force_level0 and leading_number is not None:
                previous_level = 0
                df.at[idx, "niveau"] = 0
                _reset_indents()
                previous_type = ops
                continue
            if current_level == 0:
                base_level = 0
            previous_level = base_level
            df.at[idx, "niveau"] = previous_level
            _reset_indents()
        elif ops == "hoofdletter":
            previous_level = max(base_level, previous_level)
            df.at[idx, "niveau"] = previous_level
        elif ops == "kleine letter":
            current_signature = _num_signature(row_data)
            if not list_started:
                previous_level = 0
            elif previous_type == "bullet":
                candidate = previous_level - 1 if previous_level > 0 else 0
                previous_level = max(base_level, candidate)
            elif previous_type in {"nummer", "hoofdletter"}:
                previous_level = max(base_level, previous_level + 1)
            elif previous_type in {"romeins cijfer klein", "romeins cijfer groot"}:
                previous_level = max(base_level, max(previous_level - 1, 0))
            elif previous_type == "kleine letter":
                if current_signature and previous_num_signature == current_signature:
                    previous_level = previous_level
                elif indent_raw <= previous_indent:
                    previous_level = previous_level
                else:
                    previous_level = max(base_level, previous_level + 1)
            else:
                previous_level = base_level
            df.at[idx, "niveau"] = previous_level
        elif ops in {"romeins cijfer klein", "romeins cijfer groot"}:
            if previous_type == "kleine letter":
                previous_level = max(base_level, previous_level + 1)
            else:
                previous_level = max(base_level, previous_level)
            df.at[idx, "niveau"] = previous_level
        elif ops == "bullet":
            if previous_type in {"nummer", "hoofdletter", "kleine letter"}:
                target = max(previous_level + 1, indent_level or 1)
            elif previous_type == "bullet":
                target = max(previous_level, indent_level, base_level)
            else:
                target = max(base_level, 1 if indent_raw > 0 else 0)
            previous_level = target
            df.at[idx, "niveau"] = previous_level
            force_level0 = False
            expected_number = None
        previous_type = ops
        list_started = True
        previous_indent = indent_raw
        if ops == "kleine letter":
            previous_num_signature = current_signature
        else:
            previous_num_signature = _num_signature(row_data)

    return df


def _detect_case1_merge_targets(df: pd.DataFrame) -> set:
    PROFILES = {"OPLid", "Lijstalinea"}

    def _valid_numid(value) -> bool:
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        return text not in {"", "none", "nan"}

    targets: set = set()
    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        if row.get("opmaakprofiel") not in PROFILES:
            continue
        if _valid_numid(row.get("numId")):
            continue
        prev = df.iloc[i - 1]
        nxt = df.iloc[i + 1]
        if (
            prev.get("opmaakprofiel") in PROFILES
            and nxt.get("opmaakprofiel") in PROFILES
            and _valid_numid(prev.get("numId"))
            and str(prev.get("numId")) == str(nxt.get("numId"))
        ):
            targets.add(prev.get("volgnummer"))
    return targets


def _strip_textparts_prefix(df: pd.DataFrame, index: int, trim_len: int) -> None:
    if trim_len <= 0:
        return
    if "textparts" not in df.columns:
        return
    parts = df.at[index, "textparts"]
    if not isinstance(parts, list) or not parts:
        return

    remaining = trim_len
    new_parts = []
    removed_entries = 0

    for part in parts:
        part_str = part if isinstance(part, str) else str(part)
        if remaining <= 0:
            new_parts.append(part_str)
            continue
        if len(part_str) <= remaining:
            remaining -= len(part_str)
            removed_entries += 1
            continue
        new_parts.append(part_str[remaining:])
        remaining = 0

    if not new_parts:
        new_parts = [""]
    df.at[index, "textparts"] = new_parts

    for fmt_col in ("textpartformats_b", "textpartformats_i", "textpartformats_u"):
        if fmt_col in df.columns:
            fmt = df.at[index, fmt_col]
            if isinstance(fmt, list) and fmt:
                df.at[index, fmt_col] = fmt[removed_entries:]


def _normalize_num_id(value) -> Optional[str]:
    if value in (None, "", "None"):
        return None
    try:
        as_int = int(float(value))
        return str(as_int)
    except Exception:
        return str(value).strip()


def _promote_letter_after_number(df: pd.DataFrame) -> pd.DataFrame:
    PROFILES = {"OPLid", "Lijstalinea"}

    def _safe_int(value, default=0):
        try:
            v = pd.to_numeric([value], errors="coerce")[0]
            if pd.isna(v):
                return default
            return int(v)
        except Exception:
            return default

    state: dict[str, dict[str, int | None]] = {}
    for idx, row in df.iterrows():
        profile = row.get("opmaakprofiel")
        if profile not in PROFILES:
            continue
        if _safe_int(row.get("numbered"), 0) != 1:
            continue

        current_type = str(row.get("opsommingstype") or "").lower()
        num_id = _normalize_num_id(row.get("numId"))
        if num_id is None:
            state.clear()
            continue
        level = _safe_int(row.get("niveau"), 0)

        entry = state.setdefault(num_id, {"type": None, "level": 0, "last_number_level": None})

        last_number_level = entry.get("last_number_level")
        if current_type == "nummer":
            entry["last_number_level"] = level
        elif current_type == "kleine letter" and last_number_level is not None:
            target_level = max(last_number_level + 1, 1)
            if level < target_level:
                df.at[idx, "niveau"] = target_level
                if "indented" in df.columns:
                    df.at[idx, "indented"] = target_level
            level = max(level, target_level)

        entry["type"] = current_type
        entry["level"] = level

    return df


def _looks_like_visual_level0(row) -> bool:
    try:
        ind = _to_int(row.get("indented", 0), 0)
        first_line = _to_int(row.get("first_line_indent", 0), 0)
        hanging = _to_int(row.get("hanging_indent", 0), 0)
    except Exception:
        return False

    if ind > 1:
        return False
    if first_line >= 0 or hanging <= 0:
        return False

    if abs(abs(first_line) - hanging) <= 150:  # allow small rounding differences
        return True
    return False


def _leading_numeric_value(df: pd.DataFrame, idx) -> Optional[int]:
    for column in ("leading_text", "leading_run_text"):
        if column not in df.columns:
            continue
        raw = str(df.at[idx, column] or "").strip()
        if not raw:
            continue

        pos = 0
        while pos < len(raw) and raw[pos].isdigit():
            pos += 1
        if pos == 0:
            continue

        number_part = raw[:pos]
        remainder = raw[pos:]
        had_punctuation = False

        if remainder.startswith((".", ")")):
            had_punctuation = True
            remainder = remainder[1:]

        if not remainder:
            return int(number_part)

        if remainder[0].isspace():
            return int(number_part)

        if had_punctuation and remainder[0].isalpha():
            return int(number_part)

        # No whitespace and no punctuation boundary -> treat as plain text
        continue
    return None


def _restore_levels_by_bron(df: pd.DataFrame) -> pd.DataFrame:
    if "bron_numId" not in df.columns:
        return df

    baselines: dict[str, int] = {}
    last_bron: Optional[str] = None

    for idx, row in df.iterrows():
        if _to_int(row.get("numbered"), 0) != 1:
            continue

        bron = row.get("bron_numId")
        if bron in (None, "", "None", "nan"):
            continue
        bron_key = str(bron)
        ops = str(row.get("opsommingstype") or "")
        level = _to_int(row.get("niveau"), 0)

        if ops == "kleine letter" and bron_key not in baselines:
            baselines[bron_key] = level

        if (
            ops == "kleine letter"
            and bron_key in baselines
            and level > baselines[bron_key]
        ):
            target = baselines[bron_key]
            df.at[idx, "niveau"] = target
            if "indented" in df.columns:
                df.at[idx, "indented"] = target

        last_bron = bron_key

    return df


def _num_signature(row) -> Optional[tuple]:
    props = row.get("num_properties")
    if isinstance(props, str):
        try:
            props = ast.literal_eval(props)
        except Exception:
            props = {}
    if not isinstance(props, dict):
        props = {}
    return (
        row.get("numId"),
        props.get("abstractNumId"),
        props.get("lvlText"),
        props.get("numFmt"),
    )
