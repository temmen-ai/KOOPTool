from __future__ import annotations

import ast
import re
from typing import Optional

import pandas as pd
from structure.post_numbering_cleanup import post_numbering_cleanup
from structure.case_detection import _detect_case1_merge_targets


def _to_int(value, default=0):
    try:
        v = pd.to_numeric([value], errors="coerce")[0]
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default

BRON_COLUMNS = (
    ("indented", "indented_bron"),
    ("indentation", "indentation_bron"),
    ("genummerd", "genummerd_bron"),
    ("niveau", "niveau_bron"),
    ("numId", "numId_bron"),
)

LATIN_LETTERS = "A-Za-zÀ-ÖØ-öø-ÿ"
DIGIT_BOUNDARY = r"(?:\s+|(?=[^0-9]))"
LETTER_BOUNDARY = rf"(?:\s+|(?=[{LATIN_LETTERS}]))"

OPS_PATTERNS = {
    "bullet": r"^(?:\-|\•|°)\s*",
    "kleine letter": rf"^(?:[a-z]\s+|[a-z][\.\)]{LETTER_BOUNDARY})",
    "nummer_o": rf"^(?:\d+[oO°º]\s+|\d+[oO°º][\.\)]{LETTER_BOUNDARY})",
    "nummer": rf"^(?:\d+\s+|\d+[\.\)]{DIGIT_BOUNDARY})",
    "hoofdletter": rf"^(?:[A-Z]\s+|[A-Z][\.\)]{LETTER_BOUNDARY})",
    "romeins cijfer klein": rf"^(?:(ix|iv|v?i{{0,3}})\s+|(ix|iv|v?i{{0,3}})[\.\)]{LETTER_BOUNDARY})",
    "romeins cijfer groot": rf"^(?:(IX|IV|V?I{{0,3}})\s+|(IX|IV|V?I{{0,3}})[\.\)]{LETTER_BOUNDARY})",
}


def initialize_numbering(df: pd.DataFrame) -> pd.DataFrame:
    """Voeg *_bron kolommen toe zodat broninformatie beschikbaar blijft."""
    if df is None or df.empty:
        return df

    df = df.copy()
    for source, target in BRON_COLUMNS:
        if source not in df.columns:
            df[source] = _default_column(source, len(df))
        if target not in df.columns:
            df[target] = df[source]
    return df


def infer_opsommingstype(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    if "opsommingstype" not in df.columns:
        df["opsommingstype"] = None

    for idx, row in df.iterrows():
        current = str(row.get("opsommingstype") or "").strip()
        if current:
            continue

        if _should_skip_ops_detection(row):
            continue

        inferred = _opsommingstype_from_props(row.get("num_properties"))
        if inferred:
            df.at[idx, "opsommingstype"] = inferred
            continue

        text = _get_text(row)
        if not text:
            continue

        for name, pattern in OPS_PATTERNS.items():
            match = re.match(pattern, text)
            if match:
                df.at[idx, "opsommingstype"] = name
                new_text_raw = text[match.end():]
                stripped = new_text_raw.lstrip()
                if "tekst" in df.columns:
                    df.at[idx, "tekst"] = stripped
                trim_len = len(text) - len(stripped)
                _strip_textparts_prefix(df, idx, trim_len)
                break

    return df


def assign_consecutive_numids(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    counter = 1
    in_list = False
    for idx, row in df.iterrows():
        ops = str(row.get("opsommingstype") or "").strip()
        if ops:
            if not in_list:
                in_list = True
            df.at[idx, "numId"] = counter
        else:
            df.at[idx, "numId"] = None
            if in_list:
                counter += 1
                in_list = False
    return df


def enforce_list_baseline(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "numId" not in df.columns:
        return df

    df = df.copy()
    for num_id, group in df.groupby("numId", sort=False):
        if pd.isna(num_id):
            continue
        idxs = group.index
        df.loc[idxs, "niveau"] = 0
        if "indented" in df.columns:
            df.loc[idxs, "indented"] = 0

    return df


def apply_level_rules(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    states: dict[str, dict[str, dict]] = {}

    for idx in df.index:
        num_id = df.at[idx, "numId"]
        if pd.isna(num_id):
            continue
        key = str(num_id)
        state = states.setdefault(key, {"last": None, "signatures": {}, "type_levels": {}})

        ops = str(df.at[idx, "opsommingstype"] or "").strip()
        if not ops:
            continue

        bron_id = df.at[idx, "numId_bron"] if "numId_bron" in df.columns else None
        bron_level = _to_int(df.at[idx, "niveau_bron"], 0)

        target_level = _to_int(df.at[idx, "niveau"], 0)
        last = state["last"]

        if last is None:
            target_level = 0
        elif ops == last["ops"] and bron_id == last.get("bron_id"):
            delta = bron_level - last.get("bron_level", bron_level)
            if delta > 0:
                target_level = last["level"] + 1
            elif delta < 0:
                target_level = max(0, last["level"] + delta)
            else:
                target_level = last["level"]
        else:
            signature = _signature_for_row(row=df.loc[idx])
            known = state["signatures"].get(signature)
            if known is not None:
                target_level = known
            else:
                bron_numbered = _to_int(df.at[idx, "genummerd_bron"], 0) if "genummerd_bron" in df.columns else 0
                type_levels = state["type_levels"]
                fallback_level = type_levels.get(ops)
                if last and ops == last["ops"] and bron_numbered == 0:
                    target_level = last["level"]
                elif fallback_level is not None and bron_level <= fallback_level:
                    target_level = fallback_level
                else:
                    target_level = last["level"] + 1 if last else 0
                state["signatures"][signature] = target_level

        df.at[idx, "niveau"] = target_level
        if "indented" in df.columns:
            df.at[idx, "indented"] = target_level

        state["last"] = {
            "ops": ops,
            "level": target_level,
            "bron_level": bron_level,
            "bron_id": bron_id,
        }
        state["type_levels"][ops] = target_level
        state["signatures"].setdefault(_signature_for_row(row=df.loc[idx]), target_level)

    return df


def prepare_word_numbering(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    mask = df["opsommingstype"].astype(str).str.strip().ne("")
    df["numbered"] = mask.astype(int)
    df["numId"] = pd.to_numeric(df["numId"], errors="coerce").fillna(0).astype(int)
    df["niveau"] = pd.to_numeric(df["niveau"], errors="coerce").fillna(0).astype(int)
    return df


MERGE_PROFILES = {"OPLid", "Lijstalinea"}
MERGE_WARNING = "Deze zin is samengevoegd met 1 of meerdere vervolgzinnen uit het brondocument."


def merge_bridge_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combineer handgemaakte vervolgregels (zonder opsommingsteken) met hun anker.

    We herkennen blokken binnen OPLid/Lijstalinea waar alleen de eerste regel een
    opsommingsteken heeft en voegen de daaropvolgende regels zonder opsommingstype
    samen met dat anker. Dit gebeurt voordat we nieuwe numId/niveau-logica toepassen,
    zodat downstream stappen 1 paragraaf per logisch lijstitem zien.
    """
    if df is None or df.empty:
        return df

    if "opsommingstype" not in df.columns:
        return df

    df = df.copy()
    if "regel" in df.columns and df["regel"].dtype != object:
        df["regel"] = df["regel"].astype("object")

    to_drop: list[int] = []
    anchor_idx: Optional[int] = None

    for idx, row in df.iterrows():
        profile = row.get("opmaakprofiel")
        if profile not in MERGE_PROFILES:
            anchor_idx = None
            continue

        ops = str(row.get("opsommingstype") or "").strip()
        if ops:
            anchor_idx = idx
            continue

        if anchor_idx is None:
            continue
        if df.at[anchor_idx, "opmaakprofiel"] != profile:
            continue

        _merge_row_into_anchor(df, anchor_idx, idx)
        to_drop.append(idx)

    if to_drop:
        df.drop(index=to_drop, inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def merge_case1_bridges(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    case1_targets = _detect_case1_merge_targets(df)
    if not case1_targets:
        return df

    corrected = post_numbering_cleanup(df.copy())

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

    return corrected


def _strip_textparts_prefix(df: pd.DataFrame, index: int, trim_len: int) -> None:
    if trim_len <= 0:
        return
    if "textparts" not in df.columns:
        return

    parts = df.at[index, "textparts"]
    if isinstance(parts, str):
        try:
            parts = ast.literal_eval(parts)
        except Exception:
            parts = [parts]
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

    format_cols = ("textpartformats_b", "textpartformats_i", "textpartformats_u")
    for col in format_cols:
        if col not in df.columns:
            continue
        fmt = df.at[index, col]
        if isinstance(fmt, str):
            try:
                fmt = ast.literal_eval(fmt)
            except Exception:
                fmt = []
        if isinstance(fmt, list) and fmt:
            df.at[index, col] = fmt[removed_entries:]


def _merge_row_into_anchor(df: pd.DataFrame, anchor_idx: int, source_idx: int) -> None:
    anchor_text = str(df.at[anchor_idx, "tekst"]) if "tekst" in df.columns else ""
    source_text = str(df.at[source_idx, "tekst"]) if "tekst" in df.columns else ""
    if "tekst" in df.columns:
        if anchor_text and source_text:
            merged = anchor_text.rstrip() + " " + source_text.lstrip()
        else:
            merged = anchor_text + source_text
        df.at[anchor_idx, "tekst"] = merged

    list_columns = ("textparts", "textpartformats_u", "textpartformats_b", "textpartformats_i")
    for column in list_columns:
        if column not in df.columns:
            continue
        anchor_list = _as_list(df.at[anchor_idx, column])
        source_list = _as_list(df.at[source_idx, column])
        if not anchor_list and not source_list:
            continue
        df.at[anchor_idx, column] = anchor_list + source_list

    flag_columns = ("textpartformat_u", "textpartformat_b", "textpartformat_i")
    for column in flag_columns:
        if column not in df.columns:
            continue
        anchor_val = str(df.at[anchor_idx, column]).strip()
        source_val = str(df.at[source_idx, column]).strip()
        if source_val.lower() == "ja":
            df.at[anchor_idx, column] = "Ja"
        elif anchor_val == "":
            df.at[anchor_idx, column] = source_val

    if "fout" not in df.columns:
        df["fout"] = 0
    if "regel" not in df.columns:
        df["regel"] = ""
    current_fout = _to_int(df.at[anchor_idx, "fout"], 0)
    df.at[anchor_idx, "fout"] = max(2, current_fout)
    existing_regel = df.at[anchor_idx, "regel"]
    if not _has_text(existing_regel):
        df.at[anchor_idx, "regel"] = MERGE_WARNING


def _as_list(value) -> list:
    if isinstance(value, list):
        return value.copy()
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return [value]


def _has_text(value) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if value is None:
        return False
    try:
        return not pd.isna(value)
    except Exception:
        return False


def _signature_for_row(row: pd.Series) -> tuple:
    return (
        str(row.get("opsommingstype") or ""),
        row.get("numId_bron"),
        _to_int(row.get("niveau_bron"), 0),
        row.get("genummerd_bron"),
        row.get("indentation_bron"),
        row.get("indented_bron"),
    )


def _default_column(name: str, length: int) -> pd.Series:
    if name in {"indented", "genummerd"}:
        return pd.Series([0] * length)
    if name == "indentation":
        return pd.Series([0] * length, dtype="int64")
    if name == "niveau":
        return pd.Series([pd.NA] * length)
    if name == "numId":
        return pd.Series([None] * length)
    return pd.Series([pd.NA] * length)


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
        bullet = num_props.get("lvlText")
        if isinstance(bullet, str) and bullet.strip():
            return bullet.strip()
        return "bullet"
    if num_fmt in {"decimal", "decimalZero"}:
        return "nummer"
    return num_fmt


def _get_text(row) -> str:
    for key in ("tekst", "text"):
        value = row.get(key)
        if isinstance(value, str):
            text = value.replace("\xa0", " ").lstrip()
            if text:
                return text
    return ""


def _should_skip_ops_detection(row: pd.Series) -> bool:
    profile = str(row.get("opmaakprofiel") or "")
    if profile == "OPOndertekening":
        return True
    if profile != "OPAanhef":
        return False
    text = _get_text(row).replace(" ", "").replace(".", "").lower()
    return text in {"besluit", "besluit:", "besluit:"}
