from __future__ import annotations

import logging
import re
from typing import Optional

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
            patterns = {
                "bullet": r"^(?:\-|\•|°)\s*",
                "kleine letter": r"^[a-z][\.\)]?\s+",
                "nummer_o": r"^\d+[oO°º][\.\)]?\s+",
                "nummer": r"^\d+[\.\)]?\s+",
                "hoofdletter": r"^[A-Z][\.\)]?\s+",
                "romeins cijfer klein": r"^(ix|iv|v?i{0,3})[\.\)]?\s+",
                "romeins cijfer groot": r"^(IX|IV|V?I{0,3})[\.\)]?\s+",
            }
            for type_key, pattern in patterns.items():
                if re.match(pattern, text):
                    df.at[index, "tekst"] = re.sub(pattern, "", text)
                    df.at[index, "numbered"] = 1
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

            if opsommingstype:
                df.at[index, "opsommingstype"] = opsommingstype
            else:
                num_props = row.get("num_properties", {})
                if isinstance(num_props, str):
                    try:
                        import ast

                        num_props = ast.literal_eval(num_props)
                    except Exception:
                        num_props = {}
                mapping = {
                    "decimal": "nummer",
                    "decimalZero": "nummer",
                    "cardinalText": "nummer",
                    "lowerLetter": "kleine letter",
                    "upperLetter": "hoofdletter",
                    "lowerRoman": "romeins cijfer klein",
                    "upperRoman": "romeins cijfer groot",
                    "bullet": "bullet",
                }
                num_fmt = (num_props or {}).get("numFmt")
                if num_fmt:
                    df.at[index, "opsommingstype"] = mapping.get(str(num_fmt), str(num_fmt))

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

            df.at[index, "niveau"] = int(niveau)
        else:
            nieuwe_lijst = "Ja"

    corrected = post_numbering_cleanup(df)
    corrected["niveau"] = pd.to_numeric(corrected["niveau"], errors="coerce").fillna(0).astype(int)

    if output_csv:
        corrected.to_csv(output_csv, sep=";", index=False)

    return corrected


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
            opsommingstype = row.get("opsommingstype")
            num_fmt_val = {
                "nummer": "decimal",
                "kleine letter": "lowerLetter",
                "bullet": "bullet",
                "hoofdletter": "upperLetter",
                "romeins cijfer klein": "lowerRoman",
                "romeins cijfer groot": "upperRoman",
                "nummer_o": "decimal",
            }.get(opsommingstype, "decimal")

            lvl_elem = ET.SubElement(abstract_num, f"{{{namespace}}}lvl", attrib={f"{{{namespace}}}ilvl": str(niveau)})
            ET.SubElement(lvl_elem, f"{{{namespace}}}start", attrib={f"{{{namespace}}}val": "1"})
            ET.SubElement(lvl_elem, f"{{{namespace}}}numFmt", attrib={f"{{{namespace}}}val": num_fmt_val})

            if opsommingstype == "bullet":
                lvl_text = "-"
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

        root.append(abstract_num)
        num = ET.Element(f"{{{namespace}}}num", attrib={f"{{{namespace}}}numId": abstract_num_id})
        ET.SubElement(num, f"{{{namespace}}}abstractNumId", attrib={f"{{{namespace}}}val": abstract_num_id})
        lvl_override = ET.SubElement(num, f"{{{namespace}}}lvlOverride", attrib={f"{{{namespace}}}ilvl": "0"})
        ET.SubElement(lvl_override, f"{{{namespace}}}startOverride", attrib={f"{{{namespace}}}val": "1"})
        num_elements.append(num)

    for num_element in num_elements:
        root.append(num_element)

    return numbering_xml_tree


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
