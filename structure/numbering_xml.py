from __future__ import annotations

import ast
from typing import Optional

import pandas as pd
from lxml import etree as ET

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
                    or ""
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
