from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from lxml import etree as ET

from writers.docx_helper import DocxHelper
from structure.numbering_xml import voeg_entries_toe_from_dataframe


def escape_placeholders(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def generate_xml_files(
    df: pd.DataFrame,
    *,
    base_name: str,
    template_dir: Path = Path("template/xml"),
    output_dir: Path = Path("resultaat/xml"),
    temp_dir: Path = Path("temp_files"),
) -> Tuple[Path, Path, bool, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_xml = temp_dir / f"{base_name}_temp.xml"
    temp_comments = temp_dir / f"{base_name}_tempcomments.xml"
    for path in (temp_xml, temp_comments):
        if path.exists():
            path.unlink()

    comments_generated = False
    comments_include_errors = False
    fout_teller = -1

    with temp_xml.open("w", encoding="utf-8") as body, temp_comments.open("w", encoding="utf-8") as comments:
        for _, row in df.iterrows():
            profile = str(row.get("opmaakprofiel", ""))
            foutwaarde = int(pd.to_numeric(row.get("fout", 0), errors="coerce") or 0)
            regeltekst = str(row.get("regel", ""))

            if foutwaarde in {1, 2}:
                comments_generated = True
                if foutwaarde == 1:
                    comments_include_errors = True
                fout_teller += 1
                highlight_text_1 = '<w:highlight w:val="yellow"/>'
                highlight_text_2 = '<w:highlight w:val="yellow"/>'
                highlight_text_3 = '<w:rPr><w:highlight w:val="yellow"/></w:rPr>'
                comment_start = f'<w:commentRangeStart w:id="{fout_teller}"/>'
                comment_end = (
                    f'<w:commentRangeEnd w:id="{fout_teller}"/>'
                    f'<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr>'
                    f'<w:commentReference w:id="{fout_teller}"/></w:r>'
                )
                comment = (
                    f'<w:comment w:id="{fout_teller}" w:author="KOOP Converter" w:initials="KC">'
                    f'<w:p><w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr>'
                    f'<w:annotationRef/></w:r><w:r><w:rPr><w:color w:val="000000"/></w:rPr>'
                    f'<w:t>{escape_placeholders(regeltekst)}</w:t></w:r></w:p></w:comment>'
                )
                comments.write(comment)
            else:
                highlight_text_1 = highlight_text_2 = highlight_text_3 = ""
                comment_start = comment_end = ""

            body.write(f'<w:p><w:pPr><w:pStyle w:val="{profile}"/>')

            if int(pd.to_numeric(row.get("numbered", 0), errors="coerce") or 0) == 1:
                level = int(pd.to_numeric(row.get("niveau", 0), errors="coerce") or 0)
                num_id = int(pd.to_numeric(row.get("numId", 0), errors="coerce") or 0)
                body.write(f'<w:numPr><w:ilvl w:val="{level}" /><w:numId w:val="{num_id}"/></w:numPr>')

            body.write(f'</w:pPr>{comment_start}')

            paragraph_text = str(row.get("tekst", ""))
            bold_tag = '<w:b/>' if int(row.get("bold", 0) or 0) == 1 else ""
            italic_tag = '<w:i/>' if int(row.get("italic", 0) or 0) == 1 else ""
            underline_tag = '<w:u w:val="single"/>' if int(row.get("underlined", 0) or 0) == 1 else ""

            textparts = row.get("textparts") or []
            if (
                (row.get("textpartformat_u", "Nee") == "Nee")
                and (row.get("textpartformat_b", "Nee") == "Nee")
                and (row.get("textpartformat_i", "Nee") == "Nee")
            ) or profile == "OPArtikelTitel" or not textparts:
                tekst_lb = escape_placeholders(paragraph_text).replace(
                    "\n", '</w:t><w:br/><w:t xml:space="preserve">'
                )
                text_cleaned = re.sub(r" {2,}", " ", tekst_lb)
                body.write(
                    f'<w:r><w:rPr>{highlight_text_1}{bold_tag}{italic_tag}{underline_tag}</w:rPr>'
                    f'<w:t xml:space="preserve">{text_cleaned}</w:t></w:r>'
                )
            else:
                formats_b = row.get("textpartformats_b") or []
                formats_i = row.get("textpartformats_i") or []
                formats_u = row.get("textpartformats_u") or []
                format_b_flag = row.get("textpartformat_b", "Nee") == "Ja"
                format_i_flag = row.get("textpartformat_i", "Nee") == "Ja"
                format_u_flag = row.get("textpartformat_u", "Nee") == "Ja"
                for i, part in enumerate(textparts):
                    has_format = False
                    body.write("<w:r>")
                    if (
                        (format_b_flag and formats_b and i < len(formats_b) and formats_b[i] not in ("None", False))
                        or (format_u_flag and formats_u and i < len(formats_u) and formats_u[i] not in ("None", False))
                        or (format_i_flag and formats_i and i < len(formats_i) and formats_i[i] not in ("None", False))
                    ):
                        body.write("<w:rPr>")
                        if format_b_flag and formats_b and i < len(formats_b) and formats_b[i] not in ("None", False):
                            body.write("<w:b/><w:bCs/>")
                            has_format = True
                        if format_i_flag and formats_i and i < len(formats_i) and formats_i[i] not in ("None", False):
                            body.write("<w:i/><w:iCs/>")
                            has_format = True
                        if format_u_flag and formats_u and i < len(formats_u) and formats_u[i] not in ("None", False):
                            val = int(pd.to_numeric(formats_u[i], errors="coerce") or 1)
                            underline_val = "double" if val == 3 else "single"
                            body.write(f'<w:u w:val="{underline_val}" />')
                            has_format = True
                        body.write(f"{highlight_text_2}</w:rPr>")
                    else:
                        body.write(highlight_text_3)

                    tekst_met_lb = escape_placeholders(part).replace(
                        "\n", '</w:t><w:br/><w:t xml:space="preserve">'
                    )
                    text_cleaned_met_lb = re.sub(r" {2,}", " ", tekst_met_lb)
                    body.write(f'<w:t xml:space="preserve">{text_cleaned_met_lb}</w:t></w:r>')

            body.write(f"{comment_end}</w:p>")

            if profile == "OPAanhef":
                body.write('<w:p><w:pPr><w:pStyle w:val="OPAanhef" /></w:pPr></w:p>')

    document_xml_path = output_dir / f"{base_name}_document.xml"
    comments_xml_path = output_dir / f"{base_name}_comments.xml"

    with (template_dir / "document_header.xml").open("r", encoding="utf-8") as header_file, (
        template_dir / "document_footer.xml"
    ).open("r", encoding="utf-8") as footer_file, temp_xml.open("r", encoding="utf-8") as body_file:
        header = header_file.read()
        footer = footer_file.read()
        body = body_file.read()

    document_xml_path.write_text(header + body + footer, encoding="utf-8")
    temp_xml.unlink(missing_ok=True)

    with temp_comments.open("a", encoding="utf-8") as tmp_comments:
        tmp_comments.write("</w:comments>")

    with (template_dir / "comments_header.xml").open("r", encoding="utf-8") as comments_header_file, temp_comments.open(
        "r", encoding="utf-8"
    ) as comments_body_file:
        comments_header = comments_header_file.read()
        comments_body = comments_body_file.read()

    comments_xml_path.write_text(comments_header + comments_body, encoding="utf-8")
    temp_comments.unlink(missing_ok=True)

    sanitize_xml(document_xml_path)

    return document_xml_path, comments_xml_path, comments_generated, comments_include_errors


def sanitize_xml(xml_path: Path) -> None:
    text = xml_path.read_text(encoding="utf-8")
    text = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;)", "&amp;", text)
    xml_path.write_text(text, encoding="utf-8")


def build_word_document(
    df: pd.DataFrame,
    *,
    input_document: Path | str,
    output_document: Path | str,
    base_name: str,
    template_docx: Path = Path("template/OP_Stijl Compleet Besluit v2.5_0.docx"),
    template_xml_dir: Path = Path("template/xml"),
    result_xml_dir: Path = Path("resultaat/xml"),
    temp_dir: Path = Path("temp_files"),
) -> Tuple[bool, bool]:
    document_xml_path, comments_xml_path, comments_generated, comments_include_errors = generate_xml_files(
        df,
        base_name=base_name,
        template_dir=template_xml_dir,
        output_dir=result_xml_dir,
        temp_dir=temp_dir,
    )

    dh = DocxHelper(str(template_docx))

    numbering_tree = ET.parse(template_xml_dir / "numbering.xml")
    numbering_df = df.copy()
    for col in ["numbered", "numId", "niveau"]:
        numbering_df[col] = pd.to_numeric(numbering_df.get(col, 0), errors="coerce").fillna(0).astype(int)

    numbering_tree_ext = voeg_entries_toe_from_dataframe(numbering_tree, numbering_df)

    if os.name == "nt":
        with open(document_xml_path, encoding="windows-1252") as xml_file:
            xml_tree = ET.parse(xml_file)
    else:
        xml_tree = ET.parse(document_xml_path)

    output_document = Path(output_document)
    output_document.parent.mkdir(parents=True, exist_ok=True)

    if comments_generated:
        dh.set_xmls_comments(
            "word/document.xml",
            str(comments_xml_path),
            "word/numbering.xml",
            xml_tree,
            numbering_tree_ext,
            str(output_document),
        )
    else:
        dh.set_xmls(
            "word/document.xml",
            "word/numbering.xml",
            xml_tree,
            numbering_tree_ext,
            str(output_document),
        )

    return comments_generated, comments_include_errors
