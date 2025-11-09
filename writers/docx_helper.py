from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import mkdtemp
from zipfile import ZipFile
from os import path
from lxml import etree as ET


class DocxHelper:
    def __init__(self, docx_file: str | Path):
        self.docx_file = str(docx_file)

    def get_xml(self, filename: str):
        with ZipFile(self.docx_file) as docx:
            with docx.open(filename) as xml:
                return ET.parse(xml)

    def set_xml(self, filename: str, tree: ET.ElementTree, output_filename: str | Path):
        tmp_dir = mkdtemp()
        try:
            with ZipFile(self.docx_file) as docx:
                filenames = docx.namelist()
                docx.extractall(tmp_dir)

            tree.write(path.join(tmp_dir, filename), pretty_print=True)

            with ZipFile(output_filename, "w") as docx:
                for name in filenames:
                    docx.write(path.join(tmp_dir, name), name)
        finally:
            shutil.rmtree(tmp_dir)

    def set_xmls(self, filename: str, numbering_xml: str, tree: ET.ElementTree, numbering_xml_tree: ET.ElementTree, output_filename: str | Path):
        tmp_dir = mkdtemp()
        try:
            with ZipFile(self.docx_file) as docx:
                filenames = docx.namelist()
                docx.extractall(tmp_dir)

            shutil.copyfile("template/xml/styles.xml", path.join(tmp_dir, "word/styles.xml"))

            tree.write(path.join(tmp_dir, filename), pretty_print=True)
            numbering_xml_tree.write(path.join(tmp_dir, numbering_xml), pretty_print=True)

            with ZipFile(output_filename, "w") as docx:
                for name in filenames:
                    docx.write(path.join(tmp_dir, name), name)
        finally:
            shutil.rmtree(tmp_dir)

    def set_xmls_comments(
        self,
        filename: str,
        comments_xml: str,
        numbering_xml: str,
        tree: ET.ElementTree,
        numbering_xml_tree: ET.ElementTree,
        output_filename: str | Path,
    ):
        tmp_dir = mkdtemp()
        try:
            with ZipFile(self.docx_file) as docx:
                filenames = docx.namelist()
                docx.extractall(tmp_dir)

            shutil.copyfile(comments_xml, path.join(tmp_dir, "word/comments.xml"))
            shutil.copyfile("template/xml/commentsExtended.xml", path.join(tmp_dir, "word/commentsExtended.xml"))
            shutil.copyfile("template/xml/commentsIds.xml", path.join(tmp_dir, "word/commentsIds.xml"))
            shutil.copyfile("template/xml/styles.xml", path.join(tmp_dir, "word/styles.xml"))
            shutil.copyfile("template/xml/[Content_Types].xml", path.join(tmp_dir, "[Content_Types].xml"))
            shutil.copyfile("template/xml/document.xml.rels", path.join(tmp_dir, "word/_rels/document.xml.rels"))

            filenames.extend(
                [
                    "word/comments.xml",
                    "word/commentsExtended.xml",
                    "word/commentsIds.xml",
                ]
            )

            tree.write(path.join(tmp_dir, filename), pretty_print=True)
            numbering_xml_tree.write(path.join(tmp_dir, numbering_xml), pretty_print=True)

            with ZipFile(output_filename, "w") as docx:
                for name in filenames:
                    docx.write(path.join(tmp_dir, name), name)
        finally:
            shutil.rmtree(tmp_dir)
