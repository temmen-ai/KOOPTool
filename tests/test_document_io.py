import unittest
from pathlib import Path

from document_io import ParagraphFeatures, paragraphs_to_dicts, read_document


DATA_DIR = Path(__file__).resolve().parents[1] / "inputmap"


class DocumentIOTestCase(unittest.TestCase):
    def test_hondenbelasting_counts(self):
        extraction = read_document(DATA_DIR / "Gouda 2 - Verordening hondenbelasting 2025.docx")
        self.assertEqual(len(extraction.paragraphs), 52)
        self.assertEqual(len(extraction.tables), 0)
        self.assertEqual(len(extraction.images), 0)

        first = extraction.paragraphs[0]
        self.assertIsInstance(first, ParagraphFeatures)
        self.assertEqual(first.tekst, "De raad van de gemeente Gouda;")
        self.assertEqual(first.indented, 0)
        self.assertEqual(first.genummerd, 0)
        self.assertGreater(len(first.textparts), 0)
        self.assertEqual(first.textpartformat_u, "Nee")
        self.assertEqual(first.textpartformat_b, "Nee")
        self.assertEqual(first.textpartformat_i, "Nee")

    def test_onroerendezaak_tables_and_numbering(self):
        extraction = read_document(DATA_DIR / "Gouda 3 - Verordening onroerende-zaakbelastingen 2025.docx")
        self.assertEqual(len(extraction.paragraphs), 56)
        self.assertEqual(len(extraction.tables), 1)
        self.assertEqual(extraction.tables[0].tekst, "TABEL GEVONDEN.")
        self.assertEqual(len(extraction.images), 0)

        numbered = next(p for p in extraction.paragraphs if p.genummerd == 1)
        self.assertEqual(numbered.volgnummer, 7)
        self.assertEqual(numbered.niveau, 0)
        self.assertIsInstance(numbered.numId, int)
        self.assertNotEqual(numbered.textparts, [])

    def test_paragraphs_to_dicts_structure(self):
        extraction = read_document(DATA_DIR / "Gouda 2 - Verordening hondenbelasting 2025.docx")
        dict_rows = paragraphs_to_dicts(extraction.paragraphs[:3])
        self.assertEqual(len(dict_rows), 3)
        for row in dict_rows:
            self.assertIsInstance(row, dict)
            self.assertIn("volgnummer", row)
            self.assertIn("tekst", row)
            self.assertIn("indented", row)
            self.assertIn("textparts", row)


if __name__ == "__main__":
    unittest.main()
