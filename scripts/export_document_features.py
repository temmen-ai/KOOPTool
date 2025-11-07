#!/usr/bin/env python3
"""Utility to dump paragraph/table/image data from a Word document to CSV."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from document_io import paragraphs_to_dicts, read_document

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export paragraph features (and optionally tables/images) to CSV (semicolon-delimited)."
    )
    parser.add_argument("input", type=Path, help="Path to the .doc/.docx file.")
    parser.add_argument(
        "output",
        type=Path,
        help="CSV destination for paragraph features; directories are created automatically.",
    )
    parser.add_argument(
        "--tables-output",
        type=Path,
        help="Optional CSV destination for table metadata.",
    )
    parser.add_argument(
        "--images-output",
        type=Path,
        help="Optional CSV destination for image metadata.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    extraction = read_document(args.input)

    paragraph_dicts = paragraphs_to_dicts(extraction.paragraphs)
    ensure_parent(args.output)
    pd.DataFrame(paragraph_dicts).to_csv(args.output, sep=";", index=False)

    if args.tables_output:
        ensure_parent(args.tables_output)
        pd.DataFrame([t.__dict__ for t in extraction.tables]).to_csv(
            args.tables_output, sep=";", index=False
        )

    if args.images_output:
        ensure_parent(args.images_output)
        pd.DataFrame([i.__dict__ for i in extraction.images]).to_csv(
            args.images_output, sep=";", index=False
        )

    print(
        f"Wrote {len(paragraph_dicts)} paragraphs to {args.output}"
        + (
            f", {len(extraction.tables)} tables to {args.tables_output}"
            if args.tables_output
            else ""
        )
        + (
            f", {len(extraction.images)} images to {args.images_output}"
            if args.images_output
            else ""
        )
    )


if __name__ == "__main__":
    main()
