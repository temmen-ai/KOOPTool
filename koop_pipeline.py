from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from document_io import paragraphs_to_dicts, read_document
from ml.predictor import BertOpmaakprofielPredictor
from services.title_resolver import ensure_title
from structure.checker import check_structure

logger = logging.getLogger(__name__)


def process_document(
    input_path: Path | str,
    *,
    export_features: bool = True,
    export_dir: Optional[Path | str] = None,
    run_predictions: bool = True,
    predictor: Optional[BertOpmaakprofielPredictor] = None,
    ensure_title_step: bool = True,
    run_structure_check: bool = True,
):
    """
    Entry point for the new KOOP pipeline.

    Voor nu: leest een document en schrijft (optioneel) de paragraph features naar CSV.
    Later koppelen we hier modelpredicties, controles en Word-output aan.
    """
    input_path = Path(input_path)
    extraction = read_document(input_path)

    logger.info(
        "Document '%s' gelezen: %d paragrafen, %d tabellen, %d afbeeldingen.",
        input_path,
        len(extraction.paragraphs),
        len(extraction.tables),
        len(extraction.images),
    )

    predictions_df = None
    title_added = False
    structure_df = None
    paragraphs_df = pd.DataFrame(paragraphs_to_dicts(extraction.paragraphs))
    if run_predictions:
        predictor = predictor or BertOpmaakprofielPredictor()
        logger.info("BERT-model geladen, start sliding-window voorspellingen.")
        predictions_df = predictor.predict(extraction.paragraphs)
        logger.info("Voorspellingen voltooid voor %d paragrafen.", len(extraction.paragraphs))

        if export_features:
            raw_pred_csv = (Path(export_dir) if export_dir else Path("resultaat/csv")) / f"{input_path.stem}_predictions_raw.csv"
            raw_pred_csv.parent.mkdir(parents=True, exist_ok=True)
            predictions_df.to_csv(raw_pred_csv, sep=";", index=False)
            logger.info("Ruwe voorspellingen weggeschreven naar %s", raw_pred_csv)

        if ensure_title_step:
            title_csv = None
            if export_features:
                export_dir_path = Path(export_dir) if export_dir else Path("resultaat/csv")
                title_csv = export_dir_path / f"{input_path.stem}_predictions_title.csv"
            predictions_df, title_added = ensure_title(predictions_df, export_csv_path=str(title_csv) if title_csv else None)
            logger.info("Titel toegevoegd door AI: %s", title_added)

        if run_structure_check and predictions_df is not None:
            structure_csv = None
            if export_features:
                export_dir_path = Path(export_dir) if export_dir else Path("resultaat/csv")
                structure_csv = export_dir_path / f"{input_path.stem}_structure_check.csv"
            structure_df = check_structure(
                predictions_df,
                export_csv_path=str(structure_csv) if structure_csv else None,
            )
            logger.info("Structuurcheck uitgevoerd.")

    if export_features:
        export_dir = Path(export_dir) if export_dir else Path("resultaat/csv")
        export_dir.mkdir(parents=True, exist_ok=True)

        base_name = input_path.stem
        paragraphs_csv = export_dir / f"{base_name}_paragraphs.csv"
        final_predictions_csv = export_dir / f"{base_name}_predictions_final.csv"
        structure_csv_final = export_dir / f"{base_name}_structure_check.csv"

        paragraphs_df.to_csv(paragraphs_csv, sep=";", index=False)

        if predictions_df is not None:
            merged_pred = _merge_columns(paragraphs_df, predictions_df, on="volgnummer")
            merged_pred.to_csv(final_predictions_csv, sep=";", index=False)
            if structure_df is not None:
                merged_struct = _merge_columns(merged_pred, structure_df[["volgnummer", "fout", "regel"]], on="volgnummer")
                merged_struct.to_csv(structure_csv_final, sep=";", index=False)

        logger.info("Export geschreven naar %s (paragraphs/predictions/structure)", export_dir)

    return predictions_df, structure_df


def setup_logging(log_dir: Optional[Path | str] = None) -> Path:
    """
    Configureer logging naar console én een daglogbestand in ./logging/.
    """
    log_dir = Path(log_dir) if log_dir else Path(__file__).resolve().parent / "logging"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"KOOP_pipeline_{datetime.now():%Y-%m-%d}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not any(isinstance(h, logging.FileHandler) and getattr(h, "_koop_logfile", None) == log_file for h in root.handlers):
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler._koop_logfile = log_file  # type: ignore[attr-defined]
        root.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    logger.info("Logging actief. Logbestand: %s", log_file)
    return log_file


def _merge_columns(base_df: pd.DataFrame, other_df: pd.DataFrame, *, on: str) -> pd.DataFrame:
    base = base_df.set_index(on)
    other = other_df.set_index(on)

    combined_index = base.index.union(other.index)
    base = base.reindex(combined_index)

    for column in other.columns:
        other_series = other[column]
        if column in base.columns:
            base[column] = base[column].combine_first(other_series)
        else:
            base[column] = other_series

    return base.reset_index().sort_values(on)

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Nieuwe KOOP-pipeline: stap 1 (document inlezen + features exporteren)."
    )
    parser.add_argument("input", help="Pad naar het Word-document (.doc of .docx).")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Sla het schrijven van CSV's over (handig voor snelle inspecties).",
    )
    parser.add_argument(
        "--export-dir",
        help="Doelmap voor CSV-export (default: resultaat/csv).",
    )
    parser.add_argument(
        "--no-predict",
        action="store_true",
        help="Sla de BERT-voorspellingen over.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Sla de AI-titel generatiestap over.",
    )
    parser.add_argument(
        "--no-structure",
        action="store_true",
        help="Sla de structuurcontrole over.",
    )
    args = parser.parse_args()

    log_file = setup_logging()
    logger.info("Start KOOP-pipeline run. Log: %s", log_file)

    process_document(
        args.input,
        export_features=not args.no_export,
        export_dir=args.export_dir,
        run_predictions=not args.no_predict,
        ensure_title_step=not args.no_title,
        run_structure_check=not args.no_structure,
    )


if __name__ == "__main__":
    main()
