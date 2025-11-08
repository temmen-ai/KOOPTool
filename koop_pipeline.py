from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from document_io import paragraphs_to_dicts, read_document
from llm.corrections import apply_corrections_by_windows
from llm.chatgpt_corrector import chatgpt_corrector
from ml.predictor import BertOpmaakprofielPredictor
from services.title_resolver import ensure_title
from structure.checker import check_structure
from structure.postprocess import enforce_annex_nota_rules
from structure.numbering import correct_numbering
from structure.tables_images import add_tables_and_images
from writers.doc_builder import build_word_document

logger = logging.getLogger(__name__)


def process_document(
    input_path: Path | str,
    *,
    export_features: bool = True,
    export_dir: Optional[Path | str] = None,
    doc_output_dir: Optional[Path | str] = None,
    run_predictions: bool = True,
    predictor: Optional[BertOpmaakprofielPredictor] = None,
    ensure_title_step: bool = True,
    run_structure_check: bool = True,
    run_gpt_corrections: bool = True,
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
    structure_df_before = None
    structure_df_final = None
    structure_snapshot_after_post = None
    doc_output_dir_path = Path(doc_output_dir) if doc_output_dir else Path("outputmap")
    paragraphs_df = pd.DataFrame(paragraphs_to_dicts(extraction.paragraphs))
    if run_predictions:
        predictor = predictor or BertOpmaakprofielPredictor()
        logger.info("BERT-model geladen, start sliding-window voorspellingen.")
        predictions_df = predictor.predict(extraction.paragraphs)
        logger.info("Voorspellingen voltooid voor %d paragrafen.", len(extraction.paragraphs))

        # Zorg dat alle downstream stappen (en CSV's) over dezelfde kolommen beschikken.
        predictions_df = _merge_columns(paragraphs_df, predictions_df, on="volgnummer")

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

        if run_structure_check:
            initial_structure_csv = None
            if export_features:
                export_dir_path = Path(export_dir) if export_dir else Path("resultaat/csv")
                initial_structure_csv = export_dir_path / f"{input_path.stem}_structure_check_initial.csv"
            structure_df_before = check_structure(
                predictions_df,
                export_csv_path=str(initial_structure_csv) if initial_structure_csv else None,
                reset_errors=True,
            )
            logger.info("Structuurcheck uitgevoerd (na BERT/titel).")

        if run_gpt_corrections:
            df_for_corrections = structure_df_before if structure_df_before is not None else predictions_df
            if df_for_corrections is not None and "fout" in df_for_corrections.columns:
                logger.info("Start ChatGPT-correcties voor foutieve vensters.")
                predictions_df = apply_corrections_by_windows(df_for_corrections, corrector=chatgpt_corrector)
                logger.info("ChatGPT-correcties afgerond.")
            else:
                logger.info("ChatGPT-correcties overgeslagen: geen foutinformatie beschikbaar.")
        if run_structure_check:
            after_gpt_csv = None
            if export_features:
                export_dir_path = Path(export_dir) if export_dir else Path("resultaat/csv")
                after_gpt_csv = export_dir_path / f"{input_path.stem}_structure_check_aftergpt.csv"
            structure_after_gpt = check_structure(
                predictions_df,
                export_csv_path=str(after_gpt_csv) if after_gpt_csv else None,
                reset_errors=True,
            )
            if structure_after_gpt is not None:
                for col in ("fout", "regel"):
                    if col in structure_after_gpt.columns and col in predictions_df.columns:
                        predictions_df[col] = structure_after_gpt[col]
            logger.info("Structuurcheck uitgevoerd (na GPT).")

        predictions_df = enforce_annex_nota_rules(predictions_df)

        if run_structure_check:
            final_structure_csv = None
            if export_features:
                export_dir_path = Path(export_dir) if export_dir else Path("resultaat/csv")
                final_structure_csv = export_dir_path / f"{input_path.stem}_structure_check_afterpostprocess.csv"
            structure_df_final = check_structure(
                predictions_df,
                export_csv_path=str(final_structure_csv) if final_structure_csv else None,
                reset_errors=True,
            )
            if structure_df_final is not None:
                for col in ("fout", "regel"):
                    if col in structure_df_final.columns and col in predictions_df.columns:
                        predictions_df[col] = structure_df_final[col]
            logger.info("Structuurcheck uitgevoerd (na volledige postprocess).")

        metadata_cols = ["num_properties", "bron_numId"]
        existing_cols = [c for c in metadata_cols if c in paragraphs_df.columns]
        if existing_cols:
            metadata_df = paragraphs_df[["volgnummer", *existing_cols]]
            predictions_df = _merge_columns(predictions_df, metadata_df, on="volgnummer")

        if structure_df_final is not None and structure_snapshot_after_post is None:
            structure_snapshot_after_post = predictions_df.copy()

        predictions_df = correct_numbering(predictions_df)
        predictions_df = add_tables_and_images(predictions_df, extraction.tables, extraction.images)

        output_document = doc_output_dir_path / f"inlaad_{Path(input_path).name}"
        comments_generated, comments_include_errors = build_word_document(
            predictions_df,
            input_document=input_path,
            output_document=output_document,
            base_name=input_path.stem,
        )
        logger.info("Word-document gegenereerd: %s", output_document)
        if comments_generated:
            logger.info("Commentaar toegevoegd voor afwijkingen.")
        if comments_include_errors:
            logger.info("LET OP: Er zijn fouten gemarkeerd in het commentaar.")

    if export_features:
        export_dir = Path(export_dir) if export_dir else Path("resultaat/csv")
        export_dir.mkdir(parents=True, exist_ok=True)

        base_name = input_path.stem
        paragraphs_csv = export_dir / f"{base_name}_paragraphs.csv"
        final_predictions_csv = export_dir / f"{base_name}_predictions_final.csv"
        structure_csv_final = export_dir / f"{base_name}_structure_check_afterpostprocess.csv"
        numbering_csv = export_dir / f"{base_name}_numbering.csv"

        paragraphs_df.to_csv(paragraphs_csv, sep=";", index=False)

        if predictions_df is not None:
            final_predictions_df = predictions_df.copy()
            final_predictions_df.to_csv(final_predictions_csv, sep=";", index=False)

            if structure_snapshot_after_post is not None:
                structure_snapshot_after_post.to_csv(structure_csv_final, sep=";", index=False)
            elif structure_df_final is not None:
                # Fallback: voeg fout/regel toe aan de huidige staat.
                merged_struct = _merge_columns(
                    final_predictions_df, structure_df_final[["volgnummer", "fout", "regel"]], on="volgnummer"
                )
                merged_struct.to_csv(structure_csv_final, sep=";", index=False)

            final_predictions_df.to_csv(numbering_csv, sep=";", index=False)

        logger.info("Export geschreven naar %s (paragraphs/predictions/structure)", export_dir)

    return predictions_df, structure_df_final


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

    combined = base.join(other, how="outer", lsuffix="", rsuffix="_new")

    for column in other.columns:
        new_col = f"{column}_new"
        if new_col in combined.columns and column in combined.columns:
            combined[column] = combined[column].combine_first(combined[new_col])
            combined.drop(columns=[new_col], inplace=True)
        elif new_col in combined.columns:
            combined.rename(columns={new_col: column}, inplace=True)

    result = combined.reset_index().sort_values(on)
    result = result.loc[:, ~result.columns.duplicated()]
    result = result.drop_duplicates(subset=[on], keep="last").sort_values(on)
    return result

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
        "--doc-output-dir",
        help="Doelmap voor Word-uitvoer (default: outputmap).",
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
    parser.add_argument(
        "--no-gpt",
        action="store_true",
        help="Sla de ChatGPT-correcties over.",
    )
    args = parser.parse_args()

    log_file = setup_logging()
    logger.info("Start KOOP-pipeline run. Log: %s", log_file)

    process_document(
        args.input,
        export_features=not args.no_export,
        export_dir=args.export_dir,
        doc_output_dir=args.doc_output_dir,
        run_predictions=not args.no_predict,
        ensure_title_step=not args.no_title,
        run_structure_check=not args.no_structure,
        run_gpt_corrections=not args.no_gpt,
    )


if __name__ == "__main__":
    main()
