#!/usr/bin/env python3
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter
import warnings
import contextlib, io

LOG_BASE_DIR = Path(__file__).resolve().parent / "logging"

def setup_logging() -> Path:
    """
    Initialiseer logging héél vroeg:
    - log naar ./logging/KOOP_process_YYYY-MM-DD.log
    - append modus (één log per dag)
    - stderr naar zelfde logbestand
    - ongehanteerde exceptions in de log
    """
    LOG_BASE_DIR.mkdir(parents=True, exist_ok=True)

    date_stamp = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_BASE_DIR / f"KOOP_process_{date_stamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Stuur ook sys.stderr naar hetzelfde logbestand (line-buffered)
    sys.stderr = open(log_file, "a", encoding="utf-8", buffering=1)

    # Ongehanteerde exceptions met traceback loggen
    def _excepthook(exctype, value, tb):
        logging.critical("Onbekende fout", exc_info=(exctype, value, tb))
        try:
            import traceback
            traceback.print_exception(exctype, value, tb, file=sys.__stderr__)
        except Exception:
            pass
    sys.excepthook = _excepthook

    logging.info("=" * 72)
    logging.info("Nieuwe batch-run gestart")
    logging.info(f"Logbestand: {log_file}")
    return log_file


@contextlib.contextmanager
def suppress_import_output():
    """Onderdrukt stdout/stderr tijdens imports die hard printen."""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield
    out, err = buf_out.getvalue(), buf_err.getvalue()
    if out:
        logging.debug("Suppressed import stdout:\n" + out)
    if err:
        logging.debug("Suppressed import stderr:\n" + err)


def quiet_libraries():
    """Dempt veelvoorkomende ruis van libs (transformers/torch/vision)."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # Python warnings filteren (gericht)
    warnings.filterwarnings(
        "ignore",
        message="Some weights of .* were not initialized from the model checkpoint",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Failed to load image Python extension",
        category=UserWarning,
        module="torchvision.io.image",
    )

    # Library-loggers stillen
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)


def process_file(input_path: Path, output_dir: Path, verwerk_bestand) -> bool:
    """
    Verwerk één bestand.
    - Retourneert True bij succes, False bij fout.
    - Logt tijdsduur en eventuele fout met traceback.
    """
    start = perf_counter()
    logging.info(f"Start: {input_path.name}")
    try:
        modelkeuze = 'BERT'
        verwerk_bestand(input_path, output_dir, modelkeuze)
        elapsed = perf_counter() - start
        logging.info(f"Gereed: {input_path.name} (duur: {elapsed:.2f}s)")
        return True
    except Exception:
        elapsed = perf_counter() - start
        logging.exception(f"FOUT bij {input_path.name} (na {elapsed:.2f}s)")
        return False


def main():
    # === 1) Heel vroeg logging aanzetten ===
    log_file = setup_logging()

    # === 2) Ruis van libs dempen vóór imports ===
    quiet_libraries()

    # === 3) Nu pas zware imports doen (alles gaat naar de log, niet naar shell) ===
    with suppress_import_output():
        from DaadkrachtBatch import verwerk_bestand

    # === 4) CLI-argumenten verwerken ===
    if len(sys.argv) != 3:
        print("Gebruik: python KOOP_batch.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory bestaat niet of is geen map: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Verzamel bestanden (.docx en .doc) — geen subdirectories
    files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".docx", ".doc")
    ]
    logging.info(f"Gevonden: {len(files)} bestanden in: {input_dir}")

    # === 5) Verwerken ===
    t0 = perf_counter()
    ok = fail = 0

    for f in files:
        if process_file(f, output_dir, verwerk_bestand):
            ok += 1
        else:
            fail += 1  # ga door met volgende bestand

    total_elapsed = perf_counter() - t0
    logging.info("-" * 72)
    logging.info(
        f"Samenvatting: ok={ok}, fout={fail}, totaal={ok+fail}, "
        f"totale duur={total_elapsed:.2f}s"
    )
    logging.info(f"Log opgeslagen in: {log_file}")
    logging.info("=" * 72 + "\n")


if __name__ == "__main__":
    main()
