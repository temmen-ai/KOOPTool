#!/usr/bin/env python3
import os
import sys
import io
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter
import contextlib

import streamlit as st


# ============ Noise & Logging Setup (run-safe for Streamlit) ============
LOG_DIR = Path(__file__).resolve().parent / "logging"
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "base_dirs.json"

def init_logging_and_noise_once() -> Path:
    """
    Stel één keer per sessie logging en 'ruis-demping' in:
    - Dag-logbestand in ./logging/KOOP_process_YYYY-MM-DD.log
    - sys.stderr -> logbestand (import-time prints/warnings verdwijnen uit shell)
    - Gerichte warning-filters (pandas/torchvision)
    - Library-logger levels omlaag (torch/transformers/elastic)
    """
    if st.session_state.get("log_noise_initialized"):
        return st.session_state["log_file"]

    # 1) Logmap en logbestand (daglog)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    date_stamp = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"KOOP_process_{date_stamp}.log"
    if not log_file.exists():
        log_file.touch()

    # 2) Root logger -> alleen FileHandler
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(fh)

    # 3) Stuur stderr ook naar log (alleen 1x per sessie)
    #    Bewaar de file-handle in session_state zodat hij niet GC'd wordt
    st.session_state["stderr_fp"] = open(log_file, "a", encoding="utf-8", buffering=1)
    sys.stderr = st.session_state["stderr_fp"]

    # 4) Env-vars voor minder ruis (mag elke run)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # 5) Warnings filteren (gericht)
    warnings.filterwarnings(
        "ignore",
        message="Pandas requires version '2.8.4' or newer of 'numexpr'",
        category=UserWarning,
        module="pandas.core.computation.expressions",
    )
    warnings.filterwarnings(
        "ignore",
        message="Pandas requires version '1.3.6' or newer of 'bottleneck'",
        category=UserWarning,
        module="pandas.core.arrays.masked",
    )
    warnings.filterwarnings(
        "ignore",
        message="Failed to load image Python extension",
        category=UserWarning,
        module="torchvision.io.image",
    )

    # 6) Library loggers stiller zetten
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    # 7) Ongehanteerde exceptions ook in log
    def _excepthook(exctype, value, tb):
        logging.critical("Ongehanteerde uitzondering", exc_info=(exctype, value, tb))
        try:
            import traceback
            traceback.print_exception(exctype, value, tb, file=sys.__stderr__)
        except Exception:
            pass
    sys.excepthook = _excepthook

    logging.info("=" * 72)
    logging.info("Nieuwe UI-run gestart")
    logging.info(f"Logbestand: {log_file}")

    st.session_state["log_noise_initialized"] = True
    st.session_state["log_file"] = log_file
    return log_file


@contextlib.contextmanager
def suppress_import_output():
    """Onderdrukt stdout/stderr tijdens zware imports."""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield
    if buf_out.getvalue():
        logging.debug("Onderdrukte import stdout:\n%s", buf_out.getvalue())
    if buf_err.getvalue():
        logging.debug("Onderdrukte import stderr:\n%s", buf_err.getvalue())


def quiet_libraries():
    """Dempt ruis van veelgebruikte libraries."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)


def load_pipeline_once():
    """Laad de nieuwe KOOP-pipeline maximaal één keer per sessie."""
    if st.session_state.get("process_document"):
        return st.session_state["process_document"]

    quiet_libraries()
    with suppress_import_output():
        from koop_pipeline import process_document

    st.session_state["process_document"] = process_document
    return process_document


DEFAULT_BASE_DIRS = {
    "basedirectory_input": "",
    "basedirectory_output": "",
}


def load_base_dirs() -> dict:
    """Lees de base directories uit config/base_dirs.json (met caching)."""
    if st.session_state.get("base_dirs"):
        return st.session_state["base_dirs"]

    data = DEFAULT_BASE_DIRS.copy()
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
            if isinstance(raw, dict):
                for key in data:
                    value = raw.get(key, "")
                    data[key] = str(value) if value is not None else ""
    except FileNotFoundError:
        logging.warning("Configuratiebestand %s niet gevonden; gebruik defaults.", CONFIG_PATH)
    except json.JSONDecodeError as exc:
        logging.warning("Configuratiebestand %s ongeldig (%s); gebruik defaults.", CONFIG_PATH, exc)

    st.session_state["base_dirs"] = data
    return data


def resolve_with_base(base_dir_str: str, user_value: str) -> Path:
    """Combineer gebruikersinput met de ingestelde base directory."""
    base = Path(base_dir_str or ".").expanduser()
    user_path = Path(user_value).expanduser() if user_value else Path(".")
    if user_path.is_absolute():
        return user_path
    return (base / user_path).resolve()


# ============================ UI ============================
st.set_page_config(page_title="KOOP batch UI", layout="wide")
st.title("📂 KOOP batchverwerking")

# Logging & noise: heel vroeg initialiseren
log_file = init_logging_and_noise_once()
base_dirs = load_base_dirs()
input_base_display = str(resolve_with_base(base_dirs["basedirectory_input"], ""))
output_base_display = str(resolve_with_base(base_dirs["basedirectory_output"], ""))

with st.sidebar:
    st.header("Instellingen")
    st.caption(f"Basis input: {input_base_display}")
    input_dir_str = st.text_input("Inputmap (geen submappen)", value="")
    st.caption(f"Basis output: {output_base_display}")
    output_dir_str = st.text_input("Outputmap", value="")
    start_btn = st.button("🚀 Start verwerking")

status = st.empty()
progress = st.progress(0, text="Nog niet gestart")
table_placeholder = st.empty()

# ========================= Verwerking =========================
if start_btn:
    input_dir = resolve_with_base(base_dirs["basedirectory_input"], input_dir_str)
    output_dir = resolve_with_base(base_dirs["basedirectory_output"], output_dir_str)

    if not input_dir.exists() or not input_dir.is_dir():
        st.error("Inputmap bestaat niet of is geen map.")
        st.stop()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_dir.iterdir()
             if f.is_file() and f.suffix.lower() in (".docx", ".doc")]

    total = len(files)
    if total == 0:
        st.warning("Geen .docx of .doc bestanden gevonden.")
        st.stop()

    process_document = load_pipeline_once()
    modelkeuze = "Nieuwe pipeline"
    logging.info(
        f"Start batch vanuit UI: {total} bestand(en) | input={input_dir} | "
        f"output={output_dir} | model={modelkeuze}"
    )
    status.info(f"{total} bestand(en) gevonden. Verwerken gestart…")

    results = []
    ok = fail = 0
    t0 = perf_counter()

    export_dir = Path("resultaat/csv")
    for i, fpath in enumerate(files, start=1):
        start = perf_counter()
        try:
            process_document(
                fpath,
                export_dir=export_dir,
                doc_output_dir=output_dir,
            )
            dt = perf_counter() - start
            logging.info(f"OK: {fpath.name} (duur: {dt:.2f}s)")
            results.append({"bestand": fpath.name, "status": "OK", "duur_s": round(dt, 2)})
            ok += 1
        except Exception as e:
            dt = perf_counter() - start
            logging.exception(f"FOUT bij {fpath.name} (na {dt:.2f}s): {e}")
            results.append({"bestand": fpath.name, "status": f"FOUT: {e}", "duur_s": round(dt, 2)})
            fail += 1

        progress.progress(i / total, text=f"{i}/{total} verwerkt…")
        table_placeholder.dataframe(results, use_container_width=True)

    total_dt = perf_counter() - t0
    logging.info("-" * 72)
    logging.info(f"Samenvatting (UI): ok={ok}, fout={fail}, totaal={total}, duur={total_dt:.2f}s")
    logging.info(f"Log opgeslagen in: {log_file}")
    logging.info("=" * 72 + "\n")

    status.success(f"Klaar: {ok} OK, {fail} fout. Totale duur: {total_dt:.2f}s")

    # Downloads
    import csv
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["bestand", "status", "duur_s"])
    writer.writeheader()
    writer.writerows(results)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download log",
            data=Path(log_file).read_bytes(),
            file_name=Path(log_file).name,
            mime="text/plain"
        )
    with col2:
        st.download_button(
            "📥 Download overzicht (CSV)",
            data=buf.getvalue(),
            file_name="ui_batch_overzicht.csv",
            mime="text/csv"
        )
