from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

RULE_MESSAGES = {
    1: "Het eerste profiel moet 'OPTitel' zijn.",
    2: "'OPTitel' mag maar één keer voorkomen.",
    3: "Na 'OPTitel' komt altijd 'OPAanhef'.",
    4: "Na 'OPAanhef' kan 'OPAanhef', 'OPHoofdstukTitel', 'OPParagraafTitel' of 'OPArtikelTitel' volgen.",
    5: "Na 'OPHoofdstukTitel' kan 'OPParagraafTitel', 'OPArtikelTitel', 'OPLid', 'StandaardAlinea' of 'Lijstalinea' volgen.",
    6: "Na 'OPParagraafTitel' kan 'OPArtikelTitel' volgen.",
    7: "Na 'OPArtikelTitel' kan 'OPLid' of 'StandaardAlinea' volgen.",
    8: "Na 'OPLid' kunnen 'OPLid', 'OPHoofdstukTitel', 'OPParagraafTitel', 'OPArtikelTitel', 'OPOndertekening', 'OPBijlageTitel' of 'OPNotaToelichtingTitel' volgen.",
    9: "Na 'StandaardAlinea' kunnen 'StandaardAlinea', 'Lijstalinea', 'OPHoofdstukTitel', 'OPParagraafTitel', 'OPArtikelTitel', 'OPOndertekening', 'OPBijlageTitel' of 'OPNotaToelichtingTitel' volgen.",
    10: "Na 'Lijstalinea' kunnen 'StandaardAlinea', 'Lijstalinea', 'OPHoofdstukTitel', 'OPParagraafTitel', 'OPArtikelTitel', 'OPOndertekening', 'OPBijlageTitel' of 'OPNotaToelichtingTitel' volgen.",
    11: "Na 'OPOndertekening' kunnen 'OPOndertekening', 'OPBijlageTitel' of 'OPNotaToelichtingTitel' volgen.",
    12: "Na 'OPBijlageTitel' kunnen 'OPBijlageTitel', 'StandaardAlinea' of 'Lijstalinea' volgen.",
    13: "Als 'OPBijlageTitel' is geweest, dan mag daarna niet meer voorkomen: 'OPAanhef', 'OPHoofdstukTitel', 'OPParagraafTitel', 'OPArtikelTitel' of 'OPLid' voorkomen.",
    14: "Na 'OPNotaToelichtingTitel' kunnen 'StandaardAlinea' of 'Lijstalinea' volgen.",
    15: "Als 'OPNotaToelichtingTitel' is geweest, dan mag daarna niet meer voorkomen: 'OPAanhef', 'OPHoofdstukTitel', 'OPParagraafTitel', 'OPArtikelTitel', 'OPLid' of 'OPBijlageTitel' voorkomen.",
    16: "Het profiel 'OPNotaToelichtingTitel' mag maar één keer voorkomen.",
    17: "StandaardAlinea kan geen opsomming zijn.",
    18: "OPLid moet genummerde opsomming zijn, mogelijk moet dit 'StandaardAlinea' zijn.",
}

TRANSITION_RULES: Dict[str, Dict[str, List[str]]] = {
    "OPAanhef": {"allowed": ["OPAanhef", "OPHoofdstukTitel", "OPParagraafTitel", "OPArtikelTitel"], "rule": 4},
    "OPHoofdstukTitel": {
        "allowed": ["OPParagraafTitel", "OPArtikelTitel", "OPLid", "StandaardAlinea", "Lijstalinea"],
        "rule": 5,
    },
    "OPParagraafTitel": {"allowed": ["OPArtikelTitel"], "rule": 6},
    "OPArtikelTitel": {"allowed": ["OPLid", "StandaardAlinea"], "rule": 7},
    "OPLid": {
        "allowed": [
            "OPLid",
            "OPHoofdstukTitel",
            "OPParagraafTitel",
            "OPArtikelTitel",
            "OPOndertekening",
            "OPBijlageTitel",
            "OPNotaToelichtingTitel",
        ],
        "rule": 8,
    },
    "StandaardAlinea": {
        "allowed": [
            "StandaardAlinea",
            "Lijstalinea",
            "OPHoofdstukTitel",
            "OPParagraafTitel",
            "OPArtikelTitel",
            "OPOndertekening",
            "OPBijlageTitel",
            "OPNotaToelichtingTitel",
        ],
        "rule": 9,
    },
    "Lijstalinea": {
        "allowed": [
            "StandaardAlinea",
            "Lijstalinea",
            "OPHoofdstukTitel",
            "OPParagraafTitel",
            "OPArtikelTitel",
            "OPOndertekening",
            "OPBijlageTitel",
            "OPNotaToelichtingTitel",
        ],
        "rule": 10,
    },
    "OPOndertekening": {"allowed": ["OPOndertekening", "OPBijlageTitel", "OPNotaToelichtingTitel"], "rule": 11},
    "OPBijlageTitel": {"allowed": ["OPBijlageTitel", "StandaardAlinea", "Lijstalinea"], "rule": 12},
    "OPNotaToelichtingTitel": {"allowed": ["StandaardAlinea", "Lijstalinea"], "rule": 14},
}


def check_structure(
    df_in: pd.DataFrame,
    *,
    profile_col: str = "opmaakprofiel",
    number_col: str = "numbered",
    fout_col: str = "fout",
    regel_col: str = "regel",
    export_csv_path: Optional[str] = None,
    reset_errors: bool = True,
) -> pd.DataFrame:
    """
    Controleer de documentstructuur volgens de aangeleverde regels.
    Retourneert een kopie met kolommen `fout` en `regel` gevuld.
    """
    df = df_in.copy()

    if fout_col not in df.columns:
        df[fout_col] = 0
    if regel_col not in df.columns:
        df[regel_col] = ""

    if reset_errors:
        preserve_mask = df[fout_col] == 2
        df.loc[:, fout_col] = df[fout_col].where(preserve_mask, 0)
        df.loc[:, regel_col] = df[regel_col].where(preserve_mask, "")

    seen_optitel = 0
    seen_nota = False
    seen_bijlage = False

    prev_profile: Optional[str] = None

    for idx, row in df.iterrows():
        profile = str(row.get(profile_col, "")).strip()

        if idx == 0:
            if profile != "OPTitel":
                _flag(df, idx, fout_col, regel_col, 1)
            else:
                seen_optitel += 1
        else:
            if profile == "OPTitel":
                seen_optitel += 1
                if seen_optitel > 1:
                    _flag(df, idx, fout_col, regel_col, 2)

        if prev_profile == "OPTitel" and profile != "OPAanhef":
            _flag(df, idx, fout_col, regel_col, 3)

        if prev_profile in TRANSITION_RULES:
            allowed = TRANSITION_RULES[prev_profile]["allowed"]
            rule_num = TRANSITION_RULES[prev_profile]["rule"]
            if profile not in allowed:
                _flag(df, idx, fout_col, regel_col, rule_num)

        if profile == "OPNotaToelichtingTitel":
            if seen_nota:
                _flag(df, idx, fout_col, regel_col, 16)
            seen_nota = True

        if seen_bijlage and profile in {"OPAanhef", "OPHoofdstukTitel", "OPParagraafTitel", "OPArtikelTitel", "OPLid"}:
            _flag(df, idx, fout_col, regel_col, 13)

        if profile == "OPBijlageTitel":
            seen_bijlage = True

        if seen_nota and profile in {
            "OPAanhef",
            "OPHoofdstukTitel",
            "OPParagraafTitel",
            "OPArtikelTitel",
            "OPLid",
            "OPBijlageTitel",
        }:
            _flag(df, idx, fout_col, regel_col, 15)

        if profile == "StandaardAlinea":
            number_val = _to_int(row.get(number_col, row.get("genummerd", 0)))
            if number_val == 1:
                _flag(df, idx, fout_col, regel_col, 17)

        if profile == "OPLid":
            number_val = _to_int(row.get(number_col, row.get("genummerd", 0)))
            has_numid = bool(_to_int(row.get("numId")))
            has_num_props = bool(row.get("num_properties"))
            style = str(row.get("style_name", "")).strip()
            style_norm = style.lower()
            has_style = bool(style and style_norm not in {"normal", "standaard"})  # treat default styles als geen signaal
            level_val = str(row.get("niveau", "")).strip()
            if level_val.lower() in {"", "nan", "none"}:
                level_val = None
            leading = _has_leading_ops_signal(row)
            if number_val == 0 and not has_numid and not has_num_props and not has_style and not level_val and not leading:
                _flag(df, idx, fout_col, regel_col, 18)

        prev_profile = profile

    if export_csv_path:
        df.to_csv(export_csv_path, sep=";", index=False)

    return df


def _flag(df: pd.DataFrame, idx: int, fout_col: str, regel_col: str, rule_number: int) -> None:
    if df.at[idx, fout_col] == 2:
        return
    df.at[idx, fout_col] = 1
    df.at[idx, regel_col] = f"{rule_number}: {RULE_MESSAGES[rule_number]}"


def _to_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _has_leading_ops_signal(row: pd.Series) -> bool:
    patterns = ("leading_text", "leading_run_text")
    for column in patterns:
        raw = str(row.get(column, "")).strip()
        if not raw:
            continue
        if _looks_like_number(raw) or _looks_like_letter(raw):
            return True
    return False


def _looks_like_number(text: str) -> bool:
    normalized = text.rstrip(". )")
    return normalized.isdigit()


LOWER_ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}
UPPER_ROMAN = {val.upper() for val in LOWER_ROMAN}


def _looks_like_letter(text: str) -> bool:
    normalized = text.rstrip(". )")
    if len(normalized) == 1 and normalized.isalpha():
        return True
    return normalized.lower() in LOWER_ROMAN or normalized.upper() in UPPER_ROMAN
