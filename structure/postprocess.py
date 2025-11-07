from __future__ import annotations

import logging


import pandas as pd

logger = logging.getLogger(__name__)


STOP_PROFILES = {"OPBijlageTitel", "OPNotaToelichtingTitel"}
ALLOWED_WITHIN_SECTION = {"StandaardAlinea", "Lijstalinea", "OPBijlageTitel", "OPNotaToelichtingTitel", "OPOndertekening"}


def enforce_annex_nota_rules(
    df_in: pd.DataFrame,
    *,
    profile_col: str = "opmaakprofiel",
    fout_col: str = "fout",
    regel_col: str = "regel",
    volg_col: str = "volgnummer",
) -> pd.DataFrame:
    """
    Past deterministische regels toe na de eerste OPBijlageTitel/OPNotaToelichtingTitel:
      - Binnen bijlagen of toelichtingen mogen alleen StandaardAlinea/Lijstalinea voorkomen.
      - OPLid wordt Lijstalinea; overige profielen worden StandaardAlinea.
      - OPNotaToelichtingTitel mag maar één keer voorkomen.
      - OPOndertekening-blok mag maar één keer voorkomen; na afloop mogen alleen bijlage/notitie titels volgen.
    """
    if df_in is None or df_in.empty:
        return df_in

    df = df_in.copy()
    df.sort_values(by=volg_col, kind="stable", inplace=True)

    for col in (fout_col, regel_col):
        if col not in df.columns:
            df[col] = "" if col == regel_col else 0

    profiles = df[profile_col].astype(str).tolist()
    first_section_idx = next((idx for idx, profile in enumerate(profiles) if profile in STOP_PROFILES), None)
    logger.info("Postprocess correctie uitgevoerd.")
    if first_section_idx is None:
        return df

    nota_seen = any(profile == "OPNotaToelichtingTitel" for profile in profiles[:first_section_idx])
    signature_seen = any(profile == "OPOndertekening" for profile in profiles[:first_section_idx])
    signature_block_active = False
    in_section = any(profile in STOP_PROFILES for profile in profiles[:first_section_idx])

    def set_profile(idx: int, new_profile: str, message: str = "") -> None:
        df.at[idx, profile_col] = new_profile
        if message:
            df.at[idx, regel_col] = message
        profiles[idx] = new_profile

    for idx in range(first_section_idx, len(df)):
        profile = profiles[idx]

        if profile == "OPBijlageTitel":
            in_section = True
            signature_block_active = False
            continue

        if profile == "OPNotaToelichtingTitel":
            if nota_seen:
                set_profile(idx, "StandaardAlinea", "Auto: tweede OPNotaToelichtingTitel → StandaardAlinea.")
            else:
                nota_seen = True
                in_section = True
                signature_block_active = False
            continue

        if profile == "OPOndertekening":
            if signature_seen and not signature_block_active:
                set_profile(idx, "StandaardAlinea", "Auto: extra OPOndertekening → StandaardAlinea.")
            else:
                signature_seen = True
                signature_block_active = True
            continue

        if profile not in STOP_PROFILES:
            signature_block_active = False

        if signature_block_active and profile not in {"OPBijlageTitel", "OPNotaToelichtingTitel"}:
            set_profile(idx, "StandaardAlinea", "Auto: inhoud na OPOndertekening → StandaardAlinea.")
            continue

        if not in_section:
            continue

        if profile in {"StandaardAlinea", "Lijstalinea"}:
            continue

        if profile == "OPLid":
            set_profile(idx, "Lijstalinea", "Auto: OPLid binnen bijlage/toelichting → Lijstalinea.")
        else:
            set_profile(idx, "StandaardAlinea", f"Auto: {profile} binnen bijlage/toelichting → StandaardAlinea.")

    return df


def _set_profile(*args, **kwargs):
    raise NotImplementedError("Gebruik de lokale set_profile helper in enforce_annex_nota_rules.")
