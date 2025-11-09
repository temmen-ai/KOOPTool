from __future__ import annotations

import pandas as pd

def _detect_case1_merge_targets(df: pd.DataFrame) -> set:
    PROFILES = {"OPLid", "Lijstalinea"}

    def _valid_numid(value) -> bool:
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        return text not in {"", "none", "nan"}

    targets: set = set()
    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        if row.get("opmaakprofiel") not in PROFILES:
            continue
        if _valid_numid(row.get("numId")):
            continue
        prev = df.iloc[i - 1]
        nxt = df.iloc[i + 1]
        if (
            prev.get("opmaakprofiel") in PROFILES
            and nxt.get("opmaakprofiel") in PROFILES
            and _valid_numid(prev.get("numId"))
            and str(prev.get("numId")) == str(nxt.get("numId"))
        ):
            targets.add(prev.get("volgnummer"))
    return targets
