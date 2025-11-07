from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

SECTION_TITLES = {"OPHoofdstukTitel", "OPParagraafTitel", "OPArtikelTitel"}
PREV_ANCHORS = {"OPArtikelTitel", "OPHoofdstukTitel"}
NEXT_ANCHORS = {
    "OPArtikelTitel",
    "OPHoofdstukTitel",
    "OPParagraafTitel",
    "OPOndertekening",
    "OPBijlageTitel",
    "OPNotaToelichtingTitel",
}
TERMINALS = {"OPOndertekening", "OPNotaToelichtingTitel", "OPBijlageTitel"}
STOP_PROFILES = {"OPBijlageTitel", "OPNotaToelichtingTitel"}


def apply_corrections_by_windows(
    df_in: pd.DataFrame,
    *,
    selector: Optional[Callable[[pd.DataFrame], List[dict]]] = None,
    corrector: Optional[Callable[[pd.DataFrame], List[str]]] = None,
    order_col: str = "volgnummer",
    profile_col: str = "opmaakprofiel",
    error_col: str = "fout",
    orig_col: str = "opmaakprofiel_BERT",
    inplace: bool = False,
) -> pd.DataFrame:
    df = df_in if inplace else df_in.copy()

    for col in (order_col, profile_col, error_col):
        if col not in df.columns:
            raise ValueError(f"Verwachte kolom '{col}' ontbreekt.")

    df[order_col] = pd.to_numeric(df[order_col], errors="coerce")
    df.sort_values(by=order_col, kind="stable", inplace=True)

    if orig_col not in df.columns:
        df[orig_col] = df[profile_col]
    else:
        df[orig_col] = df[orig_col].fillna(df[profile_col])

    if selector is None:
        selector = select_chatgpt_windows

    if corrector is None:
        def corrector(sub_df: pd.DataFrame) -> List[str]:
            return sub_df[profile_col].astype(str).tolist()

    windows = selector(df, order_col=order_col, profile_col=profile_col, error_col=error_col)

    for window in windows:
        start_v = window["start_volgnummer"]
        end_v = window["end_volgnummer"]
        if start_v is None or end_v is None:
            continue

        mask = (df[order_col] >= start_v) & (df[order_col] <= end_v)
        sub = df.loc[mask].copy().sort_values(by=order_col, kind="stable")
        idx = sub.index
        corrected = corrector(sub)
        if not isinstance(corrected, list) or len(corrected) != len(sub):
            corrected = sub[profile_col].astype(str).tolist()
        df.loc[idx, profile_col] = corrected

    return df


def select_chatgpt_windows(
    df: pd.DataFrame,
    *,
    order_col: str = "volgnummer",
    profile_col: str = "opmaakprofiel",
    error_col: str = "fout",
    window_after_terminal: int = 5,
) -> List[Dict]:
    if order_col not in df.columns or profile_col not in df.columns or error_col not in df.columns:
        raise ValueError("Vereiste kolommen ontbreken.")

    work = df.copy()
    work[order_col] = pd.to_numeric(work[order_col], errors="coerce")
    work = work.sort_values(by=order_col, kind="stable").reset_index(drop=True)

    n = len(work)
    if n == 0:
        return []

    volg = work[order_col].tolist()
    prof = work[profile_col].astype(str).tolist()
    err = work[error_col].fillna(0).astype(int).tolist()

    first_stop_idx: Optional[int] = next((i for i, p in enumerate(prof) if p in STOP_PROFILES), None)

    error_idxs = [i for i, v in enumerate(err) if v == 1 and (first_stop_idx is None or i < first_stop_idx)]
    if not error_idxs:
        return []

    first_section_idx: Optional[int] = next((i for i, p in enumerate(prof) if p in SECTION_TITLES), None)
    first_terminal_idx: Optional[int] = next((i for i, p in enumerate(prof) if p in TERMINALS), None)

    intervals: List[Tuple[int, int]] = []
    reasons: List[str] = []

    if first_section_idx is not None:
        if any(i <= first_section_idx for i in error_idxs):
            intervals.append((0, first_section_idx))
            reasons.append("A")
    else:
        intervals.append((0, n - 1))
        reasons.append("A-no-section")

    for e_idx in error_idxs:
        if (first_terminal_idx is not None) and (e_idx >= first_terminal_idx):
            start_i = max(0, e_idx - window_after_terminal)
            end_i = min(n - 1, e_idx + window_after_terminal)
            intervals.append((start_i, end_i))
            reasons.append("B-terminal±5")
            continue

        after_first_section = (first_section_idx is not None) and (e_idx > first_section_idx)
        if not after_first_section:
            continue

        prev_i = next((i for i in range(e_idx, -1, -1) if prof[i] in PREV_ANCHORS), 0)
        next_i = next((i for i in range(e_idx + 1, n) if prof[i] in NEXT_ANCHORS), n - 1)

        intervals.append((prev_i, next_i))
        reasons.append("B-prevnext")

    out = []

    for (s, e), r in zip(intervals, reasons):
        if r != "B-terminal±5":
            covered_errs = [i for i in error_idxs if s <= i <= e]
            out.append(
                {
                    "start_idx": s,
                    "end_idx": e,
                    "start_volgnummer": int(volg[s]) if pd.notnull(volg[s]) else None,
                    "end_volgnummer": int(volg[e]) if pd.notnull(volg[e]) else None,
                    "reason": r,
                    "covered_error_rows": covered_errs,
                }
            )

    terminal_intervals = [(s, e) for (s, e), r in zip(intervals, reasons) if r == "B-terminal±5"]
    merged_terminals = _merge_intervals(terminal_intervals)
    for (s, e) in merged_terminals:
        covered_errs = [i for i in error_idxs if s <= i <= e]
        out.append(
            {
                "start_idx": s,
                "end_idx": e,
                "start_volgnummer": int(volg[s]) if pd.notnull(volg[s]) else None,
                "end_volgnummer": int(volg[e]) if pd.notnull(volg[e]) else None,
                "reason": "C-merged",
                "covered_error_rows": covered_errs,
            }
        )

    return out


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + 1:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged
