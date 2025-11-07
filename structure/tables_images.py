from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from document_io import ImageInfo, TableInfo


def add_tables_and_images(
    df_in: pd.DataFrame,
    tables: Iterable[TableInfo],
    images: Iterable[ImageInfo],
) -> pd.DataFrame:
    combined: List[dict] = []
    for table in tables:
        combined.append(
            {
                "volgnummer": table.volgnummer,
                "tekst": table.tekst,
                "regel": "Kopieer tabel uit inputbestand en maak eventueel opnieuw op.",
            }
        )
    for image in images:
        combined.append(
            {
                "volgnummer": image.volgnummer,
                "tekst": image.tekst,
                "regel": f"Kopieer afbeelding uit inputbestand ({image.naam or 'afbeelding'}).",
            }
        )

    if not combined:
        return df_in

    df = df_in.copy()
    df["volgnummer"] = pd.to_numeric(df["volgnummer"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("volgnummer", kind="stable").reset_index(drop=True)
    combined.sort(key=lambda x: x["volgnummer"])

    default_values = {
        "indented": 0,
        "indentation": 0,
        "bold": 0,
        "italic": 0,
        "underlined": 0,
        "numbered": 0,
        "niveau": "None",
        "numId": "None",
        "opmaakprofiel": "StandaardAlinea",
        "confidence": 0.0,
        "textpartformat_u": "Nee",
        "textpartformat_b": "Nee",
        "textpartformat_i": "Nee",
        "textpartformats_u": [],
        "textpartformats_b": [],
        "textpartformats_i": [],
        "textparts": [],
        "fout": 2,
        "opsommingstype": "",
    }

    for info in combined:
        target = int(info["volgnummer"])
        greater_equal = df.index[df["volgnummer"] >= target]
        insert_idx = int(greater_equal.min()) if not greater_equal.empty else len(df)
        df.loc[df["volgnummer"] >= target, "volgnummer"] += 1

        row_data = {col: default_values.get(col, None) for col in df.columns}
        row_data.update({"volgnummer": target, "tekst": info["tekst"], "regel": info["regel"]})
        new_row = pd.DataFrame([row_data])
        df = pd.concat([df.iloc[:insert_idx], new_row, df.iloc[insert_idx:]], ignore_index=True)

    return df.sort_values("volgnummer", kind="stable").reset_index(drop=True)
