from __future__ import annotations

import os
from typing import Optional, Tuple

import pandas as pd
from openai import OpenAI

from config.model_registry import get_model_id


def ensure_title(
    predictions_df: pd.DataFrame,
    *,
    export_csv_path: Optional[str] = None,
    client: Optional[OpenAI] = None,
    model_key: str = "title_generation",
    model_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, bool]:
    """
    Zorgt ervoor dat er een OPTitel aanwezig is. Als de voorspellingen er geen hebben,
    roept deze functie een finetuned OpenAI-model aan om een titel te genereren en
    voegt die als volgnummer 0 toe.
    """
    df = predictions_df.copy()
    df.sort_values("volgnummer", kind="stable", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) > 0 and df.loc[0, "opmaakprofiel"] == "OPTitel":
        if export_csv_path:
            df.to_csv(export_csv_path, sep=";", index=False)
        return df, False

    document_text = " ".join(df.get("tekst", "").astype(str).tolist())

    if model_id is None:
        model_id = get_model_id(model_key)

    client = client or OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "Het model beantwoordt vragen over gemeentelijke verordeningen."},
            {"role": "user", "content": f"Document tekst: '{document_text}'\nVraag: Wat is de titel van deze verordening?"},
        ],
    )
    generated_title = completion.choices[0].message.content.strip()

    new_row = {
        "volgnummer": 0,
        "tekst": generated_title,
        "indented": 0,
        "indentation": 0,
        "bold": 0,
        "italic": 0,
        "underlined": 0,
        "numbered": 0,
        "niveau": "None",
        "numId": "None",
        "opmaakprofiel": "OPTitel",
        "confidence": 1.0,
        "textpartformat_u": "Nee",
        "textpartformat_b": "Nee",
        "textpartformat_i": "Nee",
        "textpartformats_u": [],
        "textpartformats_b": [],
        "textpartformats_i": [],
        "textparts": [],
    }

    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
    if export_csv_path:
        df.to_csv(export_csv_path, sep=";", index=False)
    return df, True
