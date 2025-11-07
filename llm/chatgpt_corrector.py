from __future__ import annotations

import json
import os
from typing import List, Optional

import pandas as pd
from openai import OpenAI

from config.model_registry import get_model_id

ALLOWED_LABELS = [
    "OPTitel",
    "OPAanhef",
    "OPHoofdstukTitel",
    "OPParagraafTitel",
    "OPArtikelTitel",
    "OPLid",
    "StandaardAlinea",
    "Lijstalinea",
    "OPBijlageTitel",
    "OPOndertekening",
    "OPNotaToelichtingTitel",
]

RULES_TEXT = (
    "Je krijgt zinnen van gemeentelijke documenten.\n"
    "Elke regel heeft een voorgesteld opmaakprofiel.\n"
    "Het fragment bevat N regels (gegeven in het veld 'num_rows').\n"
    "Geef uitsluitend JSON terug met sleutel 'y': een lijst van N profielen,\n"
    "in dezelfde volgorde als de aangeleverde regels.\n"
    "BELANGRIJK: Je geeft altijd exact num_rows profielen terug,\n"
    "en uitsluitend in geldig JSON-formaat."
)


def chatgpt_corrector(
    sub_df: pd.DataFrame,
    *,
    model_key: str = "profile_correction",
    model_id: Optional[str] = None,
    temperature: float = 0.0,
) -> List[str]:
    api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
    model_id = model_id or get_model_id(model_key)

    include_cols = [
        "volgnummer",
        "tekst",
        "indented",
        "numbered",
        "niveau",
        "numId",
        "opmaakprofiel",
        "confidence",
        "fout",
        "regel",
    ]
    rows = sub_df[include_cols].to_dict(orient="records")
    payload = {"num_rows": len(rows), "rows": rows}

    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": RULES_TEXT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=temperature,
    )

    content = resp.choices[0].message.content
    parsed = json.loads(content)
    y = parsed.get("y") or parsed.get("opmaakprofielen")
    if not isinstance(y, list):
        raise RuntimeError("Modelantwoord mist sleutel 'y'.")

    n = len(rows)
    if len(y) < n:
        y = y + ["StandaardAlinea"] * (n - len(y))
    elif len(y) > n:
        y = y[:n]

    y_fixed = [lab if lab in ALLOWED_LABELS else "StandaardAlinea" for lab in y]
    return y_fixed
