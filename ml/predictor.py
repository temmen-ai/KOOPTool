from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import BertConfig, BertTokenizer

from DaadkrachtBERT import BERTFineTunerWithFeatures
from document_io import ParagraphFeatures

logger = logging.getLogger(__name__)

LABEL_MAPPING = {
    0: "OPTitel",
    1: "OPAanhef",
    2: "OPHoofdstukTitel",
    3: "OPParagraafTitel",
    4: "OPArtikelTitel",
    5: "OPLid",
    6: "StandaardAlinea",
    7: "Lijstalinea",
    8: "OPBijlageTitel",
    9: "OPOndertekening",
    10: "OPNotaToelichtingTitel",
}

WEIGHT_SCHEMES: Dict[int, List[float]] = {
    5: [0.1, 0.1, 0.1, 0.4, 0.3],
    4: [0.1, 0.1, 0.3, 0.5],
    3: [0.2, 0.4, 0.4],
    2: [0.5, 0.5],
    1: [1.0],
}


class BertOpmaakprofielPredictor:
    """Wraps the fine-tuned BERT model and sliding-window inference logic."""

    def __init__(self, model_dir: str | Path = "finetuned_BERTmodel2_ronde2", window_size: int = 5):
        self.window_size = window_size
        self.model_dir = Path(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.config = BertConfig.from_pretrained(self.model_dir, local_files_only=True)
        self.model, self.device = self._load_model()

    def predict(self, paragraphs: Sequence[ParagraphFeatures]) -> pd.DataFrame:
        if not paragraphs:
            return pd.DataFrame(
                columns=[
                    "volgnummer",
                    "tekst",
                    "indented",
                    "indentation",
                    "bold",
                    "italic",
                    "underlined",
                    "numbered",
                    "niveau",
                    "numId",
                    "opmaakprofiel",
                    "confidence",
                    "textpartformat_u",
                    "textpartformat_b",
                    "textpartformat_i",
                    "textpartformats_u",
                    "textpartformats_b",
                    "textpartformats_i",
                    "textparts",
                ]
            )

        predictions_per_sentence: Dict[int, List[Tuple[np.ndarray, int]]] = {
            p.volgnummer: [] for p in paragraphs
        }

        effective_window = min(self.window_size, len(paragraphs))
        if effective_window <= 0:
            effective_window = len(paragraphs)

        total_iterations = len(paragraphs) - effective_window + 1
        for start_idx in range(total_iterations):
            window = paragraphs[start_idx : start_idx + effective_window]
            combined_text = f" {self.tokenizer.sep_token} ".join(p.tekst for p in window)
            tokenized = self.tokenizer(
                combined_text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            features = {
                "bold_features": torch.tensor([[p.bold for p in window]], dtype=torch.float32),
                "indented_features": torch.tensor([[p.indented for p in window]], dtype=torch.float32),
                "italic_features": torch.tensor([[p.italic for p in window]], dtype=torch.float32),
                "underlined_features": torch.tensor([[p.underlined for p in window]], dtype=torch.float32),
                "numbered_features": torch.tensor([[p.genummerd for p in window]], dtype=torch.float32),
                "volgnummer_features": torch.tensor([[p.volgnummer for p in window]], dtype=torch.float32),
            }
            features = {k: v.to(self.device) for k, v in features.items()}

            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **features,
                )

            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            for i, logit in enumerate(logits):
                probabilities = torch.softmax(logit, dim=-1).cpu().numpy()
                sentence_index = start_idx + i + 1
                predictions_per_sentence[sentence_index].append((probabilities, i))

        final_predictions = self._aggregate_predictions(predictions_per_sentence)

        records = []
        for paragraph in paragraphs:
            pred_idx, confidence = final_predictions.get(paragraph.volgnummer, (None, 0.0))
            if pred_idx is None:
                pred_idx = 6  # fallback to StandaardAlinea
            opmaakprofiel = LABEL_MAPPING.get(pred_idx, "StandaardAlinea")

            record = {
                "volgnummer": paragraph.volgnummer,
                "tekst": paragraph.tekst,
                "indented": paragraph.indented,
                "indentation": paragraph.indentation,
                "bold": paragraph.bold,
                "italic": paragraph.italic,
                "underlined": paragraph.underlined,
                "numbered": paragraph.genummerd,
                "niveau": paragraph.niveau,
                "numId": paragraph.numId,
                "opmaakprofiel": opmaakprofiel,
                "confidence": confidence,
                "textpartformat_u": paragraph.textpartformat_u,
                "textpartformat_b": paragraph.textpartformat_b,
                "textpartformat_i": paragraph.textpartformat_i,
                "textpartformats_u": paragraph.textpartformats_u,
                "textpartformats_b": paragraph.textpartformats_b,
                "textpartformats_i": paragraph.textpartformats_i,
                "textparts": paragraph.textparts,
            }
            records.append(record)

        return pd.DataFrame(records)

    def _load_model(self):
        model = BERTFineTunerWithFeatures(hidden_size=self.config.hidden_size, num_labels=len(LABEL_MAPPING), tokenizer=self.tokenizer)
        state_path = self.model_dir / "best_model_weights.pth"
        model.load_state_dict(torch.load(state_path, map_location=torch.device("cpu")))

        device = self._select_device()
        model.to(device)
        model.eval()
        return model, device

    def _select_device(self) -> torch.device:
        system = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
        if torch.cuda.is_available():
            return torch.device("cuda")
        if system:
            return torch.device("mps")
        return torch.device("cpu")

    def _aggregate_predictions(self, predictions_per_sentence: Dict[int, List[Tuple[np.ndarray, int]]]):
        final_predictions: Dict[int, Tuple[int, float]] = {}
        for sentence_index, prob_list in predictions_per_sentence.items():
            if not prob_list:
                continue
            weight_scheme = WEIGHT_SCHEMES.get(len(prob_list), WEIGHT_SCHEMES[max(WEIGHT_SCHEMES.keys())])
            weights = np.array(weight_scheme[: len(prob_list)])
            weighted_probs = np.zeros_like(prob_list[0][0])

            for weight, (probabilities, _) in zip(weights, prob_list):
                weighted_probs += probabilities * weight

            weighted_probs /= weights.sum()
            final_pred = int(np.argmax(weighted_probs))
            confidence = float(weighted_probs[final_pred])
            final_predictions[sentence_index] = (final_pred, confidence)
        return final_predictions
