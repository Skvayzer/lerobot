#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch


def load_labels_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix.lower() == ".parquet":
        return pd.read_parquet(table_path)
    if table_path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with open(table_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported labels table format: {table_path.suffix}")


def build_index_to_value(labels_df: pd.DataFrame, value_col: str) -> dict[int, float]:
    if "index" not in labels_df.columns:
        raise ValueError("Labels table must contain an 'index' column.")
    if value_col not in labels_df.columns:
        raise ValueError(f"Labels table is missing required column '{value_col}'.")
    subset = labels_df[["index", value_col]].dropna()
    return {int(idx): float(val) for idx, val in subset.to_numpy()}


def gather_values_by_index(
    index_tensor: torch.Tensor,
    *,
    index_to_value: dict[int, float],
    default_value: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if index_tensor.ndim == 0:
        indices = index_tensor.view(1)
    else:
        indices = index_tensor.reshape(-1)
    values = [index_to_value.get(int(i.item()), float(default_value)) for i in indices]
    return torch.tensor(values, dtype=dtype, device=device).view(-1, 1)


def apply_indicator_dropout(
    indicator: torch.Tensor,
    *,
    dropout_p: float,
    null_value: float,
    training: bool,
) -> torch.Tensor:
    if not training or dropout_p <= 0.0:
        return indicator
    mask = torch.rand_like(indicator, dtype=torch.float32) < float(dropout_p)
    return torch.where(mask, torch.full_like(indicator, float(null_value)), indicator)

