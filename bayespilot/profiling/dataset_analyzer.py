from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd

@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    missing_counts: Dict[str, int]
    candidate_targets: List[str]

def analyze_dataset(df: pd.DataFrame) -> DatasetProfile:
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=["number"]).columns.tolist()
    missing_counts = df.isnull().sum().to_dict()

    candidate_targets = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique == 2:
            candidate_targets.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            candidate_targets.append(col)

    return DatasetProfile(
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        missing_counts=missing_counts,
        candidate_targets=candidate_targets,
    )