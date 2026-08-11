"""Order-preserving preprocessing for mixed numeric/categorical feature sets.

The legacy compatibility pipeline groups numeric and categorical fields during
``transform`` even though its fitted statistics are stored in the original raw
feature order.  That misaligns columns whenever the requested feature list
interleaves types.  Final commercial modules import this isolated corrected
implementation so they can remain reproducible without modifying the historical
pipeline already published on ``main``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Preprocessor:
    raw_features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    medians: dict[str, float]
    numeric_missing_indicators: set[str]
    categories: dict[str, list[str]]
    encoded_names_all: list[str]
    encoded_raw_all: list[str]
    means_all: np.ndarray
    scales_all: np.ndarray
    keep_mask: np.ndarray

    @property
    def encoded_names(self) -> list[str]:
        return [name for name, keep in zip(self.encoded_names_all, self.keep_mask) if keep]

    @property
    def encoded_raw_features(self) -> list[str]:
        return [name for name, keep in zip(self.encoded_raw_all, self.keep_mask) if keep]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        numeric = set(self.numeric_features)
        # Preserve the order used to fit names, means, scales, and keep_mask.
        for feature in self.raw_features:
            if feature in numeric:
                values = pd.to_numeric(frame[feature], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                missing = values.isna().to_numpy(dtype=float)
                blocks.append(
                    values.fillna(self.medians[feature]).to_numpy(dtype=float).reshape(-1, 1)
                )
                if feature in self.numeric_missing_indicators:
                    blocks.append(missing.reshape(-1, 1))
            else:
                values = frame[feature].astype("string").fillna("__MISSING__").astype(str)
                known = set(self.categories[feature]) - {"__OTHER__"}
                values = values.where(values.isin(known), "__OTHER__")
                for category in self.categories[feature]:
                    blocks.append(values.eq(category).to_numpy(dtype=float).reshape(-1, 1))

        if not blocks:
            return np.zeros((len(frame), 0), dtype=float)
        encoded = np.concatenate(blocks, axis=1)
        standardized = (encoded - self.means_all) / self.scales_all
        return standardized[:, self.keep_mask]


def fit_preprocessor(frame: pd.DataFrame, raw_features: list[str]) -> Preprocessor:
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    medians: dict[str, float] = {}
    missing_indicators: set[str] = set()
    categories: dict[str, list[str]] = {}
    encoded_names: list[str] = []
    encoded_raw: list[str] = []
    blocks: list[np.ndarray] = []

    for feature in raw_features:
        series = frame[feature]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric_features.append(feature)
            values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite = values.dropna()
            median = float(finite.median()) if len(finite) else 0.0
            medians[feature] = median
            missing = values.isna().to_numpy(dtype=float)
            blocks.append(values.fillna(median).to_numpy(dtype=float).reshape(-1, 1))
            encoded_names.append(feature)
            encoded_raw.append(feature)
            if missing.any():
                missing_indicators.add(feature)
                blocks.append(missing.reshape(-1, 1))
                encoded_names.append(f"{feature}__missing")
                encoded_raw.append(feature)
        else:
            categorical_features.append(feature)
            values = series.astype("string").fillna("__MISSING__").astype(str)
            feature_categories = sorted(values.unique().tolist())
            if "__OTHER__" not in feature_categories:
                feature_categories.append("__OTHER__")
            categories[feature] = feature_categories
            for category in feature_categories:
                blocks.append(values.eq(category).to_numpy(dtype=float).reshape(-1, 1))
                encoded_names.append(f"{feature}=={category}")
                encoded_raw.append(feature)

    encoded = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(frame), 0), dtype=float)
    means = np.mean(encoded, axis=0) if encoded.shape[1] else np.array([], dtype=float)
    scales = np.std(encoded, axis=0, ddof=0) if encoded.shape[1] else np.array([], dtype=float)
    keep = np.isfinite(scales) & (scales > 1e-12)
    safe_scales = np.where(keep, scales, 1.0)
    return Preprocessor(
        raw_features=raw_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        medians=medians,
        numeric_missing_indicators=missing_indicators,
        categories=categories,
        encoded_names_all=encoded_names,
        encoded_raw_all=encoded_raw,
        means_all=means,
        scales_all=safe_scales,
        keep_mask=keep,
    )
