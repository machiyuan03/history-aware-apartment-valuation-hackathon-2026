"""Explicit, fitted model feature contract; source labels never enter by discovery."""
import numpy as np
import pandas as pd

from .features import CATEGORICAL, FEATURES, NUMERIC
from .broad_features import BROAD_CATEGORICAL, BROAD_NUMERIC, RENT_CATEGORICAL, RENT_NUMERIC


OPTIONAL_CATEGORICAL = ['address_unit_type', *BROAD_CATEGORICAL, *RENT_CATEGORICAL]
OPTIONAL_NUMERIC = ['latitude', 'longitude', 'parsed_level', 'hist_bedrooms',
                    'hist_bathrooms', 'hist_car_spaces', 'attribute_age_years', 'attribute_known', 'repeat_adjusted_unclipped', *BROAD_NUMERIC, *RENT_NUMERIC]
ALLOWED_CATEGORICAL = [*CATEGORICAL, *OPTIONAL_CATEGORICAL]
ALLOWED_NUMERIC = [*NUMERIC, *OPTIONAL_NUMERIC]


class FeatureSchema:
    """Fit once to allowed columns; prediction must supply the same named fields."""
    def fit(self, X):
        if X.columns.duplicated().any():
            raise ValueError('Duplicate input feature names')
        missing = set(FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f'Missing base features: {sorted(missing)}')
        self.categorical = [c for c in ALLOWED_CATEGORICAL if c in X]
        self.numeric = [c for c in ALLOWED_NUMERIC if c in X]
        self.columns = [*self.categorical, *self.numeric]
        self.version = 'official_explicit_v1'
        return self

    def transform(self, X):
        if X.columns.duplicated().any():
            raise ValueError('Duplicate input feature names')
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f'Missing fitted features: {sorted(missing)}')
        z = X[self.columns].copy()
        for col in self.categorical:
            z[col] = z[col].fillna('').astype(str)
        for col in self.numeric:
            source = z[col].replace('', np.nan)
            values = pd.to_numeric(source, errors='coerce').astype(float)
            if (source.notna() & values.isna()).any() or np.isinf(values.to_numpy()).any():
                raise ValueError(f'Invalid numeric feature: {col}')
            z[col] = values
        return z

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def training_weights(X, half_life_years=None, sample_weight=None):
    """Combine caller weights with mean-one recency weights from training times."""
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float).copy()
        if (weights.shape != (len(X),) or not np.isfinite(weights).all()
                or (weights < 0).any() or not (weights > 0).any()):
            raise ValueError('Sample weights must be finite, nonnegative and aligned')
    if half_life_years is not None:
        if not np.isfinite(half_life_years) or half_life_years <= 0:
            raise ValueError('half_life_years must be positive or None')
        times = pd.to_numeric(X['time_years'], errors='raise').to_numpy(dtype=float)
        if not len(times) or not np.isfinite(times).all():
            raise ValueError('Recency weighting requires finite training times')
        recency = np.exp2(-(times.max() - times) / half_life_years)
        recency /= recency.mean()
        weights = recency if weights is None else weights * recency
    return weights


def positive_targets(y, count):
    values = np.asarray(y, dtype=float)
    if values.shape != (count,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Targets must be finite positive prices aligned with X')
    return values
