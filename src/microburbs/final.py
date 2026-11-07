"""Fit the selected ensemble from aligned, cutoff-safe 57-column Parquet inputs.

Feature tables contain no identities: train_y and query_rows must retain their
respective feature table's row order. The caller prepares causal history and
attributes; checking time_years cannot establish the provenance of other fields.
The pretrained TabICL checkpoint is supplied separately via MICROBURBS_DATA_ROOT.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'

import numpy as np
import pandas as pd

from .feature_schema import ALLOWED_CATEGORICAL, ALLOWED_NUMERIC, FeatureSchema, positive_targets
from .methods.common import prediction_vector
from .models import BoostModel
from .resources import gpu_lock

CUTOFF = '2026-08-31'
SEED = 20261107
TOTAL_SCALE = 0.9715945458330023
FEATURE_COLUMNS = tuple(ALLOWED_CATEGORICAL + ALLOWED_NUMERIC)
MEMBERS = (('lightgbm', .5), ('tabicl_recent50', .25), ('tabicl_recent90', .25))
LGBM_PARAMETERS = dict(n_estimators=1200, num_leaves=47, learning_rate=.03,
                       min_child_samples=35, reg_lambda=7,
                       objective='regression_l1', half_life_years=6)
BATCH_ROWS = 4096


def validate_inputs(train_x, train_y, query_x, query_rows):
    """Check structural alignment and explicit time features before fitting."""
    for name, frame in (('train_x', train_x), ('query_x', query_x)):
        if frame.empty or frame.columns.duplicated().any():
            raise ValueError(f'{name} must be nonempty with unique feature names')
        if len(frame.columns) != 57 or set(frame.columns) != set(FEATURE_COLUMNS):
            raise ValueError(f'{name} must contain exactly the selected 57 feature columns')
    if list(train_x.columns) != list(query_x.columns):
        raise ValueError('Training and query feature columns must have the same order')
    schema = FeatureSchema().fit(train_x)
    schema.transform(train_x)
    schema.transform(query_x)
    if train_y.columns.duplicated().any() or 'price' not in train_y:
        raise ValueError('train_y requires a unique price column')
    y = positive_targets(train_y.price.to_numpy(), len(train_x))
    if query_rows.columns.duplicated().any() or 'gnaf_id' not in query_rows:
        raise ValueError('query_rows requires a unique gnaf_id column')
    ids = query_rows.gnaf_id.astype('string').reset_index(drop=True)
    if (len(ids) != len(query_x) or ids.isna().any() or ids.str.strip().eq('').any()
            or ids.duplicated().any()):
        raise ValueError('Query IDs must be nonempty, unique and aligned with query_x')
    # HistoryFeatures encodes each row's as_of date relative to 2010-01-01.
    cutoff = (pd.Timestamp(CUTOFF) - pd.Timestamp('2010-01-01')).days / 365.25
    training_time = pd.to_numeric(train_x.time_years, errors='raise').to_numpy(float)
    query_time = pd.to_numeric(query_x.time_years, errors='raise').to_numpy(float)
    if not np.isfinite(training_time).all() or (training_time > cutoff - 1 / 365.25 + 1e-10).any():
        raise ValueError('Training history must end before each cutoff-eligible sale')
    if not np.isfinite(query_time).all() or not np.allclose(query_time, cutoff, rtol=0, atol=1e-10):
        raise ValueError(f'Every query must use the fixed {CUTOFF} cutoff')
    return y, ids


def _make_model(member):
    if member == 'lightgbm':
        return BoostModel('lightgbm', threads=4, seed=SEED, params=LGBM_PARAMETERS)
    from .methods.tabicl.adapter import Method
    return Method(context_rows=8192, batch_rows=128, n_estimators=4,
                  target_mode='city_normalized', context_strategy='recent_global',
                  recent_fraction=.5 if member == 'tabicl_recent50' else .9,
                  seed=SEED)


def _member_predictions(member, train_x, y, query_x):
    model = _make_model(member)
    try:
        if member == 'lightgbm':
            model.fit(train_x, y)
        else:
            model.prepare(train_x, y, {'cutoff': CUTOFF})
        chunks = []
        for start in range(0, len(query_x), BATCH_ROWS):
            part = query_x.iloc[start:start + BATCH_ROWS]
            chunks.append(prediction_vector(model.predict(part), len(part)))
        return np.concatenate(chunks)
    finally:
        del model
        gc.collect()
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def predict(train_x, train_y, query_x, query_rows):
    """Refit all three fixed members and return ordered, whole-dollar predictions."""
    y, ids = validate_inputs(train_x, train_y, query_x, query_rows)
    log_prediction = np.zeros(len(query_x), dtype=float)
    for member, weight in MEMBERS:
        if member == 'lightgbm':
            values = _member_predictions(member, train_x, y, query_x)
        else:
            with gpu_lock(wait=True):
                values = _member_predictions(member, train_x, y, query_x)
        log_prediction += weight * np.log(prediction_vector(values, len(query_x)))
    # This single scale includes both the original and final recent calibration.
    values = prediction_vector(np.exp(log_prediction) * TOTAL_SCALE, len(query_x))
    rounded = np.maximum(1, np.rint(values))
    if (rounded >= np.iinfo(np.int64).max).any():
        raise ValueError('Predictions exceed the integer dollar output range')
    return pd.DataFrame({'gnaf_id': ids, 'value': rounded.astype(np.int64)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-x', required=True, type=Path, help='57-column feature Parquet')
    parser.add_argument('--train-y', required=True, type=Path, help='Aligned Parquet with price column')
    parser.add_argument('--query-x', required=True, type=Path, help='57-column feature Parquet')
    parser.add_argument('--query-rows', required=True, type=Path, help='Aligned Parquet with gnaf_id column')
    parser.add_argument('--output', required=True, type=Path, help='New gnaf_id,value CSV')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = [pd.read_parquet(path) for path in
              (args.train_x, args.train_y, args.query_x, args.query_rows)]
    output = predict(*inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        output.to_csv(stream, index=False)
    print(f'Wrote {len(output)} predictions to {args.output}')


if __name__ == '__main__':
    main()
