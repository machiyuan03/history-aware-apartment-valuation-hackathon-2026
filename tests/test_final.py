"""Final ensemble boundary checks, using synthetic features and model doubles."""
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

from microburbs import final


@pytest.fixture
def inputs():
    categorical = set(final.ALLOWED_CATEGORICAL)
    x = pd.DataFrame({name: ['x', 'y'] if name in categorical else [1., 2.]
                      for name in final.FEATURE_COLUMNS})
    cutoff = (pd.Timestamp('2026-08-31') - pd.Timestamp('2010-01-01')).days / 365.25
    x['time_years'] = [cutoff - 3 / 365.25, cutoff - 1 / 365.25]
    q = x.copy()
    q['time_years'] = cutoff
    return x, pd.DataFrame({'price': [500_000., 600_000.]}), q, pd.DataFrame({'gnaf_id': ['B', 'A']})


def test_fixed_ensemble_preserves_query_order_and_applies_total_scale_once(inputs, monkeypatch):
    seen = []
    predictions = {'lightgbm': [100_000., 200_000.],
                   'tabicl_recent50': [400_000., 800_000.],
                   'tabicl_recent90': [400_000., 800_000.]}
    def member(name, x, y, q):
        seen.append(name)
        return predictions[name]
    monkeypatch.setattr(final, '_member_predictions', member)
    monkeypatch.setattr(final, 'gpu_lock', lambda **kwargs: nullcontext())
    result = final.predict(*inputs)
    assert seen == ['lightgbm', 'tabicl_recent50', 'tabicl_recent90']
    assert result.gnaf_id.tolist() == ['B', 'A']
    assert result.value.tolist() == [194319, 388638]
    assert list(result) == ['gnaf_id', 'value']


@pytest.mark.parametrize('issue', ['zero_label', 'wrong_labels', 'duplicate_ids', 'null_id',
                                    'unknown_feature', 'column_order', 'future_training', 'wrong_cutoff'])
def test_invalid_input_rejected_before_model_fit(inputs, monkeypatch, issue):
    x, y, q, ids = inputs
    if issue == 'zero_label': y.loc[0, 'price'] = 0
    elif issue == 'wrong_labels': y = y.iloc[:1]
    elif issue == 'duplicate_ids': ids.loc[1, 'gnaf_id'] = 'B'
    elif issue == 'null_id': ids.loc[1, 'gnaf_id'] = None
    elif issue == 'unknown_feature': x = x.rename(columns={'rent_weekly': 'future_sale_price'})
    elif issue == 'column_order': q = q[q.columns[::-1]]
    elif issue == 'future_training': x.loc[0, 'time_years'] = q.time_years.iloc[0]
    elif issue == 'wrong_cutoff': q.loc[0, 'time_years'] += 1 / 365.25
    monkeypatch.setattr(final, '_make_model', lambda *args: pytest.fail('Fit began before validation'))
    with pytest.raises(ValueError):
        final.predict(x, y, q, ids)


def test_invalid_member_output_cannot_be_written(inputs, monkeypatch):
    monkeypatch.setattr(final, '_member_predictions', lambda *args: [1., np.nan])
    with pytest.raises(ValueError, match='finite positive'):
        final.predict(*inputs)


def test_member_cache_batching_keeps_order_and_uses_fixed_parameters(inputs, monkeypatch):
    from microburbs.methods.tabicl import adapter
    events = []
    class FakeModel:
        def __init__(self, **kwargs):
            events.append(kwargs)
        def prepare(self, x, y, context):
            assert context['cutoff'] == '2026-08-31'
        def predict(self, q):
            return q.unit_number.to_numpy() * 100
    monkeypatch.setattr(adapter, 'Method', FakeModel)
    monkeypatch.setattr(final, 'BATCH_ROWS', 1)
    x, y, q, ids = inputs
    for name in ('tabicl_recent50', 'tabicl_recent90'):
        actual = final._member_predictions(name, x, y.price.to_numpy(), q)
        np.testing.assert_equal(actual, [100, 200])
    assert [p['recent_fraction'] for p in events] == [.5, .9]
    assert all(p['context_rows'] == 8192 and p['n_estimators'] == 4 and p['batch_rows'] == 128
               and p['seed'] == 20261107 for p in events)


def test_cli_writes_expected_csv_and_refuses_overwrite(inputs, monkeypatch, tmp_path):
    names = ['train-x', 'train-y', 'query-x', 'query-rows']
    args = []
    for name, frame in zip(names, inputs):
        path = tmp_path / (name + '.parquet')
        frame.to_parquet(path, index=False)
        args.extend(['--' + name, str(path)])
    output = tmp_path / 'predictions.csv'
    args.extend(['--output', str(output)])
    monkeypatch.setattr(final, '_member_predictions', lambda *args: np.array([100_000., 200_000.]))
    monkeypatch.setattr(final, 'gpu_lock', lambda **kwargs: nullcontext())
    final.main(args)
    result = pd.read_csv(output)
    assert result.gnaf_id.tolist() == ['B', 'A']
    assert result.value.tolist() == [97159, 194319]
    with pytest.raises(FileExistsError):
        final.main(args)
