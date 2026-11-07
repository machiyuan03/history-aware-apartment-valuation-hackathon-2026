import json

import numpy as np
import pytest

from microburbs.avm_metrics import avm_metrics


def test_perfect_predictions_and_strict_json():
    result = avm_metrics([100, 250, 700, 1200], [100, 250, 700, 1200])
    assert result['n'] == 4 and result['coverage'] == 1
    for key in ['mae', 'median_absolute_error', 'rmse', 'mape', 'mdape', 'p90_ape',
                'p95_ape', 'log_rmse', 'median_signed_percentage_error',
                'mean_signed_percentage_error', 'cod', 'prb']:
        assert result[key] == pytest.approx(0)
    for key in ['median_ratio', 'prd', 'hit5', 'hit10', 'hit20']:
        assert result[key] == pytest.approx(1)
    json.dumps(result, allow_nan=False)


def test_global_scale_bias_is_visible_without_ratio_dispersion():
    y = np.array([100, 200, 500, 1000])
    result = avm_metrics(y, 1.2 * y)
    for key in ['mape', 'mdape', 'p90_ape', 'p95_ape',
                'median_signed_percentage_error', 'mean_signed_percentage_error']:
        assert result[key] == pytest.approx(20)
    assert result['median_ratio'] == pytest.approx(1.2)
    assert result['cod'] == pytest.approx(0)
    assert result['prd'] == pytest.approx(1)
    assert result['prb'] == pytest.approx(0)
    assert result['log_rmse'] == pytest.approx(np.log(1.2))
    assert result['hit5'] == result['hit10'] == 0
    assert result['hit20'] == 1
    below = avm_metrics(y, .8 * y)
    assert below['median_signed_percentage_error'] == pytest.approx(-20)


def test_one_large_error_remains_in_tail_and_squared_error():
    y = np.full(10, 100.0)
    p = y.copy()
    p[-1] = 1000
    result = avm_metrics(y, p)
    assert result['n'] == 10 and result['coverage'] == 1
    assert result['mae'] == 90 and result['median_absolute_error'] == 0
    assert result['rmse'] == pytest.approx(900 / np.sqrt(10))
    assert result['rmse'] > 3 * result['mae']
    assert result['mape'] == 90 and result['mdape'] == 0
    assert result['p90_ape'] == pytest.approx(90)
    assert result['p95_ape'] == pytest.approx(495)
    assert result['log_rmse'] == pytest.approx(np.log(10) / np.sqrt(10))
    assert result['hit10'] == .9


def test_prb_known_slope_and_independent_ols():
    ratio = np.array([.8, .9, 1, 1.1, 1.2])
    expected = .1
    x = 10 + (ratio - 1) / expected
    y = np.exp2(x) / (.5 * (1 + ratio))
    p = ratio * y
    result = avm_metrics(y, p)
    assert result['prb'] == pytest.approx(expected)
    median_ratio = np.median(p / y)
    regressor = np.log2(.5 * (y + p / median_ratio))
    response = ((p / y) - median_ratio) / median_ratio
    independent_slope = np.linalg.lstsq(np.column_stack([np.ones(len(y)), regressor]),
                                       response, rcond=None)[0][1]
    assert result['prb'] == pytest.approx(independent_slope)
    assert result['cod'] == pytest.approx(12)
    assert result['prd'] == pytest.approx(np.mean(p / y) / (p.sum() / y.sum()))


@pytest.mark.parametrize('y,p', [([100], [120]), ([100, 100], [100, 100])])
def test_prb_without_predictor_variance_is_none(y, p):
    result = avm_metrics(y, p)
    assert result['prb'] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('y,p', [
    ([], []), ([1, 2], [1]), ([[1, 2]], [[1, 2]]), ([1, 2], [[1, 2]]),
    ([1, 2], [1, np.nan]), ([1, np.nan], [1, 2]), ([1, 2], [1, None]),
    ([1, 2], [1, np.inf]), ([1, -np.inf], [1, 2]),
    ([1, 2], [1, 0]), ([1, 0], [1, 2]), ([1, 2], [1, -1]),
    ([-1, 2], [1, 2]), ([1, 2], [1, 'missing']),
])
def test_invalid_or_missing_values_are_rejected(y, p):
    with pytest.raises(ValueError):
        avm_metrics(y, p)
