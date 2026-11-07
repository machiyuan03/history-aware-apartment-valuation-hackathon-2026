"""AVM diagnostics on every supplied observation, with NumPy as the only dependency.

For positive sale values y and predictions p, let r = p / y and m = median(r).
Errors are e = p - y; APE = 100 * abs(e / y). MAE, median absolute error and
RMSE retain y's currency units. MAPE/MdAPE and p90/p95 APE are percentages;
signed percentage errors are 100 * (r - 1). log_rmse uses natural logarithms:
sqrt(mean(log(p / y)**2)). median_ratio is m.

COD = 100 * mean(abs(r - m)) / m.
PRD = mean(r) / (sum(p) / sum(y)).
PRB is the OLS slope, with an intercept, of (r - m) / m on
log2(0.5 * (y + p / m)). It is a fractional slope, not multiplied by 100;
it is None for fewer than two observations or numerically constant regressors.
Hit@5/10/20 are inclusive proportions, with floating-point roundoff tolerance.

Formula source: IAAO, Standard on Ratio Studies (2013),
https://www.iaao.org/wp-content/uploads/Standard_on_Ratio_Studies.pdf .
The IAAO Standard on Automated Valuation Models (2018) informs the principle of
reporting multiple accuracy and uniformity diagnostics. This helper neither
sets acceptance thresholds nor ranks models. It performs no trimming,
winsorization, label-based grouping, or silent omission of invalid predictions.
Percentiles use NumPy's default linear interpolation. All results are Python
numbers (or None for unavailable PRB), suitable for strict JSON serialization.
"""
from __future__ import annotations

import numpy as np


def avm_metrics(y, p):
    """Return full-sample diagnostics; reject empty, mismatched or invalid input."""
    try:
        y = np.asarray(y, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('y and p must be numeric one-dimensional arrays') from None
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError('y and p must be nonempty, one-dimensional, and equal-length')
    if not np.isfinite(y).all() or not np.isfinite(p).all() or (y <= 0).any() or (p <= 0).any():
        raise ValueError('Every y and p must be finite and strictly positive; missing predictions are not omitted')
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            error = p - y
            absolute_error = np.abs(error)
            ratio = p / y
            relative_error = absolute_error / y
            ape = 100 * relative_error
            signed = 100 * (ratio - 1)
            median_ratio = float(np.median(ratio))
            p90, p95 = np.percentile(ape, [90, 95])
            x = np.log2(0.5 * (y + p / median_ratio))
            relative_ratio = (ratio - median_ratio) / median_ratio
            spread_tolerance = 8 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(x))))
            prb = None
            if y.size >= 2 and float(np.ptp(x)) > spread_tolerance:
                centered_x = x - np.mean(x)
                centered_ratio = relative_ratio - np.mean(relative_ratio)
                prb = float(np.dot(centered_x, centered_ratio) / np.dot(centered_x, centered_x))
            result = {
                'n': int(y.size),
                'coverage': 1.0,
                'mae': float(np.mean(absolute_error)),
                'median_absolute_error': float(np.median(absolute_error)),
                'rmse': float(np.sqrt(np.mean(error ** 2))),
                'mape': float(np.mean(ape)),
                'mdape': float(np.median(ape)),
                'p90_ape': float(p90),
                'p95_ape': float(p95),
                'log_rmse': float(np.sqrt(np.mean((np.log(p) - np.log(y)) ** 2))),
                'median_signed_percentage_error': float(np.median(signed)),
                'mean_signed_percentage_error': float(np.mean(signed)),
                'median_ratio': median_ratio,
                'cod': float(100 * np.mean(np.abs(ratio - median_ratio)) / median_ratio),
                'prd': float(np.mean(ratio) / (np.sum(p) / np.sum(y))),
                'prb': prb,
            }
            tolerance = 8 * np.finfo(float).eps
            for percent in (5, 10, 20):
                result[f'hit{percent}'] = float(np.mean(relative_error <= percent / 100 + tolerance))
    except FloatingPointError:
        raise ValueError('Input magnitudes exceed the supported numerical range') from None
    if any(value is not None and not np.isfinite(value) for value in result.values()):
        raise ValueError('Input magnitudes produced nonfinite metrics')
    return result
