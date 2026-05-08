"""
VBT cache builder — single source of truth for all expensive VBT calls.

All downstream stats modules read from this cache dict; no VBT attribute
access happens outside of this module (except ledger / trade-level work).

Cache key index:
  daily_pf               — daily-resampled portfolio (master for all D stats)
  vbt_stats              — daily_pf.stats() pd.Series (kept for unmapped attrs)
  daily_returns          — NaN-stripped daily returns array            (D)
  bm_returns             — benchmark returns array                     (D)
  bm_returns_series      — benchmark returns pd.Series w/ datetime idx (D)
  value_array            — daily portfolio value array                 (D)
  cash_array             — daily cash array                            (D)
  cash_flow_array        — daily cash flow array                       (D)
  drawdowns_df           — daily_pf.drawdowns.records_readable         (D)
  gross/net/long/short_exposure_array — daily exposure arrays          (D)
  trades_df              — original pf.trades.records_readable         (O)
  mfe_values             — original pf.trades.mfe.values               (O)
  mae_values             — original pf.trades.mae.values               (O)
  sharpe_ratio           — direct daily_pf.sharpe_ratio                (V)
  sortino_ratio          — direct daily_pf.sortino_ratio               (V)
  calmar_ratio           — direct daily_pf.calmar_ratio                (V)
  omega_ratio            — direct daily_pf.omega_ratio                 (V)
  alpha                  — direct daily_pf.alpha                       (V)
  beta                   — direct daily_pf.beta                        (V)
  information_ratio      — direct daily_pf.information_ratio           (V)
  capture_ratio          — direct daily_pf.capture_ratio               (V)
  up_capture_ratio       — direct daily_pf.up_capture_ratio            (V)
  down_capture_ratio     — direct daily_pf.down_capture_ratio          (V)
  treynor_ratio          — CAGR / beta (manual; not a VBT builtin)
  total_return_pct       — daily_pf.total_return * 100                 (V)
  benchmark_return_pct   — computed from bm_returns array              (D)
  max_drawdown_pct       — abs(daily_pf.max_drawdown) * 100            (V)
  max_drawdown_duration_days — from drawdowns_df Duration col          (D)
  max_value              — nanmax(value_array)                         (D)
  min_value              — nanmin(value_array)                         (D)
  initial_value          — value_array[0]                              (D)
  final_value            — daily_pf.final_value                        (V)
  start_date             — daily_pf.wrapper.index[0]                   (V)
  end_date               — daily_pf.wrapper.index[-1]                  (V)
  total_duration_days    — (end_date - start_date) in days             (V)
  max_gross_exposure_pct — nanmax(gross_exposure_array) * 100          (D)
  position_coverage_pct  — daily_pf.position_coverage * 100            (V)
  total_fees_paid        — daily_pf.total_fees or vbt_stats fallback   (V)
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

from ..shared.utils import ANN_FACTOR

logger = logging.getLogger(__name__)


def _build_vbt_cache(pf) -> Dict[str, Any]:
    """
    Build unified cache from the daily-resampled portfolio.

    All expensive VBT calls are made exactly once here.
    Downstream modules read from this dict — no VBT attribute access elsewhere.
    """
    cache: Dict[str, Any] = {}

    # 1. Daily resampled portfolio -----------------------------------------
    try:
        daily_pf = pf.resample('1D')
    except Exception as e:
        logger.warning(f"Daily resample failed ({e}), using original pf.")
        daily_pf = pf
    cache['daily_pf'] = daily_pf

    # 2. VBT stats — called ONCE; kept for any attrs not yet mapped --------
    try:
        cache['vbt_stats'] = daily_pf.stats()          # pd.Series
    except Exception as e:
        logger.warning(f"daily_pf.stats() failed: {e}")
        cache['vbt_stats'] = pd.Series(dtype=float)

    # 3. Daily returns array -----------------------------------------------
    try:
        dr = daily_pf.returns.values
        cache['daily_returns'] = dr[~np.isnan(dr)]
    except Exception:
        cache['daily_returns'] = np.array([])

    # 4. Benchmark returns (buy-and-hold from close price) ----------------
    try:
        bm_series = daily_pf.bm_returns          # pd.Series with datetime index
        bm_vals   = bm_series.values
        cache['bm_returns']        = bm_vals[~np.isnan(bm_vals)]
        cache['bm_returns_series'] = bm_series
    except Exception:
        cache['bm_returns']        = np.array([])
        cache['bm_returns_series'] = None

    # 5. Portfolio value (daily) ------------------------------------------
    try:
        cache['value_array'] = daily_pf.value.values
    except Exception:
        cache['value_array'] = np.array([])

    # 6. Cash (daily) ------------------------------------------------------
    try:
        cache['cash_array'] = daily_pf.cash.values
    except Exception:
        cache['cash_array'] = np.array([])

    # 7. Cash flow (daily) -------------------------------------------------
    try:
        cache['cash_flow_array'] = daily_pf.cash_flow.values
    except Exception:
        cache['cash_flow_array'] = np.array([])

    # 8. Drawdowns — durations already in calendar days -------------------
    try:
        cache['drawdowns_df'] = daily_pf.drawdowns.records_readable
    except Exception:
        cache['drawdowns_df'] = pd.DataFrame()

    # 9. Exposure arrays (daily) ------------------------------------------
    for _attr, _key in [
        ('gross_exposure',       'gross_exposure_array'),
        ('net_exposure',         'net_exposure_array'),
        ('long_gross_exposure',  'long_exposure_array'),
        ('short_gross_exposure', 'short_exposure_array'),
    ]:
        try:
            cache[_key] = getattr(daily_pf, _attr).values
        except Exception:
            cache[_key] = np.array([])

    # 10. Trade records — ORIGINAL pf (not resampled) ---------------------
    try:
        cache['trades_df'] = pf.trades.records_readable
    except Exception:
        cache['trades_df'] = pd.DataFrame()

    # 11. MFE / MAE — original pf -----------------------------------------
    try:
        cache['mfe_values'] = pf.trades.mfe.values
    except Exception:
        cache['mfe_values'] = np.array([])
    try:
        cache['mae_values'] = pf.trades.mae.values
    except Exception:
        cache['mae_values'] = np.array([])

    # 12. Performance ratios — direct attr access -------------------------
    for _attr, _key in [
        ('sharpe_ratio',  'sharpe_ratio'),
        ('sortino_ratio', 'sortino_ratio'),
        ('calmar_ratio',  'calmar_ratio'),
        ('omega_ratio',   'omega_ratio'),
    ]:
        try:
            _val = getattr(daily_pf, _attr)
            cache[_key] = float(_val() if callable(_val) else _val)
        except Exception:
            cache[_key] = 0.0

    # 13. Benchmark-relative scalars — direct attr access -----------------
    for _attr, _key in [
        ('alpha',              'alpha'),
        ('beta',               'beta'),
        ('information_ratio',  'information_ratio'),
        ('capture_ratio',      'capture_ratio'),
        ('up_capture_ratio',   'up_capture_ratio'),
        ('down_capture_ratio', 'down_capture_ratio'),
    ]:
        try:
            _val = getattr(daily_pf, _attr)
            cache[_key] = float(_val() if callable(_val) else _val)
        except Exception:
            cache[_key] = 0.0

    # 14. Treynor ratio — not a VBT builtin; computed as CAGR / beta ------
    try:
        _beta = cache.get('beta', 0.0)
        _dr   = cache.get('daily_returns', np.array([]))
        if _beta != 0 and len(_dr) > 0:
            _years = len(_dr) / ANN_FACTOR
            _cagr  = float(np.prod(1 + _dr) ** (1.0 / _years) - 1) if _years > 0 else 0.0
            cache['treynor_ratio'] = _cagr / _beta
        else:
            cache['treynor_ratio'] = 0.0
    except Exception:
        cache['treynor_ratio'] = 0.0

    # 15. Return metrics ---------------------------------------------------
    try:
        _tr = getattr(daily_pf, 'total_return')
        cache['total_return_pct'] = float(_tr() if callable(_tr) else _tr) * 100.0
    except Exception:
        _va = cache.get('value_array', np.array([]))
        cache['total_return_pct'] = (
            float((_va[-1] / _va[0] - 1) * 100.0)
            if len(_va) >= 2 and _va[0] != 0
            else 0.0
        )

    _bm = cache.get('bm_returns', np.array([]))
    cache['benchmark_return_pct'] = (
        float((np.prod(1 + _bm) - 1) * 100.0) if len(_bm) > 0 else 0.0
    )

    # 16. Drawdown metrics ------------------------------------------------
    try:
        _mdd = getattr(daily_pf, 'max_drawdown')
        # direct attr returns a fraction (e.g. -0.105); multiply by 100 for %
        cache['max_drawdown_pct'] = abs(float(_mdd() if callable(_mdd) else _mdd)) * 100.0
    except Exception:
        cache['max_drawdown_pct'] = 0.0

    try:
        _ddf = cache.get('drawdowns_df', pd.DataFrame())
        if not _ddf.empty and 'Start Index' in _ddf.columns and 'End Index' in _ddf.columns:
            _starts = pd.to_datetime(_ddf['Start Index'])
            _ends   = pd.to_datetime(_ddf['End Index']).fillna(_starts.max())
            _durs   = (_ends - _starts).dt.total_seconds().values / 86400.0
            cache['max_drawdown_duration_days'] = float(np.max(_durs))
        else:
            cache['max_drawdown_duration_days'] = 0.0
    except Exception:
        cache['max_drawdown_duration_days'] = 0.0

    # 17. Portfolio value stats -------------------------------------------
    _va = cache.get('value_array', np.array([]))
    if len(_va) > 0:
        cache['max_value']     = float(np.nanmax(_va))
        cache['min_value']     = float(np.nanmin(_va))
        cache['initial_value'] = float(_va[0])
    else:
        cache['max_value'] = cache['min_value'] = cache['initial_value'] = 0.0

    try:
        _fv = getattr(daily_pf, 'final_value')
        cache['final_value'] = float(_fv() if callable(_fv) else _fv)
    except Exception:
        cache['final_value'] = float(_va[-1]) if len(_va) > 0 else 0.0

    # 18. Dates and duration ----------------------------------------------
    try:
        idx = daily_pf.wrapper.index
        cache['start_date'] = idx[0]
        cache['end_date']   = idx[-1]
        _td = idx[-1] - idx[0]
        cache['total_duration_days'] = (
            _td.total_seconds() / 86400.0
            if hasattr(_td, 'total_seconds')
            else float(_td)
        )
    except Exception:
        cache['start_date']          = None
        cache['end_date']            = None
        cache['total_duration_days'] = 0.0

    # 19. Exposure / coverage ---------------------------------------------
    _ge = cache.get('gross_exposure_array', np.array([]))
    cache['max_gross_exposure_pct'] = (
        float(np.nanmax(_ge) * 100.0) if len(_ge) > 0 else 0.0
    )

    try:
        _pc = getattr(daily_pf, 'position_coverage')
        cache['position_coverage_pct'] = float(_pc() if callable(_pc) else _pc) * 100.0
    except Exception:
        cache['position_coverage_pct'] = (
            float(np.mean(_ge > 0) * 100.0) if len(_ge) > 0 else 0.0
        )

    # 20. Fees paid -------------------------------------------------------
    try:
        _fees = getattr(daily_pf, 'total_fees')
        cache['total_fees_paid'] = float(_fees() if callable(_fees) else _fees)
    except Exception:
        _vs = cache.get('vbt_stats', pd.Series(dtype=float))
        try:
            cache['total_fees_paid'] = float(_vs.get('Total Fees Paid', 0.0))
        except Exception:
            cache['total_fees_paid'] = 0.0

    return cache


def _build_essential_vbt_cache(pf) -> Dict[str, Any]:
    """Minimal cache for essential stats (9 core metrics only)."""
    cache: Dict[str, Any] = {}
    try:
        daily_pf = pf.resample('1D')
    except Exception:
        daily_pf = pf
    try:
        cache['vbt_stats'] = daily_pf.stats()
    except Exception:
        cache['vbt_stats'] = pd.Series(dtype=float)
    try:
        cache['value_array'] = daily_pf.value.values
    except Exception:
        cache['value_array'] = np.array([])
    return cache
