"""
Profit/Loss analysis — computed entirely from cache.

Sources:
  cache['trades_df']    — trade-level PnL and Return (L / O)
  cache['daily_returns'] — daily returns array (D)
  cache['daily_pf'].value — equity curve for monthly breakdown + rolling windows (D)
  cache scalars          — total_return_pct, total_fees_paid (V)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

from ..shared.utils import ANN_FACTOR

_ROLLING_DAYS = [1, 7, 15, 30, 45, 60, 90]


def _get_empty_profit_loss() -> Dict[str, Any]:
    result = {
        'profit_factor': 0.0, 'payoff_ratio': 0.0, 'profit_loss_ratio': 0.0,
        'gain_to_pain_ratio': 0.0, 'cpc_ratio': 0.0,
        'total_return_pct': 0.0, 'total_fees_paid': 0.0,
        'total_gains': 0.0, 'total_losses': 0.0,
        'expected_return_annualized_pct': 0.0,
        'monthly_returns_mean_pct': 0.0, 'monthly_returns_std_pct': 0.0,
        'best_month_return_pct': 0.0, 'worst_month_return_pct': 0.0,
        'positive_months_pct': 0.0, 'monthly_win_rate': 0.0,
        'monthly_returns': {},
    }
    for d in _ROLLING_DAYS:
        result[f'{d}d_pnl_pct'] = 0.0
    return result


def _extract_profit_loss_from_cache(cache: Dict) -> Dict[str, Any]:
    """
    All profit/loss stats from cache only — no ledger input required.

    profit_factor, total_gains, total_losses  — trades_df['PnL']  (currency)
    payoff_ratio, profit_loss_ratio, cpc_ratio — trades_df['Return'] (fraction)
    gain_to_pain_ratio, CAGR                  — cache['daily_returns']
    rolling windows                            — cache['daily_pf'].value
    total_return_pct, total_fees_paid          — cache scalars
    """
    result = _get_empty_profit_loss()

    # 1. Pre-computed scalars (V) ----------------------------------------
    result['total_return_pct'] = float(cache.get('total_return_pct', 0.0))
    result['total_fees_paid']  = float(cache.get('total_fees_paid',  0.0))

    # 2. Trade-level stats from trades_df (L) ----------------------------
    trades_df = cache.get('trades_df', pd.DataFrame())
    if not trades_df.empty:

        # profit_factor, total_gains, total_losses — from PnL (currency units)
        if 'PnL' in trades_df.columns:
            pnl = trades_df['PnL'].to_numpy(dtype=np.float64, na_value=np.nan)
            pnl = pnl[~np.isnan(pnl)]
            if len(pnl) > 0:
                win_mask = pnl > 0
                los_mask = pnl < 0
                sum_win  = float(np.sum(pnl[win_mask])) if win_mask.any() else 0.0
                sum_los  = float(np.sum(pnl[los_mask])) if los_mask.any() else 0.0
                result['total_gains']   = sum_win
                result['total_losses']  = abs(sum_los)
                result['profit_factor'] = sum_win / abs(sum_los) if sum_los != 0 else 0.0

        # payoff_ratio, profit_loss_ratio, cpc_ratio — from Return (fraction)
        if 'Return' in trades_df.columns:
            ret = trades_df['Return'].to_numpy(dtype=np.float64, na_value=np.nan)
            ret = ret[~np.isnan(ret)]
            if len(ret) > 0:
                win_mask  = ret > 0
                los_mask  = ret < 0
                avg_win   = float(np.mean(ret[win_mask]))        if win_mask.any() else 0.0
                avg_los   = float(abs(np.mean(ret[los_mask])))   if los_mask.any() else 0.0
                sum_win_r = float(np.sum(ret[win_mask]))         if win_mask.any() else 0.0
                sum_los_r = float(abs(np.sum(ret[los_mask])))    if los_mask.any() else 0.0

                result['payoff_ratio']      = avg_win / avg_los   if avg_los   > 0 else 0.0
                result['profit_loss_ratio'] = sum_win_r / sum_los_r if sum_los_r > 0 else 0.0
                result['cpc_ratio']         = result['profit_factor'] * result['payoff_ratio']

    # 3. Daily-return-based stats (D) ------------------------------------
    dr = cache.get('daily_returns', np.array([]))
    if len(dr) > 0:
        pos_dr  = dr[dr > 0]
        neg_dr  = dr[dr < 0]
        sum_pos = float(np.sum(pos_dr))        if len(pos_dr) > 0 else 0.0
        sum_neg = float(abs(np.sum(neg_dr)))   if len(neg_dr) > 0 else 0.0
        result['gain_to_pain_ratio'] = sum_pos / sum_neg if sum_neg > 0 else 0.0

        years = len(dr) / ANN_FACTOR
        if years > 0:
            result['expected_return_annualized_pct'] = float(
                (np.prod(1.0 + dr) ** (1.0 / years) - 1.0) * 100.0
            )

    # 4. Rolling N-day PnL % from equity curve (D) ----------------------
    try:
        value_series = cache['daily_pf'].value      # pd.Series, datetime index
        if len(value_series) > 0:
            idx       = value_series.index
            final_val = float(value_series.iloc[-1])
            now       = idx[-1]

            for days in _ROLLING_DAYS:
                cutoff    = now - pd.Timedelta(days=days)
                positions = np.where(idx <= cutoff)[0]
                if len(positions) > 0:
                    past_val = float(value_series.iloc[positions[-1]])
                    result[f'{days}d_pnl_pct'] = (
                        (final_val / past_val - 1.0) * 100.0 if past_val != 0 else 0.0
                    )
    except Exception:
        pass

    return result
