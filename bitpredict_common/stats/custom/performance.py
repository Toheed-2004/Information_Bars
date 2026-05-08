import pandas as pd
import numpy as np
from typing import Dict, Optional


def _calculate_monthly_breakdown(
    df_ledger: pd.DataFrame,
    returns_df: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Monthly returns using daily returns_df (matching VBT's daily_pf.value resampled to month-end).
    Falls back to ledger-based calculation if returns_df not provided.
    """
    # ── Use daily returns_df (preferred, matches VBT) ─────────────────────
    if returns_df is not None and not returns_df.empty and 'balance' in returns_df.columns:
        try:
            balance = pd.Series(
                returns_df['balance'].values,
                index=pd.to_datetime(returns_df['datetime'])
            ).sort_index()

            monthly_balance = balance.resample('M').last()

            if len(monthly_balance) < 2:
                return {}

            monthly_returns = (monthly_balance[1:] / monthly_balance[:-1].values) - 1

            return {
                date.strftime('%Y-%m-%d'): float(ret * 100)
                for date, ret in zip(monthly_balance.index[1:], monthly_returns)
            }
        except Exception:
            pass

    # ── Fallback: ledger-based ────────────────────────────────────────────
    if df_ledger is None or len(df_ledger) == 0 or 'exit_datetime' not in df_ledger.columns:
        return {}

    try:
        df_temp = df_ledger
        if not pd.api.types.is_datetime64_any_dtype(df_temp['exit_datetime']):
            df_temp = df_temp.copy()
            df_temp['exit_datetime'] = pd.to_datetime(df_temp['exit_datetime'])

        df_temp = df_temp.set_index('exit_datetime').sort_index()

        if 'balance' not in df_temp.columns or len(df_temp) < 2:
            return {}

        monthly_balance = df_temp['balance'].resample('M').last()

        if len(monthly_balance) < 2:
            return {}

        monthly_returns = (monthly_balance[1:] / monthly_balance[:-1].values) - 1

        return {
            date.strftime('%Y-%m-%d'): float(ret * 100)
            for date, ret in zip(monthly_balance.index[1:], monthly_returns)
        }
    except Exception:
        return {}


def _calculate_recent_performance(
    df_ledger: pd.DataFrame,
    returns_df: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Recent performance (1d, 7d, 15d, 30d, 45d, 60d, 90d) using daily returns.
    Uses returns_df if available (matches VBT), falls back to ledger.
    """
    defaults = {f'{d}d_pnl_pct': 0.0 for d in [1, 7, 15, 30, 45, 60, 90]}

    # ── Use daily returns_df (preferred, matches VBT) ─────────────────────
    if returns_df is not None and not returns_df.empty and 'portfolio_return' in returns_df.columns:
        try:
            daily_returns = returns_df['portfolio_return'].values
            if len(daily_returns) == 0:
                return defaults

            result = {}
            for period in [1, 7, 15, 30, 45, 60, 90]:
                recent = daily_returns[-period:] if len(daily_returns) >= period else daily_returns
                result[f'{period}d_pnl_pct'] = float((np.prod(1 + recent) - 1) * 100)
            return result
        except Exception:
            pass

    # ── Fallback: ledger-based ────────────────────────────────────────────
    if df_ledger is None or len(df_ledger) == 0 or 'balance' not in df_ledger.columns:
        return defaults

    try:
        df_temp = df_ledger.copy()
        if 'exit_datetime' in df_temp.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_temp['exit_datetime']):
                df_temp['exit_datetime'] = pd.to_datetime(df_temp['exit_datetime'])
            df_temp = df_temp.set_index('exit_datetime').sort_index()
            daily_balance = df_temp['balance'].resample('D').last()
            if len(daily_balance) < 2:
                return defaults
            daily_returns = daily_balance.pct_change().fillna(0).values
        else:
            return defaults

        result = {}
        for period in [1, 7, 15, 30, 45, 60, 90]:
            recent = daily_returns[-period:] if len(daily_returns) >= period else daily_returns
            result[f'{period}d_pnl_pct'] = float((np.prod(1 + recent) - 1) * 100)
        return result
    except Exception:
        return defaults
