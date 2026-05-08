import numpy as np
import pandas as pd
from typing import Dict, Any


def _extract_drawdown_stats_vectorized(cache: Dict) -> Dict[str, Any]:
    """
    Drawdown statistics from cache.

    daily_pf.drawdown is always ≤ 0 by VBT convention.
    drawdowns_df durations are already in calendar days.
    max_drawdown_pct, max_drawdown_duration_days, total_return_pct
    are pre-computed in cache_builder — used directly here.
    """
    stats: Dict[str, Any] = {}
    daily_pf  = cache.get('daily_pf')
    value_arr = cache.get('value_array', np.array([]))

    # ------------------------------------------------------------------
    # 1. Drawdown array — daily_pf.drawdown (authoritative, always ≤ 0)
    # ------------------------------------------------------------------
    drawdown_array = np.array([])
    try:
        if daily_pf is not None:
            drawdown_array = daily_pf.drawdown.values
    except Exception:
        pass

    if len(drawdown_array) == 0 and len(value_arr) > 0:
        running_max = np.maximum.accumulate(value_arr)
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown_array = np.where(
                running_max > 0, (value_arr - running_max) / running_max, 0.0
            )

    if len(drawdown_array) > 0:
        stats.update({
            'current_drawdown_pct':    float(drawdown_array[-1] * 100),
            'drawdown_volatility_pct': float(np.std(drawdown_array, ddof=1) * 100) if len(drawdown_array) > 1 else 0.0,
        })

    # ------------------------------------------------------------------
    # 2. Max drawdown % — pre-computed in cache (step 16)
    # ------------------------------------------------------------------
    max_drawdown_pct = float(cache.get('max_drawdown_pct', 0.0))
    stats['max_drawdown_pct'] = max_drawdown_pct

    # ------------------------------------------------------------------
    # 3. Drawdown records — columns: Start Value, Valley Value,
    #    Start Index, End Index, Status
    # ------------------------------------------------------------------
    drawdowns_df = cache.get('drawdowns_df', pd.DataFrame())

    has_records = (
        not drawdowns_df.empty
        and 'Start Value' in drawdowns_df.columns
        and 'Valley Value' in drawdowns_df.columns
        and 'Start Index' in drawdowns_df.columns
        and 'End Index' in drawdowns_df.columns
    )

    if has_records:
        sv  = drawdowns_df['Start Value'].values.astype(np.float64)
        vv  = drawdowns_df['Valley Value'].values.astype(np.float64)

        # Drawdown % per period: (Start - Valley) / Start
        with np.errstate(divide='ignore', invalid='ignore'):
            dd_pct = np.where(sv > 0, (sv - vv) / sv * 100.0, 0.0)

        # Duration: End Index - Start Index (Active rows use last End Index)
        starts = pd.to_datetime(drawdowns_df['Start Index'])
        ends   = pd.to_datetime(drawdowns_df['End Index']).fillna(starts.max())
        duration_days = ((ends - starts).dt.total_seconds().values / 86400.0)

        max_dd_dur = float(cache.get('max_drawdown_duration_days', 0.0))
        stats.update({
            'max_drawdown_duration_days': max_dd_dur,
            'max_drawdown_days':          int(np.ceil(max_dd_dur)),
            'avg_drawdown_pct':           float(np.mean(dd_pct)),
            'avg_drawdown_days':          float(np.mean(duration_days)),
            'avg_drawdown_duration_days': float(np.mean(duration_days)),
            'drawdown_periods_count':     int(len(drawdowns_df)),
            'drawdown_duration_total':    float(np.sum(duration_days)),
        })

        if 'Status' in drawdowns_df.columns:
            active = drawdowns_df['Status'].values == 'Active'
            if np.any(active):
                idx = int(np.where(active)[0][-1])
                # current_drawdown_pct comes from drawdown_array (peak-to-now),
                # already set above — just set days here
                stats['current_drawdown_days'] = int(np.ceil(duration_days[idx]))
            else:
                stats['current_drawdown_days'] = 0
        else:
            stats['current_drawdown_days'] = 0
    else:
        stats.update(_empty_subset())

    # ------------------------------------------------------------------
    # 4. Recovery factor — total_return_pct pre-computed in cache (step 15)
    # ------------------------------------------------------------------
    total_return = cache.get('total_return_pct', 0.0) / 100.0
    max_dd_frac  = max_drawdown_pct / 100.0
    stats['recovery_factor'] = (
        float(abs(total_return) / max_dd_frac) if max_dd_frac > 0 else 0.0
    )

    return stats


def _empty_subset() -> Dict[str, Any]:
    return {
        'max_drawdown_duration_days': 0.0,
        'max_drawdown_days':          0,
        'avg_drawdown_pct':           0.0,
        'avg_drawdown_days':          0.0,
        'avg_drawdown_duration_days': 0.0,
        'current_drawdown_days':      0,
        'drawdown_periods_count':     0,
        'drawdown_duration_total':    0.0,
    }
