"""
Fully vectorized profit/loss analysis for custom_1.
Operates directly on LedgerArrays — batched computation for all strategies.
"""
import numpy as np
from .config import ANN_FACTOR

NS_PER_DAY = np.int64(86_400_000_000_000)
_RECENT_DAYS = [1, 15, 30, 45, 60]


def calculate_profit_loss(stacked, regime_analysis: bool = False) -> np.ndarray:
    """
    Calculate profit/loss analysis for ALL strategies in batched mode.
    
    Parameters
    ----------
    stacked : LedgerArrays namedtuple
    regime_analysis : bool
        If True, return only profit_factor and total_return_pct
    
    Returns
    -------
    Structured array (n_strats,) with all profit/loss metrics
    """
    n_strats = len(stacked.names)
    
    # Pre-allocate
    profit_factor = np.zeros(n_strats)
    total_return = np.zeros(n_strats)
    
    if not regime_analysis:
        payoff_ratio = np.zeros(n_strats)
        profit_loss_ratio = np.zeros(n_strats)
        gain_to_pain = np.zeros(n_strats)
        cpc_ratio = np.zeros(n_strats)
        monthly_mean = np.zeros(n_strats)
        monthly_std = np.zeros(n_strats)
        best_month = np.zeros(n_strats)
        worst_month = np.zeros(n_strats)
        pos_months_pct = np.zeros(n_strats)
        monthly_wr = np.zeros(n_strats)
        total_gains = np.zeros(n_strats)
        total_losses = np.zeros(n_strats)
        expected_ret_ann = np.zeros(n_strats)
        recent_1d = np.zeros(n_strats)
        recent_15d = np.zeros(n_strats)
        recent_30d = np.zeros(n_strats)
        recent_45d = np.zeros(n_strats)
        recent_60d = np.zeros(n_strats)
    
    for s in range(n_strats):
        n = int(stacked.lengths[s])
        if n == 0:
            continue
        
        acc_ret = stacked.numeric_3d[s, :n, 5] / 100.0  # COL_ACC_RET
        balance = stacked.numeric_3d[s, :n, 7]          # COL_BALANCE
        exit_ts = stacked.datetime_3d[s, :n, 1]
        
        denom = 1.0 + acc_ret[0]
        initial = balance[0] / denom if denom != 0.0 else balance[0]
        
        balance_before = np.empty(n, dtype=np.float64)
        balance_before[0] = initial
        if n > 1:
            balance_before[1:] = balance[:-1]
        
        pnl = acc_ret * balance_before
        
        pos_mask = pnl > 0
        neg_mask = pnl < 0
        pos_pnl = pnl[pos_mask]
        neg_pnl = pnl[neg_mask]
        
        tg = np.sum(pos_pnl) if len(pos_pnl) > 0 else 0.0
        tl = np.abs(np.sum(neg_pnl)) if len(neg_pnl) > 0 else 0.0
        profit_factor[s] = tg / tl if tl > 0 else 0.0
        total_return[s] = (balance[-1] / initial - 1) * 100
        
        if not regime_analysis:
            total_gains[s] = tg
            total_losses[s] = tl
            
            pos_ret = acc_ret[pos_mask]
            neg_ret = acc_ret[neg_mask]
            
            avg_gain = np.mean(pos_ret) if len(pos_ret) > 0 else 0.0
            avg_loss = np.abs(np.mean(neg_ret)) if len(neg_ret) > 0 else 0.0
            payoff_ratio[s] = avg_gain / avg_loss if avg_loss > 0 else 0.0
            
            tg_frac = np.sum(pos_ret) if len(pos_ret) > 0 else 0.0
            tl_frac = np.abs(np.sum(neg_ret)) if len(neg_ret) > 0 else 0.0
            profit_loss_ratio[s] = tg_frac / tl_frac if tl_frac > 0 else 0.0
            
            cpc_ratio[s] = profit_factor[s] * payoff_ratio[s]
            
            total_ret_dec = np.sum(acc_ret)
            gain_to_pain[s] = total_ret_dec / tl_frac if tl_frac > 0 else 0.0
            
            expected_ret_ann[s] = np.mean(acc_ret) * ANN_FACTOR * 100
            
            # Monthly
            exit_days_ns = (exit_ts // NS_PER_DAY) * NS_PER_DAY
            exit_months = exit_days_ns.astype('datetime64[ns]').astype('datetime64[M]').view('int64')
            
            _, first_in_rev = np.unique(exit_months[::-1], return_index=True)
            last_indices = np.sort(n - 1 - first_in_rev)
            monthly_bal = balance[last_indices]
            
            if len(monthly_bal) > 1:
                monthly_ret = monthly_bal[1:] / monthly_bal[:-1] - 1.0
                monthly_mean[s] = np.mean(monthly_ret) * 100
                monthly_std[s] = np.std(monthly_ret, ddof=1) * 100 if len(monthly_ret) > 1 else 0.0
                best_month[s] = np.max(monthly_ret) * 100
                worst_month[s] = np.min(monthly_ret) * 100
                pos_m = np.sum(monthly_ret > 0)
                pos_months_pct[s] = pos_m / len(monthly_ret) * 100
                monthly_wr[s] = pos_months_pct[s] / 100.0
            else:
                monthly_mean[s] = total_return[s]
                best_month[s] = worst_month[s] = total_return[s]
                pos_months_pct[s] = 100.0 if total_return[s] > 0 else 0.0
                monthly_wr[s] = pos_months_pct[s] / 100.0
            
            # # Recent
            # exit_day_int = exit_ts // NS_PER_DAY
            # last_day = exit_day_int[-1]
            
            # for days, arr in zip(_RECENT_DAYS, [recent_1d, recent_15d, recent_30d, recent_45d, recent_60d]):
            #     cutoff = last_day - days
            #     idx_arr = np.searchsorted(exit_day_int, cutoff, side='left')
            #     if idx_arr < n:
            #         start_bal = balance[idx_arr]
            #         arr[s] = (balance[-1] / start_bal - 1) * 100 if start_bal != 0 else 0.0
    
    if regime_analysis:
        dtype = [('profit_factor', 'f8'), ('total_return_pct', 'f8')]
        result = np.zeros(n_strats, dtype=dtype)
        result['profit_factor'] = profit_factor
        result['total_return_pct'] = total_return
    else:
        dtype = [
            ('profit_factor', 'f8'), ('payoff_ratio', 'f8'), ('profit_loss_ratio', 'f8'),
            ('gain_to_pain_ratio', 'f8'), ('cpc_ratio', 'f8'), ('total_return_pct', 'f8'),('total_gains', 'f8'), ('total_losses', 'f8'),('expected_return_annualized_pct', 'f8'),
            ('monthly_returns_mean_pct', 'f8'), ('monthly_returns_std_pct', 'f8'),
            ('best_month_return_pct', 'f8'), ('worst_month_return_pct', 'f8'),
            ('positive_months_pct', 'f8'), ('monthly_win_rate', 'f8'),
        
        ]
        result = np.zeros(n_strats, dtype=dtype)
        result['profit_factor'] = profit_factor
        result['payoff_ratio'] = payoff_ratio
        result['profit_loss_ratio'] = profit_loss_ratio
        result['gain_to_pain_ratio'] = gain_to_pain
        result['cpc_ratio'] = cpc_ratio
        result['total_return_pct'] = total_return
        result['monthly_returns_mean_pct'] = monthly_mean
        result['monthly_returns_std_pct'] = monthly_std
        result['best_month_return_pct'] = best_month
        result['worst_month_return_pct'] = worst_month
        result['positive_months_pct'] = pos_months_pct
        result['monthly_win_rate'] = monthly_wr
        result['total_gains'] = total_gains
        result['total_losses'] = total_losses
        result['expected_return_annualized_pct'] = expected_ret_ann
    
    return result
