import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .performance import _calculate_monthly_breakdown, _calculate_recent_performance
from ..shared.utils import ANN_FACTOR

def _get_empty_profit_loss() -> Dict[str, Any]:
    return {
        'profit_factor': 0.0, 'payoff_ratio': 0.0, 'profit_loss_ratio': 0.0,
        'gain_to_pain_ratio': 0.0, 'cpc_ratio': 0.0, 'total_return_pct': 0.0,
        'monthly_returns_mean_pct': 0.0, 'monthly_returns_std_pct': 0.0,
        'best_month_return_pct': 0.0, 'worst_month_return_pct': 0.0,
        'positive_months_pct': 0.0, 'monthly_win_rate': 0.0,
        # Missing QuantStats profit_loss metrics
        'total_gains': 0.0, 'total_losses': 0.0, 'expected_return_annualized_pct': 0.0,
        'monthly_breakdown': {},
        # Recent performance metrics
        '1d_pnl_pct': 0.0, '15d_pnl_pct': 0.0, '30d_pnl_pct': 0.0, '45d_pnl_pct': 0.0, '60d_pnl_pct': 0.0
    }

def _calculate_profit_loss(
    df_ledger: pd.DataFrame, 
    true_initial: Optional[float] = None, 
    regime_analysis: bool = False,
    returns_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Enhanced profit/loss analysis matching VBT's approach.
    
    Args:
        df_ledger: Ledger DataFrame
        true_initial: Initial balance for calculations
        regime_analysis: If True, return only profit_factor and total_return_pct
        returns_df: Daily returns DataFrame for daily-based metrics
    """
    if df_ledger is None or len(df_ledger) == 0:
        return _get_empty_profit_loss()

    # Compute true initial if not given
    if true_initial is None:
        first_balance = float(df_ledger['balance'].iloc[0])
        if 'account_return_pct' in df_ledger.columns:
            first_ret = float(df_ledger['account_return_pct'].iloc[0]) / 100.0
            true_initial = first_balance / (1.0 + first_ret) if (1.0 + first_ret) != 0 else first_balance
        else:
            true_initial = first_balance

    balance_values = df_ledger['balance'].values
    if len(balance_values) < 1:
        return _get_empty_profit_loss()

    # Calculate balance_before per trade (balance before each trade)
    balance_before = np.empty(len(balance_values))
    if 'account_return_pct' in df_ledger.columns:
        first_ret = df_ledger['account_return_pct'].iloc[0] / 100.0
        balance_before[0] = balance_values[0] / (1.0 + first_ret) if (1.0 + first_ret) != 0 else balance_values[0]
    else:
        balance_before[0] = balance_values[0]
    if len(balance_values) > 1:
        balance_before[1:] = balance_values[:-1]

    # Use account_return_pct / 100 directly for returns (actual account impact)
    if 'account_return_pct' in df_ledger.columns:
        returns_clean = df_ledger['account_return_pct'].values / 100.0
    else:
        if len(balance_values) < 2:
            return _get_empty_profit_loss()
        prev_balance = balance_values[:-1]
        curr_balance = balance_values[1:]
        returns_clean = np.where(prev_balance != 0,
                                 (curr_balance - prev_balance) / prev_balance, 0.0)

    # PnL in dollar amounts per trade (matching VBT's trades_df['PnL'])
    pnl_per_trade = returns_clean * balance_before

    positive_mask = pnl_per_trade > 0
    negative_mask = pnl_per_trade < 0
    positive_pnl  = pnl_per_trade[positive_mask]
    negative_pnl  = pnl_per_trade[negative_mask]

    # profit_factor from dollar PnL (matching VBT)
    total_gains  = float(np.sum(positive_pnl)) if len(positive_pnl) > 0 else 0.0
    total_losses = float(abs(np.sum(negative_pnl))) if len(negative_pnl) > 0 else 0.0
    profit_factor = total_gains / total_losses if total_losses > 0 else 0.0

    # For ratio calculations use fractional returns
    positive_returns = returns_clean[positive_mask]
    negative_returns = returns_clean[negative_mask]

    avg_gain = np.mean(positive_returns) if len(positive_returns) > 0 else 0.0
    avg_loss = abs(np.mean(negative_returns)) if len(negative_returns) > 0 else 0.0
    payoff_ratio = avg_gain / avg_loss if avg_loss > 0 else 0.0

    # profit_loss_ratio: sum of winning returns / sum of losing returns
    # profit_loss_ratio: sum of winning returns / sum of losing returns (from fractional returns)
    total_gains_frac  = float(np.sum(positive_returns)) if len(positive_returns) > 0 else 0.0
    total_losses_frac = float(abs(np.sum(negative_returns))) if len(negative_returns) > 0 else 0.0
    profit_loss_ratio = total_gains_frac / total_losses_frac if total_losses_frac > 0 else 0.0

    # cpc_ratio: profit_factor * payoff_ratio (VBT formula)
    cpc_ratio = profit_factor * payoff_ratio
    
    # Daily-based metrics (if returns_df provided)
    if returns_df is not None and not returns_df.empty and 'portfolio_return' in returns_df.columns:
        daily_returns = returns_df['portfolio_return'].values
        
        # gain_to_pain_ratio: sum(positive daily returns) / sum(abs(negative daily returns))
        pos_daily = daily_returns[daily_returns > 0]
        neg_daily = daily_returns[daily_returns < 0]
        sum_pos = np.sum(pos_daily) if len(pos_daily) > 0 else 0.0
        sum_neg = abs(np.sum(neg_daily)) if len(neg_daily) > 0 else 0.0
        gain_to_pain_ratio = sum_pos / sum_neg if sum_neg > 0 else 0.0
        
        # expected_return_annualized_pct: CAGR from daily returns
        years = len(daily_returns) / ANN_FACTOR
        if years > 0 and len(daily_returns) > 0:
            expected_return_annualized = (np.prod(1.0 + daily_returns) ** (1.0 / years) - 1.0) * 100.0
        else:
            expected_return_annualized = 0.0
    else:
        # Fallback to trade-level calculation
        total_return_decimal = np.sum(returns_clean)
        gain_to_pain_ratio = total_return_decimal / total_losses_frac if total_losses_frac > 0 else 0.0
        
        mean_return = np.mean(returns_clean) if len(returns_clean) > 0 else 0.0
        expected_return_annualized = mean_return * ANN_FACTOR * 100
    
    # Total return
    total_return_pct = ((balance_values[-1] / true_initial) - 1) * 100
    
    # Early return for regime analysis - only return essential metrics
    if regime_analysis:
        return {
            'profit_factor': float(profit_factor),
            'total_return_pct': float(total_return_pct)
        }
    
    # Proper monthly analysis using actual data
    if 'exit_datetime' in df_ledger.columns and len(df_ledger) > 1:
        # Fast numpy month bucketing — no copy, no Period objects, no pd.to_datetime
        exit_months = df_ledger['exit_datetime'].values.astype('datetime64[M]')
        balance_vals = balance_values  # already extracted above

        unique_months, last_indices = np.unique(exit_months, return_index=True)
        # np.unique returns first occurrence; we need last occurrence per month
        # Flip, unique (gets last in original = first in flipped), flip indices back
        unique_months_last, first_indices_flipped = np.unique(exit_months[::-1], return_index=True)
        last_indices = len(exit_months) - 1 - first_indices_flipped
        last_indices.sort()
        monthly_balance_vals = balance_vals[last_indices]

        if len(monthly_balance_vals) > 1:
            # Calculate monthly returns
            monthly_returns = (monthly_balance_vals[1:] / monthly_balance_vals[:-1]) - 1

            if len(monthly_returns) > 0:
                # Monthly statistics
                monthly_returns_mean = np.mean(monthly_returns) * 100
                monthly_returns_std = np.std(monthly_returns, ddof=1) * 100 if len(monthly_returns) > 1 else 0.0

                # Best/worst month returns
                best_month_return = np.max(monthly_returns) * 100
                worst_month_return = np.min(monthly_returns) * 100

                # Positive months analysis
                positive_months_count = np.sum(monthly_returns > 0)
                positive_months_pct = (positive_months_count / len(monthly_returns)) * 100
                monthly_win_rate = positive_months_pct  # Already in percentage
            else:
                monthly_returns_mean = monthly_returns_std = 0.0
                best_month_return = worst_month_return = 0.0
                positive_months_pct = monthly_win_rate = 0.0
        else:
            # Not enough months for analysis
            monthly_returns_mean = total_return_pct
            monthly_returns_std = 0.0
            best_month_return = worst_month_return = total_return_pct
            positive_months_pct = 100.0 if total_return_pct > 0 else 0.0
            monthly_win_rate = positive_months_pct / 100
    else:
        # Fallback if no datetime column
        monthly_returns_mean = total_return_pct
        monthly_returns_std = 0.0
        best_month_return = worst_month_return = total_return_pct
        positive_months_pct = 100.0 if total_return_pct > 0 else 0.0
        monthly_win_rate = positive_months_pct / 100
    
    # Expected return annualized (QuantStats methodology)
    mean_return = np.mean(returns_clean) if len(returns_clean) > 0 else 0.0
    expected_return_annualized = mean_return * ANN_FACTOR  # Standard daily annualization
    
    return {
        'profit_factor': float(profit_factor),
        'payoff_ratio': float(payoff_ratio),
        'profit_loss_ratio': float(profit_loss_ratio),
        'gain_to_pain_ratio': float(gain_to_pain_ratio),
        'cpc_ratio': float(cpc_ratio),
        'total_return_pct': float(total_return_pct),
        'monthly_returns_mean_pct': float(monthly_returns_mean),
        'monthly_returns_std_pct': float(monthly_returns_std),
        'best_month_return_pct': float(best_month_return),
        'worst_month_return_pct': float(worst_month_return),
        'positive_months_pct': float(positive_months_pct),
        'monthly_win_rate': float(monthly_win_rate),
        
        # Missing QuantStats profit_loss metrics
        'total_gains': float(total_gains),
        'total_losses': float(total_losses),
        'expected_return_annualized_pct': float(expected_return_annualized),
        
        # Monthly breakdown dictionary (VBT style - uses daily returns)
        'monthly_breakdown': _calculate_monthly_breakdown(df_ledger, returns_df=returns_df),
        
        # Recent performance metrics (1d, 7d, 15d, 30d, 45d, 60d, 90d PnL)
        **_calculate_recent_performance(df_ledger, returns_df=returns_df)
    }
