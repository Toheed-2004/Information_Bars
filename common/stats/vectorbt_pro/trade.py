import numpy as np
import pandas as pd
from typing import Dict, Any

from ..shared.utils import _max_consecutive_numpy


def _get_default_stats_dict() -> Dict[str, Any]:
    """Return dictionary with all trade stats initialized to 0"""
    return {
        'total_trades': 0,
        'win_rate_pct': 0.0,
        'loss_rate_pct': 0.0,
        'best_trade_pct': 0.0,
        'worst_trade_pct': 0.0,
        'winning_trades': 0,
        'losing_trades': 0,
        'avg_winning_trade_pct': 0.0,
        'avg_losing_trade_pct': 0.0,
        'avg_winning_trade_duration_days': 0.0,
        'avg_losing_trade_duration_days': 0.0,
        'consecutive_wins': 0,
        'consecutive_losses': 0,
        'avg_duration_trades': 0.0,
        'total_pnl_pct': 0.0,
        'trade_duration_std': 0.0,
        'trade_return_std': 0.0,
        'sqn': 0.0,
        'edge_ratio': 0.0,
        'max_winning_streak': 0,
        'max_losing_streak': 0,
        'win_loss_ratio': 0.0,
        'outlier_win_ratio': 0.0,
        'outlier_loss_ratio': 0.0,
        'avg_return_all_trades': 0.0,
        'geometric_mean_returns': 0.0,
    }

def _calculate_return_stats(returns_pct: np.ndarray) -> Dict[str, Any]:
    """Calculate return-based statistics: best, worst, avg wins/losses"""
    stats = {}

    if len(returns_pct) == 0:
        return {
            'best_trade_pct': 0.0,
            'worst_trade_pct': 0.0,
            'avg_winning_trade_pct': 0.0,
            'avg_losing_trade_pct': 0.0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate_pct': 0.0,
            'loss_rate_pct': 0.0,
        }

    # Boolean masks for wins/losses (in decimal form)
    wins_mask = returns_pct > 0
    losses_mask = returns_pct < 0

    winning_trades = int(np.sum(wins_mask))
    losing_trades = int(np.sum(losses_mask))
    total_trades = len(returns_pct)

    # Best and worst trades (convert decimal to percentage)
    stats['best_trade_pct'] = float(np.max(returns_pct) * 100) if total_trades > 0 else 0.0
    stats['worst_trade_pct'] = float(np.min(returns_pct) * 100) if total_trades > 0 else 0.0

    # Average winning and losing trades (convert to percentage)
    stats['avg_winning_trade_pct'] = float(np.mean(returns_pct[wins_mask]) * 100) if winning_trades > 0 else 0.0
    stats['avg_losing_trade_pct'] = float(np.mean(returns_pct[losses_mask]) * 100) if losing_trades > 0 else 0.0

    stats['winning_trades'] = winning_trades
    stats['losing_trades'] = losing_trades
    stats['win_rate_pct'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    stats['loss_rate_pct'] = (losing_trades / total_trades * 100) if total_trades > 0 else 0.0

    return stats

def _calculate_duration_stats(entry_times: Any, exit_times: Any, returns_pct: np.ndarray) -> Dict[str, Any]:
    """Calculate duration-based statistics in days"""
    stats = {}

    if len(returns_pct) == 0 or entry_times is None or exit_times is None:
        return {
            'avg_winning_trade_duration_days': 0.0,
            'avg_losing_trade_duration_days': 0.0,
            'avg_duration_trades': 0.0,
            'trade_duration_std': 0.0,
        }

    try:
        # Calculate durations in days
        durations = (pd.to_datetime(exit_times) - pd.to_datetime(entry_times)).dt.total_seconds() / 86400.0
        durations = durations.values if hasattr(durations, 'values') else durations

        if len(durations) == 0:
            return {
                'avg_winning_trade_duration_days': 0.0,
                'avg_losing_trade_duration_days': 0.0,
                'avg_duration_trades': 0.0,
                'trade_duration_std': 0.0,
            }

        wins_mask = returns_pct > 0
        losses_mask = returns_pct < 0

        winning_durations = durations[wins_mask] if np.any(wins_mask) else np.array([])
        losing_durations = durations[losses_mask] if np.any(losses_mask) else np.array([])

        avg_win_duration = float(np.mean(winning_durations)) if len(winning_durations) > 0 else 0.0
        avg_loss_duration = float(np.mean(losing_durations)) if len(losing_durations) > 0 else 0.0

        stats['avg_winning_trade_duration_days'] = avg_win_duration
        stats['avg_losing_trade_duration_days'] = avg_loss_duration
        stats['avg_duration_trades'] = float(np.mean(durations)) if len(durations) > 0 else 0.0
        stats['trade_duration_std'] = float(np.std(durations)) if len(durations) > 1 else 0.0

    except Exception:
        return {
            'avg_winning_trade_duration_days': 0.0,
            'avg_losing_trade_duration_days': 0.0,
            'avg_duration_trades': 0.0,
            'trade_duration_std': 0.0,
        }

    return stats

def _calculate_streak_stats(returns_pct: np.ndarray) -> Dict[str, Any]:
    """Calculate consecutive wins/losses and max streaks"""
    stats = {}

    if len(returns_pct) == 0:
        return {
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_winning_streak': 0,
            'max_losing_streak': 0,
        }

    wins_mask = returns_pct > 0
    losses_mask = returns_pct < 0

    consecutive_wins = _max_consecutive_numpy(wins_mask)
    consecutive_losses = _max_consecutive_numpy(losses_mask)

    stats['consecutive_wins'] = consecutive_wins
    stats['consecutive_losses'] = consecutive_losses
    stats['max_winning_streak'] = consecutive_wins
    stats['max_losing_streak'] = consecutive_losses

    return stats

def _calculate_advanced_stats(returns_pct: np.ndarray) -> Dict[str, Any]:
    """Calculate SQN, edge ratio, win/loss ratio, outliers, geometric mean, profit factor"""
    stats = {}

    if len(returns_pct) == 0:
        return {
            'sqn': 0.0,
            'edge_ratio': 0.0,
            'win_loss_ratio': 0.0,
            'outlier_win_ratio': 0.0,
            'outlier_loss_ratio': 0.0,
            'geometric_mean_returns': 0.0,
        }

    wins_mask = returns_pct > 0
    losses_mask = returns_pct < 0

    winning_trades = int(np.sum(wins_mask))
    losing_trades = int(np.sum(losses_mask))

    # Profit Factor (gross profit / gross loss) - calculated from returns
    winning_returns = returns_pct[wins_mask]
    losing_returns = returns_pct[losses_mask]
    
    total_gains = np.sum(winning_returns) if len(winning_returns) > 0 else 0.0
    total_losses = np.abs(np.sum(losing_returns)) if len(losing_returns) > 0 else 0.0
    
   
    # SQN (System Quality Number)
    mean_return = np.mean(returns_pct)
    std_return = np.std(returns_pct)
    if std_return > 0 and len(returns_pct) > 0:
        stats['sqn'] = float(mean_return / std_return * np.sqrt(len(returns_pct)))
    else:
        stats['sqn'] = 0.0

    # Edge Ratio
    avg_win = np.mean(returns_pct[wins_mask]) if winning_trades > 0 else 0.0
    avg_loss = np.abs(np.mean(returns_pct[losses_mask])) if losing_trades > 0 else 1.0
    stats['edge_ratio'] = float(avg_win / avg_loss) if avg_loss != 0 else 0.0

    # Win/Loss Ratio
    stats['win_loss_ratio'] = float(winning_trades / losing_trades) if losing_trades > 0 else 0.0

    # Outlier Ratios (based on IQR method)
    q3, q1 = np.percentile(returns_pct, [75, 25])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr

    total_trades = len(returns_pct)
    outlier_wins = np.sum(returns_pct > upper_bound)
    outlier_losses = np.sum(returns_pct < lower_bound)

    stats['outlier_win_ratio'] = float(outlier_wins / total_trades) if total_trades > 0 else 0.0
    stats['outlier_loss_ratio'] = float(outlier_losses / total_trades) if total_trades > 0 else 0.0

    # Geometric Mean Returns (convert decimal to percentage)
    with np.errstate(invalid='ignore', divide='ignore'):
        gross_returns = 1 + returns_pct
        gross_returns = np.maximum(gross_returns, 1e-10)  # Avoid log(0)
        try:
            if np.all(gross_returns > 0):
                geometric_mean = np.prod(gross_returns) ** (1 / len(gross_returns)) - 1
                stats['geometric_mean_returns'] = float(geometric_mean * 100)
            else:
                stats['geometric_mean_returns'] = 0.0
        except (OverflowError, ValueError):
            stats['geometric_mean_returns'] = 0.0

    return stats

def _extract_compatibility_stats_vectorized(pf, cache, ledger_input=None) -> Dict[str, Any]:
    """
    UNIFIED: Extract all trade statistics using VBT portfolio or ledger fallback.
    Calculates all 26 metrics that match custom stats implementation.
    """
    stats = _get_default_stats_dict()

    try:
        # Get returns and trade data
        returns_pct = None
        entry_times = None
        exit_times = None
        total_return = 0.0

        # Primary: Try to extract from portfolio trades
        trades_df = cache.get('trades_df', pd.DataFrame())
        
        # Try VBT trades directly if cache is empty
        if trades_df.empty and hasattr(pf, 'trades'):
            try:
                trades_df = pf.trades.records_readable
                if not trades_df.empty:
                    cache['trades_df'] = trades_df  # Update cache
            except:
                pass
        
        if not trades_df.empty and 'Return' in trades_df.columns:
            returns_pct = trades_df['Return'].values
            # Extract Entry/Exit Index columns (they exist as columns in VBT trades dataframe)
            entry_times = trades_df['Entry Index'].values if 'Entry Index' in trades_df.columns else None
            exit_times = trades_df['Exit Index'].values if 'Exit Index' in trades_df.columns else None
            
            # Try to get durations directly from VBT if times are None
            if entry_times is None or exit_times is None:
                try:
                    if hasattr(pf, 'trades'):
                        # Get duration from VBT trades object
                        if hasattr(pf.trades, 'duration'):
                            durations_td = pf.trades.duration
                            if hasattr(durations_td, 'values'):
                                # Convert timedelta to days
                                durations_days = durations_td.dt.total_seconds().values / 86400.0 if hasattr(durations_td, 'dt') else durations_td.values / 86400.0
                                
                                # Calculate duration stats directly
                                wins_mask = returns_pct > 0
                                losses_mask = returns_pct < 0
                                winning_durations = durations_days[wins_mask] if np.any(wins_mask) else np.array([])
                                losing_durations = durations_days[losses_mask] if np.any(losses_mask) else np.array([])
                                
                                stats['avg_winning_trade_duration_days'] = float(np.mean(winning_durations)) if len(winning_durations) > 0 else 0.0
                                stats['avg_losing_trade_duration_days'] = float(np.mean(losing_durations)) if len(losing_durations) > 0 else 0.0
                                stats['avg_duration_trades'] = float(np.mean(durations_days)) if len(durations_days) > 0 else 0.0
                                stats['trade_duration_std'] = float(np.std(durations_days)) if len(durations_days) > 1 else 0.0
                except:
                    pass

            # Calculate total PnL from portfolio value
            value_array = cache.get('value_array', np.array([]))
            if len(value_array) > 1:
                total_return = ((value_array[-1] / value_array[0]) - 1) * 100
            else:
                # Sum returns in decimal form and convert to percentage
                total_return = np.sum(returns_pct) * 100

        # Fallback: Try ledger_input if portfolio data unavailable
        if (returns_pct is None or len(returns_pct) == 0) and ledger_input is not None and not ledger_input.empty:
            # Prefer trade_return_pct (% return on invested capital) for trade-level stats
            if 'trade_return_pct' in ledger_input.columns:
                returns_pct = ledger_input['trade_return_pct'].values / 100.0
            elif 'account_return_pct' in ledger_input.columns:
                returns_pct = ledger_input['account_return_pct'].values / 100.0

            if returns_pct is not None and 'entry_datetime' in ledger_input.columns and 'exit_datetime' in ledger_input.columns:
                entry_times = pd.to_datetime(ledger_input['entry_datetime']).values
                exit_times = pd.to_datetime(ledger_input['exit_datetime']).values
                total_return = np.sum(returns_pct) * 100

        # If we have returns data, calculate all stats
        if returns_pct is not None and len(returns_pct) > 0:
            total_trades = len(returns_pct)

            # Calculate return-based stats
            stats.update(_calculate_return_stats(returns_pct))

            # Calculate duration stats directly if not already set (e.g., from VBT)
            if stats['avg_winning_trade_duration_days'] == 0.0 and entry_times is not None and exit_times is not None:
                # Compute durations in days
                # First attempt with numpy (works if entry_times/exit_times are datetime64)
                try:
                    durations = (exit_times - entry_times) / np.timedelta64(1, 'D')
                except:
                    # Fallback using pandas
                    entry_dt = pd.to_datetime(entry_times)
                    exit_dt = pd.to_datetime(exit_times)
                    durations = (exit_dt - entry_dt).dt.total_seconds().values / 86400.0

                wins_mask = returns_pct > 0
                losses_mask = returns_pct < 0
                winning_durations = durations[wins_mask]
                losing_durations = durations[losses_mask]

                stats['avg_winning_trade_duration_days'] = float(np.mean(winning_durations)) if len(winning_durations) > 0 else 0.0
                stats['avg_losing_trade_duration_days'] = float(np.mean(losing_durations)) if len(losing_durations) > 0 else 0.0
                stats['avg_duration_trades'] = float(np.mean(durations))
                stats['trade_duration_std'] = float(np.std(durations)) if len(durations) > 1 else 0.0

            # Calculate streak stats
            stats.update(_calculate_streak_stats(returns_pct))

            # Calculate advanced stats 
            stats.update(_calculate_advanced_stats(returns_pct))

            # Trade return std (in percentage)
            stats['trade_return_std'] = float(np.std(returns_pct) * 100) if len(returns_pct) > 1 else 0.0

            # Total trades and total PnL
            stats['total_trades'] = total_trades
            stats['total_pnl_pct'] = float(total_return)

            # Average return of all trades (in percentage)
            stats['avg_return_all_trades'] = float(np.mean(returns_pct) * 100)

        # Add R-squared from cached value array (for compatibility)
        value_array = cache.get('value_array', np.array([]))
    except Exception as e:
        # Return default stats on error
        stats = _get_default_stats_dict()

    return stats

