"""
Plotting module for candlestick charts with technical indicators
Enhanced with indicator grouping and TradingView-style layout
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, List, Dict, Set, Tuple
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


class IndicatorPlotter:
    """Plot OHLCV data with technical indicators on candlestick charts
    with intelligent grouping of related indicators"""
    
    # ================================================================
    # INDICATOR GROUPING RULES BASED ON TALIB REGISTRY
    # ================================================================
    
    # Indicators that have MULTIPLE outputs and should be plotted together
    MULTI_OUTPUT_GROUPS: Dict[str, List[str]] = {
        # MACD Family - all 3 outputs together
        'MACD': ['macd', 'macdsignal', 'macdhist'],
        'MACDEXT': ['macd', 'macdsignal', 'macdhist'],
        'MACDFIX': ['macd', 'macdsignal', 'macdhist'],
        
        # Stochastic Family
        'STOCH': ['slowk', 'slowd'],
        'STOCHF': ['fastk', 'fastd'],
        'STOCHRSI': ['fastk', 'fastd'],
        
        # Bollinger Bands - all 3 outputs together (overlay)
        'BBANDS': ['upperband', 'middleband', 'lowerband'],
        
        # Aroon
        'AROON': ['aroondown', 'aroonup'],
        
        # ADX Family
        'ADX': ['adx'],
        'ADXR': ['adxr'],
        'DX': ['dx'],
        
        # Hilbert Transform
        'HT_PHASOR': ['inphase', 'quadrature'],
        'HT_SINE': ['sine', 'leadsine'],
        
        # MAMA
        'MAMA': ['mama', 'fama'],
        
        # Min/Max
        'MINMAX': ['minmax_min', 'minmax_max'],
        'MINMAXINDEX': ['minmaxindex_min', 'minmaxindex_max'],
    }
    
    # Indicators that should be plotted on MAIN CHART (overlay)
    OVERLAY_INDICATORS: Set[str] = {
        # Moving averages (single output)
        'SMA', 'EMA', 'WMA', 'HMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'T3', 'MA', 'MAVP',
        'MIDPOINT', 'MIDPRICE', 'HT_TRENDLINE',
        
        # Bollinger Bands components (part of BBANDS group)
        'upperband', 'middleband', 'lowerband',
        
        # Parabolic SAR
        'SAR', 'SAREXT',
        
        # MAMA components (part of MAMA group)
        'mama', 'fama',
        
        # Price-based
        'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE',
    }
    
    # Single output oscillators (each gets own subplot)
    OSCILLATOR_INDICATORS: Set[str] = {
        'RSI', 'CCI', 'CMO', 'BOP', 'MFI', 'WILLR',
        'APO', 'PPO', 'TRIX', 'ULTOSC',
        'ROC', 'ROCP', 'ROCR', 'ROCR100', 'MOM',
        'AROONOSC',
        'ATR', 'NATR', 'TRANGE',
        'HT_DCPERIOD', 'HT_DCPHASE', 'HT_TRENDMODE',
        'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE',
        'STDDEV', 'TSF', 'VAR',
        'BETA', 'CORREL',
    }
    
    # Volume indicators
    VOLUME_INDICATORS: Set[str] = {'AD', 'ADOSC', 'OBV'}
    
    def __init__(self, figsize: tuple = (16, 10), style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize plotter"""
        self.figsize = figsize
        try:
            plt.style.use(style)
        except Exception:
            logger.warning(f"Style '{style}' not available, using default")
    
    def plot(self, df: pd.DataFrame, title: str = "Price and Indicators", 
             overlay_indicators: Optional[List[str]] = None,
             separate_indicators: Optional[List[str]] = None,
             save_path: Optional[str] = None) -> plt.Figure:
        """Plot candlestick chart with indicators grouped appropriately
        
        Args:
            df: DataFrame with OHLCV + indicator columns (lowercase: open, high, low, close, volume)
            title: Chart title
            overlay_indicators: List of indicators to plot on main chart
            separate_indicators: List of indicators to plot in separate subplots
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure object
        """
        # Validate input
        required_cols = ['open', 'high', 'low', 'close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")
        
        # Get all indicator columns
        ohlcv_cols = required_cols + (['volume'] if 'volume' in df.columns else [])
        all_indicators = [col for col in df.columns if col not in ohlcv_cols]
        
        # Group indicators based on TALib output structure
        grouped_indicators = self._group_indicators_by_talib_structure(all_indicators)
        
        # Categorize groups
        overlay_groups, separate_groups, volume_groups = self._categorize_groups(
            grouped_indicators, overlay_indicators, separate_indicators
        )
        
        # Create subplots layout
        n_subplots = 1 + len(separate_groups)
        if volume_groups:
            n_subplots += len(volume_groups)
        
        fig, axes = self._create_subplots_layout(n_subplots)
        
        # Plot candlesticks on main axis
        self._plot_candlesticks(axes[0], df)
        
        # Plot overlay groups on main axis
        self._plot_overlay_groups(axes[0], df, overlay_groups)
        
        axes[0].set_ylabel('Price', fontsize=11, fontweight='bold')
        axes[0].set_title(title, fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot separate groups in their own subplots
        current_axis = 1
        
        # Plot volume groups first (if any)
        for group_name, indicators in volume_groups.items():
            self._plot_volume_group(axes[current_axis], df, group_name, indicators)
            current_axis += 1
        
        # Plot other separate groups
        for group_name, indicators in separate_groups.items():
            self._plot_separate_group(axes[current_axis], df, group_name, indicators)
            current_axis += 1
        
        # Configure x-axis
        self._configure_x_axis(axes[-1], df)
        
        # Add legends
        for ax in axes:
            if ax.get_legend_handles_labels()[0]:  # Check if there are any legend entries
                ax.legend(loc='best', fontsize=9, framealpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Chart saved to {save_path}")
        
        return fig
    
    def _group_indicators_by_talib_structure(self, indicators: List[str]) -> Dict[str, List[str]]:
        """Group indicators based on TALib's output structure"""
        groups = {}
        used_indicators = set()
        
        # First, identify and group multi-output indicators
        for indicator in indicators:
            if indicator in used_indicators:
                continue
                
            indicator_lower = indicator.lower()
            
            # Check for multi-output patterns
            for group_name, output_names in self.MULTI_OUTPUT_GROUPS.items():
                group_lower = group_name.lower()
                
                # Check if this indicator matches any output pattern
                for output in output_names:
                    # Look for pattern: output_param1_param2...
                    # Example: macd_12_26_9, slowk_5_3_3, etc.
                    if output in indicator_lower:
                        # Find all related indicators with same parameters
                        base_pattern = self._extract_base_pattern(indicator, output)
                        related_indicators = []
                        
                        for other_ind in indicators:
                            if other_ind in used_indicators:
                                continue
                                
                            other_lower = other_ind.lower()
                            # Check if same output type and same parameters
                            for other_output in output_names:
                                if other_output in other_lower:
                                    other_base = self._extract_base_pattern(other_ind, other_output)
                                    if other_base == base_pattern:
                                        related_indicators.append(other_ind)
                                        used_indicators.add(other_ind)
                                        break
                        
                        if related_indicators:
                            # Sort by output order from MULTI_OUTPUT_GROUPS
                            sorted_indicators = []
                            for output_name in output_names:
                                for ind in related_indicators:
                                    if output_name in ind.lower():
                                        sorted_indicators.append(ind)
                                        break
                            
                            group_key = f"{group_lower}_{base_pattern}"
                            groups[group_key] = sorted_indicators
                        break
                if indicator in used_indicators:
                    break
        
        # Add remaining single indicators
        for indicator in indicators:
            if indicator not in used_indicators:
                groups[indicator] = [indicator]
                used_indicators.add(indicator)
        
        return groups
    
    def _extract_base_pattern(self, indicator: str, output_name: str) -> str:
        """Extract the parameter pattern from indicator name"""
        # Remove the output name and get parameters
        pattern = indicator.lower().replace(output_name.lower(), '')
        pattern = pattern.strip('_')
        return pattern
    
    def _categorize_groups(self, groups: Dict[str, List[str]], 
                          overlay_indicators: Optional[List[str]] = None,
                          separate_indicators: Optional[List[str]] = None) -> Tuple[Dict, Dict, Dict]:
        """Categorize groups into overlay, separate, and volume"""
        overlay_groups = {}
        separate_groups = {}
        volume_groups = {}
        
        for group_name, indicators in groups.items():
            group_lower = group_name.lower()
            
            # User-specified override
            if overlay_indicators and any(ind in overlay_indicators for ind in indicators):
                overlay_groups[group_name] = indicators
                continue
            
            if separate_indicators and any(ind in separate_indicators for ind in indicators):
                separate_groups[group_name] = indicators
                continue
            
            # Check if it's a volume indicator
            is_volume = False
            for indicator in indicators:
                indicator_lower = indicator.lower()
                if any(vol in indicator_lower for vol in ['ad', 'adosc', 'obv']):
                    is_volume = True
                    break
            
            if is_volume:
                volume_groups[group_name] = indicators
                continue
            
            # Check if it should be overlay
            is_overlay = False
            for indicator in indicators:
                indicator_lower = indicator.lower()
                
                # Check overlay patterns
                if any(overlay in indicator_lower for overlay in [
                    'sma', 'ema', 'wma', 'hma', 'dema', 'tema', 'trima', 'kama', 't3', 'ma_',
                    'upperband', 'middleband', 'lowerband', 'bb_',
                    'sar', 'sarext',
                    'mama', 'fama',
                    'avgprice', 'medprice', 'typprice', 'wclprice',
                    'midpoint', 'midprice', 'ht_trendline'
                ]):
                    is_overlay = True
                    break
            
            if is_overlay:
                overlay_groups[group_name] = indicators
                continue
            
            # Everything else is separate
            separate_groups[group_name] = indicators
        
        return overlay_groups, separate_groups, volume_groups
    
    def _create_subplots_layout(self, n_subplots: int) -> Tuple[plt.Figure, List]:
        """Create subplots with appropriate height ratios"""
        if n_subplots == 1:
            fig, axes = plt.subplots(1, 1, figsize=self.figsize)
            return fig, [axes]
        
        # Height ratios: price chart is taller, others are shorter
        height_ratios = [3]  # Price chart
        height_ratios.extend([1] * (n_subplots - 1))  # Other indicators
        
        fig, axes = plt.subplots(
            n_subplots, 1,
            figsize=(self.figsize[0], self.figsize[1] + (n_subplots - 1) * 2),
            sharex=True,
            gridspec_kw={'height_ratios': height_ratios}
        )
        
        return fig, axes if isinstance(axes, np.ndarray) else [axes]
    
    def _plot_candlesticks(self, ax, df: pd.DataFrame):
        """Plot candlestick chart"""
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        
        width = 0.6
        x_range = range(len(df))
        
        # Plot wicks (high-low lines)
        for i in x_range:
            color = '#26A69A' if close_prices[i] >= open_prices[i] else '#EF5350'
            ax.plot([i, i], [low_prices[i], high_prices[i]], 
                   color=color, linewidth=0.8, alpha=0.8)
        
        # Plot bodies (open-close rectangles)
        for i in x_range:
            if close_prices[i] >= open_prices[i]:
                color = '#26A69A'  # Green
                height = close_prices[i] - open_prices[i]
                bottom = open_prices[i]
            else:
                color = '#EF5350'  # Red
                height = open_prices[i] - close_prices[i]
                bottom = close_prices[i]
            
            if height > 0:
                rect = patches.Rectangle(
                    (i - width / 2, bottom), width, height,
                    linewidth=0.8, edgecolor=color, facecolor=color,
                    alpha=0.8, zorder=2
                )
                ax.add_patch(rect)
            else:
                # Doji
                ax.plot([i - width / 2, i + width / 2], [open_prices[i], open_prices[i]], 
                       color='black', linewidth=1, alpha=0.5)
        
        # Set limits
        ax.set_xlim(-1, len(df))
        price_min, price_max = low_prices.min(), high_prices.max()
        ax.set_ylim(price_min * 0.995, price_max * 1.005)
    
    def _plot_overlay_groups(self, ax, df: pd.DataFrame, overlay_groups: Dict[str, List[str]]):
        """Plot overlay indicator groups on main chart"""
        x_range = range(len(df))
        
        for group_name, indicators in overlay_groups.items():
            group_lower = group_name.lower()
            
            # Special handling for Bollinger Bands
            if any('bbands' in group_lower or 'upperband' in ind.lower() or 'lowerband' in ind.lower() 
                   for ind in indicators):
                self._plot_bbands_group(ax, df, group_name, indicators)
                continue
            
            # Special handling for moving averages
            if any('sma' in ind.lower() or 'ema' in ind.lower() or 'ma_' in ind.lower() 
                   for ind in indicators):
                self._plot_ma_group(ax, df, group_name, indicators)
                continue
            
            # Special handling for Parabolic SAR
            if any('sar' in ind.lower() for ind in indicators):
                self._plot_sar_group(ax, df, group_name, indicators)
                continue
            
            # Default overlay plotting
            for indicator in indicators:
                values = df[indicator].values
                color = self._get_indicator_color(indicator)
                ax.plot(x_range, values, color=color, linewidth=1.5,
                       label=indicator, alpha=0.8, zorder=1)
    
    def _plot_bbands_group(self, ax, df: pd.DataFrame, group_name: str, indicators: List[str]):
        """Plot Bollinger Bands group together on main chart"""
        x_range = range(len(df))
        
        # Find upper, middle, lower bands
        upper_band = None
        middle_band = None
        lower_band = None
        
        for indicator in indicators:
            indicator_lower = indicator.lower()
            if 'upper' in indicator_lower:
                upper_band = indicator
            elif 'middle' in indicator_lower:
                middle_band = indicator
            elif 'lower' in indicator_lower:
                lower_band = indicator
        
        # Plot bands
        if lower_band:
            values = df[lower_band].values
            ax.plot(x_range, values, color='blue', linestyle='--',
                   linewidth=1.2, alpha=0.6, label='BB Lower', zorder=1)
        
        if middle_band:
            values = df[middle_band].values
            ax.plot(x_range, values, color='red', linestyle='-',
                   linewidth=1.5, alpha=0.7, label='BB Middle', zorder=1)
        
        if upper_band:
            values = df[upper_band].values
            ax.plot(x_range, values, color='blue', linestyle='--',
                   linewidth=1.2, alpha=0.6, label='BB Upper', zorder=1)
        
        # Fill between upper and lower bands
        if upper_band and lower_band:
            upper_values = df[upper_band].values
            lower_values = df[lower_band].values
            ax.fill_between(x_range, lower_values, upper_values, 
                          color='blue', alpha=0.1, label='BB Channel', zorder=0)
    
    def _plot_ma_group(self, ax, df: pd.DataFrame, group_name: str, indicators: List[str]):
        """Plot Moving Average group on main chart"""
        x_range = range(len(df))
        
        for indicator in indicators:
            indicator_lower = indicator.lower()
            values = df[indicator].values
            
            # Determine color based on MA type
            if 'sma' in indicator_lower:
                color = 'blue'
                linewidth = 2.0
            elif 'ema' in indicator_lower:
                color = 'red'
                linewidth = 2.0
            elif 'wma' in indicator_lower:
                color = 'green'
                linewidth = 1.8
            elif 'hma' in indicator_lower:
                color = 'orange'
                linewidth = 1.8
            elif 'ma_' in indicator_lower:
                color = 'purple'
                linewidth = 2.0
            elif 'dema' in indicator_lower:
                color = 'cyan'
                linewidth = 1.8
            elif 'tema' in indicator_lower:
                color = 'magenta'
                linewidth = 1.8
            else:
                color = self._get_indicator_color(indicator)
                linewidth = 1.5
            
            ax.plot(x_range, values, color=color, linewidth=linewidth,
                   label=indicator, alpha=0.8, zorder=1)
    
    def _plot_sar_group(self, ax, df: pd.DataFrame, group_name: str, indicators: List[str]):
        """Plot Parabolic SAR group on main chart"""
        x_range = range(len(df))
        
        for indicator in indicators:
            values = df[indicator].values
            ax.scatter(x_range, values, color='red', s=10, alpha=0.7, 
                      label=indicator, zorder=3)
    
    def _plot_volume_group(self, ax, df: pd.DataFrame, group_name: str, indicators: List[str]):
        """Plot volume indicator group"""
        x_range = range(len(df))
        
        for indicator in indicators:
            indicator_lower = indicator.lower()
            values = df[indicator].values
            
            if 'volume' in indicator_lower:
                # Color bars by price direction
                colors = ['#26A69A' if df['close'].iloc[i] >= df['open'].iloc[i] 
                         else '#EF5350' for i in range(len(df))]
                
                ax.bar(x_range, values, color=colors, alpha=0.7, width=0.8, 
                      label=indicator, edgecolor='none')
                
                # Add volume moving average if available
                vol_ma_indicators = [ind for ind in df.columns if 
                                    ('volume_sma' in ind.lower() or 
                                     'volume_ema' in ind.lower()) and 
                                    ind != indicator]
                for vol_ma in vol_ma_indicators[:2]:  # Max 2 volume MAs
                    ax.plot(x_range, df[vol_ma].values, color='blue', 
                           linewidth=1.5, alpha=0.8, label=vol_ma)
            else:
                # Other volume indicators (OBV, AD, ADOSC)
                color = self._get_indicator_color(indicator)
                ax.plot(x_range, values, color=color, linewidth=2, label=indicator)
        
        ax.set_ylabel('Volume', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_separate_group(self, ax, df: pd.DataFrame, group_name: str, indicators: List[str]):
        """Plot separate indicator group in its own subplot"""
        x_range = range(len(df))
        group_lower = group_name.lower()
        
        # MACD Family
        if any('macd' in ind.lower() for ind in indicators):
            self._plot_macd_group(ax, df, group_name, indicators)
        
        # Stochastic Family
        elif any('stoch' in group_lower or 'slowk' in ind.lower() or 'fastk' in ind.lower() 
                for ind in indicators):
            self._plot_stochastic_group(ax, df, group_name, indicators)
        
        # RSI
        elif any('rsi' in ind.lower() for ind in indicators):
            self._plot_rsi_group(ax, df, group_name, indicators)
        
        # Aroon
        elif any('aroon' in group_lower for ind in indicators):
            self._plot_aroon_group(ax, df, group_name, indicators)
        
        # ADX/DX
        elif any('adx' in group_lower or 'dx' in group_lower for ind in indicators):
            self._plot_adx_group(ax, df, group_name, indicators)
        
        # Default plotting for other groups
        else:
            self._plot_default_group(ax, df, group_name, indicators)
        
        ax.set_ylabel(self._get_subplot_label(group_name, indicators), 
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_macd_group(self, ax, df: pd.DataFrame, group_name: str, indicators: List[str]):
        """Plot MACD indicator group (macd, macdsignal, macdhist together)"""
        x_range = range(len(df))
        
        for indicator in indicators:
            indicator_lower = indicator.lower()
            values = df[indicator].values
            
            if 'hist' in indicator_lower:
                # Histogram
                colors = ['#26A69A' if val >= 0 else '#EF5350' for val in values]
                ax.bar(x_range, values, color=colors, alpha=0.7, width=0.8, 
                      label=indicator, edgecolor='none')
            elif 'signal' in indicator_lower:
                # Signal line
                ax.plot(x_range, values, color='red', linewidth=1.5, 
                       label=indicator, alpha=0.8)
            else:
                # MACD line
                ax.plot(x_range, values, color='blue', linewidth=1.5, 
                       label=indicator, alpha=0.8)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    def _plot_stochastic_group(self, ax, df: pd.DataFrame, group_name: str, indicators: List[str]):
        """Plot Stochastic indicator group (K and D lines together)"""
        x_range = range(len(df))
        
        # Sort: K line first, D line second
        k_line = None
        d_line = None
        
        for indicator in indicators:
            indicator_lower = indicator.lower()
            if 'k' in indicator_lower:
                k_line = indicator
            elif 'd' in indicator_lower:
                d_line = indicator
        
        if k_line:
            ax.plot(x_range, df[k_line].values, color='blue', 
                   linewidth=1.5, label=k_line, alpha=0.8)
        
        if d_line:
            ax.plot(x_range, df[d_line].values, color='red', 
                   linewidth=1.5, label=d_line, alpha=0.8)
        
        # Add overbought/oversold lines
        ax.axhline(y=80, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(y=20, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_ylim(0, 100)
    
    def _plot_rsi_group(self, ax, df: pd.DataFrame, group_name: str, indicators: List[str]):
        """Plot RSI indicator"""
        x_range = range(len(df))
        
        for indicator in indicators:
            ax.plot(x_range, df[indicator].values, color='purple', 
                   linewidth=2, label=indicator, alpha=0.8)
        
        # Add overbought/oversold lines
        ax.axhline(y=70, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(y=30, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(y=50, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_ylim(0, 100)
    
    def _plot_aroon_group(self, ax, df: pd.DataFrame, group_name: str, indicators: List[str]):
        """Plot Aroon indicator group (up and down together)"""
        x_range = range(len(df))
        
        for indicator in indicators:
            indicator_lower = indicator.lower()
            if 'up' in indicator_lower:
                color = 'green'
            elif 'down' in indicator_lower:
                color = 'red'
            else:
                color = self._get_indicator_color(indicator)
            
            ax.plot(x_range, df[indicator].values, color=color, 
                   linewidth=1.5, label=indicator, alpha=0.8)
        
        ax.set_ylim(0, 100)
    
    def _plot_adx_group(self, ax, df: pd.DataFrame, group_name: str, indicators: List[str]):
        """Plot ADX indicator group"""
        x_range = range(len(df))
        
        for indicator in indicators:
            color = self._get_indicator_color(indicator)
            ax.plot(x_range, df[indicator].values, color=color, 
                   linewidth=1.5, label=indicator, alpha=0.8)
    
    def _plot_default_group(self, ax, df: pd.DataFrame, group_name: str, indicators: List[str]):
        """Default plotting for indicator groups"""
        x_range = range(len(df))
        
        for indicator in indicators:
            color = self._get_indicator_color(indicator)
            ax.plot(x_range, df[indicator].values, color=color, 
                   linewidth=1.5, label=indicator, alpha=0.8)
    
    def _configure_x_axis(self, ax, df: pd.DataFrame):
        """Configure x-axis with dates"""
        n_points = len(df)
        step = max(1, n_points // 10)
        
        # Set ticks
        ax.set_xticks(range(0, n_points, step))
        
        # Format labels
        if hasattr(df.index, '__len__') and len(df.index) > 0:
            labels = [str(idx) for idx in df.index[::step]]
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_xlim(-1, n_points)
    
    def _get_indicator_color(self, indicator_name: str) -> str:
        """Get color for indicator based on its name"""
        indicator_lower = indicator_name.lower()
        
        # Predefined colors
        if 'sma' in indicator_lower:
            return 'blue'
        elif 'ema' in indicator_lower:
            return 'red'
        elif 'wma' in indicator_lower:
            return 'green'
        elif 'hma' in indicator_lower:
            return 'orange'
        elif 'rsi' in indicator_lower:
            return 'purple'
        elif 'macd' in indicator_lower:
            return 'blue'
        elif 'signal' in indicator_lower:
            return 'red'
        elif 'hist' in indicator_lower:
            return 'green'
        elif 'cci' in indicator_lower:
            return 'brown'
        elif 'stoch' in indicator_lower or 'k' in indicator_lower:
            return 'blue'
        elif 'd' in indicator_lower:
            return 'red'
        elif 'upper' in indicator_lower or 'lower' in indicator_lower:
            return 'blue'
        elif 'middle' in indicator_lower:
            return 'red'
        
        # Default: generate color from hash
        import hashlib
        h = hashlib.md5(indicator_lower.encode()).hexdigest()
        return f"#{h[:6]}"
    
    def _get_subplot_label(self, group_name: str, indicators: List[str]) -> str:
        """Get label for subplot based on group name and indicators"""
        group_lower = group_name.lower()
        
        if 'macd' in group_lower:
            return 'MACD'
        elif 'stoch' in group_lower:
            return 'STOCH'
        elif 'rsi' in group_lower:
            return 'RSI'
        elif 'aroon' in group_lower:
            return 'AROON'
        elif 'adx' in group_lower:
            return 'ADX'
        elif 'dx' in group_lower:
            return 'DX'
        elif 'cci' in group_lower:
            return 'CCI'
        elif 'mfi' in group_lower:
            return 'MFI'
        elif 'willr' in group_lower:
            return 'WILLR'
        elif 'roc' in group_lower:
            return 'ROC'
        elif 'cmo' in group_lower:
            return 'CMO'
        elif 'bop' in group_lower:
            return 'BOP'
        elif 'apo' in group_lower:
            return 'APO'
        elif 'ppo' in group_lower:
            return 'PPO'
        elif 'trix' in group_lower:
            return 'TRIX'
        elif 'ultosc' in group_lower:
            return 'ULTOSC'
        elif 'volume' in group_lower:
            return 'Volume'
        else:
            # Use the first indicator's name
            if indicators:
                return indicators[0].split('_')[0].upper()
            return group_name.upper()


def plot_indicators(df: pd.DataFrame, title: str = "Price and Indicators",
                   overlay_indicators: Optional[List[str]] = None,
                   separate_indicators: Optional[List[str]] = None,
                   save_path: Optional[str] = None, figsize: tuple = (16, 10)) -> plt.Figure:
    """Convenience function to plot indicators
    
    Args:
        df: DataFrame with OHLCV + indicator columns (lowercase)
        title: Chart title
        overlay_indicators: List of indicators to plot on main chart
        separate_indicators: List of indicators to plot in separate subplots
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    plotter = IndicatorPlotter(figsize=figsize)
    return plotter.plot(df, title=title, 
                       overlay_indicators=overlay_indicators,
                       separate_indicators=separate_indicators,
                       save_path=save_path)