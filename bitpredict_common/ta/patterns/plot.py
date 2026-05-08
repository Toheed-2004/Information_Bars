import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from bitpredict.common.logging import get_logger
from bitpredict.common.constants import PATTERN_TALIB_COL_PREFIX


# Logger
logger = get_logger(__name__)


def plot_patterns(
    data,
    save_image=False,
    save_path=None,
    figsize=(18, 9),
    title="Candlestick Patterns"
):
    """
    Plot candlestick patterns with dynamic marker sizing based on signal strength.
    
    Args:
        data: Result from calculate_patterns (DataFrame or numpy dict)
        save_image: If True, save plot to specified path
        save_path: Path to save image (required if save_image=True)
        figsize: Figure size
        title: Plot title
    """
    # Convert numpy dict to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        logger.error("Data must be DataFrame or numpy dict")
        raise ValueError("Data must be DataFrame or numpy dict")
    
    logger.info(f"Plotting patterns for {len(df)} candlesticks")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot candlesticks
    for i in range(len(df)):
        open_price = df['open'].iloc[i]
        close_price = df['close'].iloc[i]
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]
        
        color = 'green' if close_price > open_price else 'red'
        
        # Plot wick
        ax.plot([i, i], [low_price, high_price], color=color, linewidth=1)
        
        # Plot body
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        body_height = body_top - body_bottom
        
        if body_height == 0:
            body_height = 0.001 * close_price
        
        rect = plt.Rectangle(
            (i - 0.3, body_bottom), 0.6, body_height,
            facecolor=color, edgecolor=color
        )
        ax.add_patch(rect)
    
    # Find pattern columns starting with the defined prefix
    pattern_cols = [col for col in df.columns if col.startswith(PATTERN_TALIB_COL_PREFIX)]
    logger.info(f"Found {len(pattern_cols)} pattern columns")
    
    # Get all unique values from pattern columns to determine size mapping
    all_values = set()
    for pattern in pattern_cols:
        unique_vals = df[pattern].dropna().unique()
        all_values.update(unique_vals[unique_vals != 0])
    
    # Create dynamic size mapping based on absolute values
    abs_values = sorted(set(abs(v) for v in all_values if v != 0))
    size_mapping = {}
    base_size = 50
    size_increment = 30
    
    for idx, abs_val in enumerate(abs_values):
        size_mapping[abs_val] = base_size + (idx * size_increment)
    
    logger.debug(f"Size mapping: {size_mapping}")
    
    # Plot patterns with dynamic sizing
    pattern_count = 0
    for pattern in pattern_cols:
        if pattern in df.columns:
            signals = df[df[pattern] != 0]
            
            if len(signals) > 0:
                pattern_count += len(signals)
                
                # Bullish signals (positive values)
                bullish = signals[signals[pattern] > 0]
                if len(bullish) > 0:
                    sizes = [size_mapping[abs(val)] for val in bullish[pattern]]
                    ax.scatter(
                        bullish.index, bullish['close'],
                        color='lime', marker='o', s=sizes,
                        label=f"{pattern} (Bullish)", zorder=5, alpha=0.7
                    )
                
                # Bearish signals (negative values)
                bearish = signals[signals[pattern] < 0]
                if len(bearish) > 0:
                    sizes = [size_mapping[abs(val)] for val in bearish[pattern]]
                    ax.scatter(
                        bearish.index, bearish['close'],
                        color='red', marker='o', s=sizes,
                        label=f"{pattern} (Bearish)", zorder=5, alpha=0.7
                    )
    
    logger.info(f"Plotted {pattern_count} pattern signals")
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis with dates if available
    if 'datetime' in df.columns:
        dates = pd.to_datetime(df['datetime'])
        tick_positions = range(0, len(dates), 50)
        tick_labels = [dates.iloc[i].strftime('%Y-%m-%d') for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)
    
    plt.tight_layout()
    
    # Handle saving
    if save_image:
        if not save_path:
            logger.error("save_path is required when save_image=True")
            plt.close()
            return
        
        # Validate and save to provided path
        filepath = Path(save_path)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.png')
        
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {filepath}")
        except Exception as e:
            logger.error(f"Invalid path or unable to save: {e}")
        finally:
            plt.close()
    else:
        plt.show()