import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Import your SMC and plot modules
from bitpredict.common.ta.smc import smc, plot


def generate_random_market_data(n_periods=1000, start_price=100, volatility=0.02):
    """
    Generate realistic random market data with some patterns
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate datetimes
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_periods/24)  # Assuming hourly data
    
    datetimes = pd.date_range(start=start_date, end=end_date, periods=n_periods)
    
    # Generate base prices with trend and cycles
    time = np.linspace(0, 10, n_periods)
    
    # Create multiple trends/cycles
    trend = 0.0005 * time  # Upward trend
    cycle1 = 5 * np.sin(time * 0.5)  # Long-term cycle
    cycle2 = 2 * np.sin(time * 2)  # Short-term cycle
    noise = np.random.normal(0, volatility, n_periods).cumsum()  # Random walk
    
    # Combine components
    base_prices = start_price + trend + cycle1 + cycle2 + noise
    
    # Generate OHLC candles
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    for i in range(n_periods):
        if i == 0:
            open_price = base_prices[i]
        else:
            open_price = closes[i-1]
        
        # Generate realistic candle
        daily_range = volatility * base_prices[i]  # Daily range as % of price
        close_price = base_prices[i] + np.random.normal(0, daily_range * 0.3)
        
        # Ensure high > low > 0
        candle_range = abs(close_price - open_price) + daily_range * 0.2
        high_price = max(open_price, close_price) + candle_range * 0.5
        low_price = max(0.01, min(open_price, close_price) - candle_range * 0.5)
        
        # Ensure high is highest, low is lowest
        high_price = max(high_price, open_price, close_price, low_price)
        low_price = min(low_price, open_price, close_price, high_price)
        
        # Generate volume correlated with price movement
        volume = random.randint(1000, 10000) * (1 + abs(close_price - open_price) / open_price)
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(volume)
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': datetimes,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    return df


def test_smc_indicators():
    """Test all SMC indicators with random data"""
    
    print("=" * 60)
    print("SMC INDICATORS COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Step 1: Generate random market data
    print("\n1. Generating random market data...")
    df = generate_random_market_data(n_periods=500, start_price=100)
    
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"   Sample data:")
    print(df.head())
    
    # Step 2: Run ALL SMC indicators
    print("\n2. Running ALL SMC indicators...")
    
    # List all available indicators
    all_indicators = [
        "fvg",
        "swing_highs_lows", 
        "bos_choch",
        "ob",
        "previous_high_low",
        "sessions",
        "breaker_block",
        "mitigation_block",
        "retracements",
        "algorithmic_order_block",
        "bpr",
        "liquidity_swing_hl"
    ]
    
    print(f"   Running {len(all_indicators)} indicators:")
    for i, indicator in enumerate(all_indicators, 1):
        print(f"   {i:2d}. {indicator}")
    
    # Run indicators
    out = smc.run(
        df,
        *all_indicators,
        prefix=True,
        fill_na=True,
        shift=True
    )
    
    print(f"\n   Output shape: {out.shape}")
    print(f"   Output columns: {len(out.columns)}")
    
    # Count non-null values for each indicator
    print("\n   Indicator signals detected:")
    for indicator in all_indicators:
        # Look for columns related to this indicator
        indicator_cols = [col for col in out.columns if indicator in col.lower()]
        if indicator_cols:
            # Count non-NaN values in the main signal column
            main_col = indicator_cols[0]
            if main_col in out.columns:
                signal_count = out[main_col].notna().sum()
                print(f"   • {indicator}: {signal_count} signals")
    
    # Step 3: Save results
    print("\n3. Saving results...")
    out.to_csv("smc_comprehensive_test_output.csv", index=False)
    print("   Saved to: smc_comprehensive_test_output.csv")
    
    # Step 4: Create individual plots for each indicator
    print("\n4. Creating individual plots...")
    
    # Create a directory for plots
    import os
    os.makedirs("smc_test_plots", exist_ok=True)
    
    # Test each indicator individually for clarity
    print("   Testing individual indicators:")
    
    for i, indicator in enumerate(all_indicators, 1):
        try:
            # Run single indicator
            single_out = smc.run(
                df,
                indicator,
                prefix=False,
                fill_na=True,
                shift=True
            )
            
            # Create plot
            fig = plot(
                single_out,
                return_type="png",
                save_path=f"smc_test_plots/{indicator}_test.png",
                width=1600,
                height=900
            )
            
            if fig:  # If saved successfully, fig will be True
                print(f"   {i:2d}. ✓ {indicator} - plot saved")
            else:
                print(f"   {i:2d}. ✗ {indicator} - failed to save plot")
                
        except Exception as e:
            print(f"   {i:2d}. ✗ {indicator} - error: {str(e)}")
    
    # Step 5: Create comprehensive plot
    print("\n5. Creating comprehensive plot with ALL indicators...")
    try:
        fig = plot(
            out,
            return_type="png",
            save_path="smc_test_plots/ALL_INDICATORS_comprehensive.png",
            width=1920,
            height=1080
        )
        
        if fig:
            print("Comprehensive plot saved")
        else:
            print("Failed to save comprehensive plot")
            
    except Exception as e:
        print(f"Error creating comprehensive plot: {str(e)}")
    
    # Step 6: Test plot function with different return types
    print("\n6. Testing plot function with different configurations...")
    
    # Test 1: PNG without saving
    print("   Test 1: PNG without saving (returns figure)...")
    try:
        fig = plot(out, return_type="png")
        print(f"Success - Figure type: {type(fig)}")
        
        # You could add more tests here, like checking figure properties
        # print(f"   Figure layout: {fig.layout}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test 2: HTML output
    print("\n   Test 2: HTML output...")
    try:
        result = plot(
            out,
            return_type="html",
            save_path="smc_test_plots/interactive_chart.html"
        )
        if result:
            print("HTML chart saved successfully")
        else:
            print("Failed to save HTML chart")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Step 7: Verify data integrity
    print("\n7. Verifying data integrity...")
    
    # Check for basic issues
    issues = []
    
    # Check 1: No NaN values in essential columns
    essential_cols = ['open', 'high', 'low', 'close']
    for col in essential_cols:
        if col in out.columns and out[col].isna().any():
            issues.append(f"NaN values found in {col}")
    
    # Check 2: High >= Low for all candles
    if 'high' in out.columns and 'low' in out.columns:
        invalid_candles = (out['high'] < out['low']).sum()
        if invalid_candles > 0:
            issues.append(f"{invalid_candles} candles have high < low")
    
    # Check 3: Data ordering
    if 'datetime' in out.columns:
        if not pd.to_datetime(out['datetime']).is_monotonic_increasing:
            issues.append("datetimes are not in increasing order")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print("All checks passed!")
    
    # Step 8: Sample output
    print("\n8. Sample output data:")
    print("   First 5 rows with indicator columns:")
    
    # Get indicator columns (excluding OHLCV)
    ohlcv_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    indicator_cols = [col for col in out.columns if col not in ohlcv_cols]
    
    # Show sample of indicator data
    sample_df = out[ohlcv_cols[:5] + indicator_cols[:5]].head()  # Show first 5 of each
    print(sample_df.to_string())
    
    # Step 9: Summary statistics
    print("\n9. Summary statistics:")
    print(f"   Total data points: {len(out)}")
    print(f"   Total columns: {len(out.columns)}")
    print(f"   OHLCV columns: {len([c for c in out.columns if c in ohlcv_cols])}")
    print(f"   Indicator columns: {len(indicator_cols)}")
    
    # Count signals per indicator type
    print("\n   Signal summary:")
    for indicator in all_indicators:
        indicator_signal_cols = [col for col in indicator_cols if indicator in col.lower()]
        if indicator_signal_cols:
            # Use the first column that looks like a signal
            for col in indicator_signal_cols:
                if col.endswith('_type') or col in ['fvg', 'bos', 'ob', 'aob_type', 'bb_type', 'mb_type', 'lshl_type']:
                    signal_count = out[col].notna().sum()
                    print(f"   • {indicator}: {signal_count} signals")
                    break
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)
    
    # Return the output DataFrame for further analysis if needed
    return out


def quick_test():
    """Quick test for development/debugging"""
    print("Running quick test...")
    
    # Generate small dataset
    df = generate_random_market_data(n_periods=100, start_price=50)
    
    # Test a few key indicators
    test_indicators = ["fvg", "swing_highs_lows", "bos_choch", "ob"]
    
    print(f"\nTesting indicators: {test_indicators}")
    
    out = smc.run(
        df,
        *test_indicators,
        prefix=False,
        fill_na=True,
        shift=False
    )
    
    # Quick plot
    fig = plot(out, return_type="png")
    
    # Show some stats
    print(f"\nData shape: {out.shape}")
    print("\nSample of signals:")
    signal_cols = [col for col in out.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'datetime']]
    for col in signal_cols[:5]:  # Show first 5 signal columns
        signal_count = out[col].notna().sum()
        print(f"  {col}: {signal_count} signals")
    
    return fig, out


if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Comprehensive test (all indicators)")
    print("2. Quick test (few indicators, faster)")
    
    choice = input("\nEnter choice (1, 2): ").strip()
    
    if choice == "1":
        print("\nRunning comprehensive test...")
        results = test_smc_indicators()
        
        # Ask if user wants to see plots
        show_plots = input("\nDo you want to display the comprehensive plot? (y/n): ").lower().strip()
        if show_plots == 'y':
            # Load and show the saved comprehensive plot
            try:
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                
                img = mpimg.imread('smc_test_plots/ALL_INDICATORS_comprehensive.png')
                plt.figure(figsize=(16, 9))
                plt.imshow(img)
                plt.axis('off')
                plt.title('SMC Comprehensive Test Results')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Could not display image: {e}")
                print("You can view it at: smc_test_plots/ALL_INDICATORS_comprehensive.png")
    
    elif choice == "2":
        print("\nRunning quick test...")
        fig, out = quick_test()
        
        # Show the plot
        show = input("\nDo you want to show the plot? (y/n): ").lower().strip()
        if show == 'y':
            fig.show()
    
