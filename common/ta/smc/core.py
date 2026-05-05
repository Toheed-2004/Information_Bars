import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from datetime import datetime


class smc:
    @classmethod
    def fvg(cls, ohlc: DataFrame, join_consecutive=False, shift=True) -> Series:
        """
        FVG - Fair Value Gap
        A fair value gap occurs when the previous high is lower than the next low
        if the current candle is bullish, or when the previous low is higher than
        the next high if the current candle is bearish.

        Parameters:
        -----------
        ohlc : DataFrame
            DataFrame containing OHLC data with columns: ['open', 'high', 'low', 'close']
        join_consecutive : bool, optional (default=False)
            If True, merges consecutive FVGs into one using highest top and lowest bottom
        shift : bool, optional (default=True)
            If True, shifts FVG signals by 2 candles to remove look-ahead bias

        Returns:
        --------
        Series
            DataFrame with columns:
            - fvg: 1 for bullish, -1 for bearish
            - top: top of the FVG
            - bottom: bottom of the FVG
            - mitigated_index: index where FVG was mitigated (0 if not mitigated)
        """
        # Shift calculation to handle look-ahead
        fvg = np.where(
            (
                (ohlc["high"].shift(1) < ohlc["low"].shift(-1))
                & (ohlc["close"] > ohlc["open"])
            )
            | (
                (ohlc["low"].shift(1) > ohlc["high"].shift(-1))
                & (ohlc["close"] < ohlc["open"])
            ),
            np.where(ohlc["close"] > ohlc["open"], 1, -1),
            np.nan,
        )

        top = np.where(
            ~np.isnan(fvg),
            np.where(
                ohlc["close"] > ohlc["open"],
                ohlc["low"].shift(-1),
                ohlc["low"].shift(1),
            ),
            np.nan,
        )

        bottom = np.where(
            ~np.isnan(fvg),
            np.where(
                ohlc["close"] > ohlc["open"],
                ohlc["high"].shift(1),
                ohlc["high"].shift(-1),
            ),
            np.nan,
        )

        # Join consecutive FVGs
        if join_consecutive:
            for i in range(len(fvg) - 1):
                if fvg[i] == fvg[i + 1]:
                    top[i + 1] = max(top[i], top[i + 1])
                    bottom[i + 1] = min(bottom[i], bottom[i + 1])
                    fvg[i] = top[i] = bottom[i] = np.nan

        # Find mitigation points
        mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(~np.isnan(fvg))[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)
            if fvg[i] == 1:
                mask = ohlc["low"][i + 2 :] <= top[i]
            elif fvg[i] == -1:
                mask = ohlc["high"][i + 2 :] >= bottom[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                mitigated_index[i] = j

        mitigated_index = np.where(np.isnan(fvg), np.nan, mitigated_index)

        result = pd.concat(
            [
                pd.Series(fvg, name="fvg"),
                pd.Series(top, name="fvg_top"),
                pd.Series(bottom, name="fvg_bottom"),
                pd.Series(mitigated_index, name="fvg_mitigated_index"),
            ],
            axis=1,
        )

        # Shift to remove look-ahead bias
        if shift:
            result = result.shift(2).reset_index(drop=True)

        return result

    @classmethod
    def swing_highs_lows(
        cls, ohlc: DataFrame, swing_length: int = 5, shift=True
    ) -> Series:
        """
        Swing Highs and Lows
        Identifies swing highs (local maxima) and swing lows (local minima).

        Parameters:
        -----------
        ohlc : DataFrame
            DataFrame containing OHLC data
        swing_length : int, optional (default=10)
            Number of candles to look back and forward (total window = swing_length * 2)
        shift : bool, optional (default=True)
            If True, shifts swing signals by swing_length//2 candles to remove look-ahead bias

        Returns:
        --------
        Series
            DataFrame with columns:
            - high_low: 1 for swing high, -1 for swing low
            - level: price level of the swing point
        """
        original_swing_length = swing_length
        swing_length *= 2

        # Find local maxima and minima
        swing_highs_lows = np.where(
            ohlc["high"]
            == ohlc["high"].shift(-(swing_length // 2)).rolling(swing_length).max(),
            1,
            np.where(
                ohlc["low"]
                == ohlc["low"].shift(-(swing_length // 2)).rolling(swing_length).min(),
                -1,
                np.nan,
            ),
        )

        # Remove consecutive swings in same direction
        while True:
            positions = np.where(~np.isnan(swing_highs_lows))[0]
            if len(positions) < 2:
                break

            current = swing_highs_lows[positions[:-1]]
            next_vals = swing_highs_lows[positions[1:]]
            highs = ohlc["high"].iloc[positions[:-1]].values
            lows = ohlc["low"].iloc[positions[:-1]].values
            next_highs = ohlc["high"].iloc[positions[1:]].values
            next_lows = ohlc["low"].iloc[positions[1:]].values

            index_to_remove = np.zeros(len(positions), dtype=bool)

            # For consecutive highs, keep the higher one
            consecutive_highs = (current == 1) & (next_vals == 1)
            index_to_remove[:-1] |= consecutive_highs & (highs < next_highs)
            index_to_remove[1:] |= consecutive_highs & (highs >= next_highs)

            # For consecutive lows, keep the lower one
            consecutive_lows = (current == -1) & (next_vals == -1)
            index_to_remove[:-1] |= consecutive_lows & (lows > next_lows)
            index_to_remove[1:] |= consecutive_lows & (lows <= next_lows)

            if not index_to_remove.any():
                break
            swing_highs_lows[positions[index_to_remove]] = np.nan

        # Add boundary swings
        positions = np.where(~np.isnan(swing_highs_lows))[0]
        if len(positions) > 0:
            if swing_highs_lows[positions[0]] == 1:
                swing_highs_lows[0] = -1
            if swing_highs_lows[positions[0]] == -1:
                swing_highs_lows[0] = 1
            if swing_highs_lows[positions[-1]] == -1:
                swing_highs_lows[-1] = 1
            if swing_highs_lows[positions[-1]] == 1:
                swing_highs_lows[-1] = -1

        level = np.where(
            ~np.isnan(swing_highs_lows),
            np.where(swing_highs_lows == 1, ohlc["high"], ohlc["low"]),
            np.nan,
        )

        result = pd.concat(
            [
                pd.Series(swing_highs_lows, name="swing_high_low"),
                pd.Series(level, name="swing_high_low_level"),
            ],
            axis=1,
        )

        # Shift to remove look-ahead bias
        if shift:
            shift_amount = original_swing_length
            result = result.shift(shift_amount).reset_index(drop=True)

        return result

    @classmethod
    def bos_choch(
        cls,
        ohlc: DataFrame,
        swing_length: int = 10,
        close_break: bool = True,
        shift: bool = False,
    ) -> Series:
        """
        Break of Structure (BOS) and Change of Character (CHoCH)
        Identifies market structure changes based on swing points.

        Parameters:
        -----------
        ohlc : DataFrame
            DataFrame containing OHLC data
        swing_length : int, optional (default=10)
            Swing length for internal swing calculation
        close_break : bool, optional (default=True)
            If True, uses close prices for break detection; otherwise uses high/low
        shift : bool, optional (default=True)
            If True, shifts BOS/CHoCH signals to remove look-ahead bias

        Returns:
        --------
        Series
            DataFrame with columns:
            - bos: 1 for bullish BOS, -1 for bearish BOS
            - choch: 1 for bullish CHoCH, -1 for bearish CHoCH
            - level: price level of the structure break
            - broken: index where level was broken
        """
        # Calculate swing highs/lows internally
        swing_highs_lows = cls.swing_highs_lows(ohlc, swing_length, shift=False)

        level_order = []
        highs_lows_order = []
        bos = np.zeros(len(ohlc), dtype=np.int32)
        choch = np.zeros(len(ohlc), dtype=np.int32)
        level = np.zeros(len(ohlc), dtype=np.float32)
        last_positions = []

        # Iterate through swing points
        for i in range(len(swing_highs_lows["swing_high_low"])):
            if not np.isnan(swing_highs_lows["swing_high_low"][i]):
                level_order.append(swing_highs_lows["swing_high_low_level"][i])
                highs_lows_order.append(swing_highs_lows["swing_high_low"][i])

                # Need at least 4 swings to identify BOS/CHoCH
                if len(level_order) >= 4:
                    # Bullish BOS: HLHL pattern with rising lows and highs
                    if np.all(highs_lows_order[-4:] == [-1, 1, -1, 1]) and np.all(
                        level_order[-4]
                        < level_order[-2]
                        < level_order[-3]
                        < level_order[-1]
                    ):
                        bos[last_positions[-2]] = 1
                        level[last_positions[-2]] = level_order[-3]

                    # Bearish BOS: LHLH pattern with falling highs and lows
                    elif np.all(highs_lows_order[-4:] == [1, -1, 1, -1]) and np.all(
                        level_order[-4]
                        > level_order[-2]
                        > level_order[-3]
                        > level_order[-1]
                    ):
                        bos[last_positions[-2]] = -1
                        level[last_positions[-2]] = level_order[-3]

                    # Bullish CHoCH: HLHL pattern with different structure
                    elif np.all(highs_lows_order[-4:] == [-1, 1, -1, 1]) and np.all(
                        level_order[-1]
                        > level_order[-3]
                        > level_order[-4]
                        > level_order[-2]
                    ):
                        choch[last_positions[-2]] = 1
                        level[last_positions[-2]] = level_order[-3]

                    # Bearish CHoCH: LHLH pattern with different structure
                    elif np.all(highs_lows_order[-4:] == [1, -1, 1, -1]) and np.all(
                        level_order[-1]
                        < level_order[-3]
                        < level_order[-4]
                        < level_order[-2]
                    ):
                        choch[last_positions[-2]] = -1
                        level[last_positions[-2]] = level_order[-3]

                last_positions.append(i)

        # Find where levels were broken
        broken = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(np.logical_or(bos != 0, choch != 0))[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)
            if bos[i] == 1 or choch[i] == 1:
                mask = ohlc["close" if close_break else "high"][i + 2 :] > level[i]
            elif bos[i] == -1 or choch[i] == -1:
                mask = ohlc["close" if close_break else "low"][i + 2 :] < level[i]

            if np.any(mask):
                j = np.argmax(mask) + i + 2
                broken[i] = j

                # Remove overlapping signals
                for k in np.where(np.logical_or(bos != 0, choch != 0))[0]:
                    if k < i and broken[k] >= j:
                        bos[k] = 0
                        choch[k] = 0
                        level[k] = 0

        # Clean up unbroken signals
        for i in np.where(
            np.logical_and(np.logical_or(bos != 0, choch != 0), broken == 0)
        )[0]:
            bos[i] = 0
            choch[i] = 0
            level[i] = 0

        # Convert to NaN for missing values
        bos = np.where(bos != 0, bos, np.nan)
        choch = np.where(choch != 0, choch, np.nan)
        level = np.where(level != 0, level, np.nan)
        broken = np.where(broken != 0, broken, np.nan)

        result = pd.concat(
            [
                pd.Series(bos, name="bos"),
                pd.Series(choch, name="bos_choch"),
                pd.Series(level, name="bos_level"),
                pd.Series(broken, name="bos_broken"),
            ],
            axis=1,
        )

        # Shift to remove look-ahead bias
        # swing high/low has shift of swing_length, break detection adds 1 more

        if shift:
            shift_amount = 1  # shift by 1 candle to avoid look-ahead
            result = result.shift(shift_amount).reset_index(drop=True)

        return result

    @classmethod
    def ob(
        cls,
        ohlc: DataFrame,
        swing_length: int = 10,
        close_mitigation: bool = False,
        shift: bool = False,
    ) -> Series:
        """
        Order Blocks (OB)
        Identifies order blocks where significant market orders exist.

        Parameters:
        -----------
        ohlc : DataFrame
            DataFrame containing OHLC data with volume
        swing_length : int, optional (default=10)
            Swing length for internal swing calculation
        close_mitigation : bool, optional (default=False)
            If True, uses close prices for mitigation; otherwise uses high/low
        shift : bool, optional (default=True)
            If True, shifts OB signals to remove look-ahead bias

        Returns:
        --------
        Series
            DataFrame with columns:
            - ob: 1 for bullish, -1 for bearish
            - top: top of order block
            - bottom: bottom of order block
            - ob_volume: cumulative volume (current + 2 previous)
            - mitigated_index: index where OB was mitigated
            - percentage: strength percentage of OB
        """
        # Calculate swing highs/lows internally
        swing_highs_lows = cls.swing_highs_lows(ohlc, swing_length, shift=False)

        ohlc_len = len(ohlc)
        _open = ohlc["open"].values
        _high = ohlc["high"].values
        _low = ohlc["low"].values
        _close = ohlc["close"].values
        _volume = ohlc["volume"].values
        swing_hl = swing_highs_lows["swing_high_low"].values

        # Initialize arrays
        crossed = np.full(ohlc_len, False, dtype=bool)
        ob = np.zeros(ohlc_len, dtype=np.int32)
        top_arr = np.zeros(ohlc_len, dtype=np.float32)
        bottom_arr = np.zeros(ohlc_len, dtype=np.float32)
        obVolume = np.zeros(ohlc_len, dtype=np.float32)
        lowVolume = np.zeros(ohlc_len, dtype=np.float32)
        highVolume = np.zeros(ohlc_len, dtype=np.float32)
        percentage = np.zeros(ohlc_len, dtype=np.float32)
        mitigated_index = np.zeros(ohlc_len, dtype=np.int32)
        breaker = np.full(ohlc_len, False, dtype=bool)

        # Get swing indices
        swing_high_indices = np.flatnonzero(swing_hl == 1)
        swing_low_indices = np.flatnonzero(swing_hl == -1)

        # Process bullish order blocks
        active_bullish = []
        for i in range(ohlc_len):
            close_index = i

            # Check existing bullish OBs for mitigation
            for idx in active_bullish:
                if breaker[idx]:
                    if _high[close_index] > top_arr[idx]:
                        # Remove mitigated OB
                        ob[idx] = 0
                        top_arr[idx] = 0.0
                        bottom_arr[idx] = 0.0
                        obVolume[idx] = 0.0
                        lowVolume[idx] = 0.0
                        highVolume[idx] = 0.0
                        mitigated_index[idx] = 0
                        percentage[idx] = 0.0
                        active_bullish.remove(idx)
                else:
                    # Check if OB is mitigated
                    if (
                        not close_mitigation and _low[close_index] < bottom_arr[idx]
                    ) or (
                        close_mitigation
                        and min(_open[close_index], _close[close_index])
                        < bottom_arr[idx]
                    ):
                        breaker[idx] = True
                        mitigated_index[idx] = close_index - 1

            # Find last swing high
            pos = np.searchsorted(swing_high_indices, close_index)
            last_top_index = swing_high_indices[pos - 1] if pos > 0 else None

            # Create new bullish OB if conditions met
            if last_top_index is not None:
                if (
                    _close[close_index] > _high[last_top_index]
                    and not crossed[last_top_index]
                ):
                    crossed[last_top_index] = True
                    default_index = close_index - 1
                    obBtm = _high[default_index]
                    obTop = _low[default_index]
                    obIndex = default_index

                    # Find lowest low between swing high and close
                    if close_index - last_top_index > 1:
                        start = last_top_index + 1
                        end = close_index
                        if end > start:
                            segment = _low[start:end]
                            min_val = segment.min()
                            candidates = np.nonzero(segment == min_val)[0]
                            if candidates.size:
                                candidate_index = start + candidates[-1]
                                obBtm = _low[candidate_index]
                                obTop = _high[candidate_index]
                                obIndex = candidate_index

                    # Set OB values
                    ob[obIndex] = 1
                    top_arr[obIndex] = obTop
                    bottom_arr[obIndex] = obBtm
                    vol_cur = _volume[close_index]
                    vol_prev1 = _volume[close_index - 1] if close_index >= 1 else 0.0
                    vol_prev2 = _volume[close_index - 2] if close_index >= 2 else 0.0
                    obVolume[obIndex] = vol_cur + vol_prev1 + vol_prev2
                    lowVolume[obIndex] = vol_prev2
                    highVolume[obIndex] = vol_cur + vol_prev1
                    max_vol = max(highVolume[obIndex], lowVolume[obIndex])
                    percentage[obIndex] = (
                        (min(highVolume[obIndex], lowVolume[obIndex]) / max_vol * 100.0)
                        if max_vol != 0
                        else 100.0
                    )
                    active_bullish.append(obIndex)

        # Process bearish order blocks
        active_bearish = []
        for i in range(ohlc_len):
            close_index = i

            # Check existing bearish OBs for mitigation
            for idx in active_bearish.copy():
                if breaker[idx]:
                    if _low[close_index] < bottom_arr[idx]:
                        # Remove mitigated OB
                        ob[idx] = 0
                        top_arr[idx] = 0.0
                        bottom_arr[idx] = 0.0
                        obVolume[idx] = 0.0
                        lowVolume[idx] = 0.0
                        highVolume[idx] = 0.0
                        mitigated_index[idx] = 0
                        percentage[idx] = 0.0
                        active_bearish.remove(idx)
                else:
                    # Check if OB is mitigated
                    if (not close_mitigation and _high[close_index] > top_arr[idx]) or (
                        close_mitigation
                        and max(_open[close_index], _close[close_index]) > top_arr[idx]
                    ):
                        breaker[idx] = True
                        mitigated_index[idx] = close_index

            # Find last swing low
            pos = np.searchsorted(swing_low_indices, close_index)
            last_btm_index = swing_low_indices[pos - 1] if pos > 0 else None

            # Create new bearish OB if conditions met
            if last_btm_index is not None:
                if (
                    _close[close_index] < _low[last_btm_index]
                    and not crossed[last_btm_index]
                ):
                    crossed[last_btm_index] = True
                    default_index = close_index - 1
                    obTop = _high[default_index]
                    obBtm = _low[default_index]
                    obIndex = default_index

                    # Find highest high between swing low and close
                    if close_index - last_btm_index > 1:
                        start = last_btm_index + 1
                        end = close_index
                        if end > start:
                            segment = _high[start:end]
                            max_val = segment.max()
                            candidates = np.nonzero(segment == max_val)[0]
                            if candidates.size:
                                candidate_index = start + candidates[-1]
                                obTop = _high[candidate_index]
                                obBtm = _low[candidate_index]
                                obIndex = candidate_index

                    # Set OB values
                    ob[obIndex] = -1
                    top_arr[obIndex] = obTop
                    bottom_arr[obIndex] = obBtm
                    vol_cur = _volume[close_index]
                    vol_prev1 = _volume[close_index - 1] if close_index >= 1 else 0.0
                    vol_prev2 = _volume[close_index - 2] if close_index >= 2 else 0.0
                    obVolume[obIndex] = vol_cur + vol_prev1 + vol_prev2
                    lowVolume[obIndex] = vol_cur + vol_prev1
                    highVolume[obIndex] = vol_prev2
                    max_vol = max(highVolume[obIndex], lowVolume[obIndex])
                    percentage[obIndex] = (
                        (min(highVolume[obIndex], lowVolume[obIndex]) / max_vol * 100.0)
                        if max_vol != 0
                        else 100.0
                    )
                    active_bearish.append(obIndex)

        # Clean up and convert to Series
        ob = np.where(ob != 0, ob, np.nan)
        top_arr = np.where(~np.isnan(ob), top_arr, np.nan)
        bottom_arr = np.where(~np.isnan(ob), bottom_arr, np.nan)
        obVolume = np.where(~np.isnan(ob), obVolume, np.nan)
        mitigated_index = np.where(~np.isnan(ob), mitigated_index, np.nan)
        percentage = np.where(~np.isnan(ob), percentage, np.nan)

        result = pd.concat(
            [
                pd.Series(ob, name="ob"),
                pd.Series(top_arr, name="ob_top"),
                pd.Series(bottom_arr, name="ob_bottom"),
                pd.Series(obVolume, name="ob_volume"),
                pd.Series(mitigated_index, name="ob_mitigated_index"),
                pd.Series(percentage, name="ob_percentage"),
            ],
            axis=1,
        )

        # Shift to remove look-ahead bias
        if shift:
            shift_amount = 1  # shift by 1 candle to avoid look-ahead
            result = result.shift(shift_amount).reset_index(drop=True)

        return result

    @classmethod
    def previous_high_low(
        cls, ohlc: DataFrame, time_frame: str = "1D", shift: bool = False
    ) -> Series:
        """
        Previous High/Low of Higher Timeframe
        Identifies previous session's high and low levels.

        Parameters:
        -----------
        ohlc : DataFrame
            DataFrame containing OHLC data with datetime index
        time_frame : str, optional (default="1D")
            Timeframe for resampling: 15m, 1H, 4H, 1D, 1W, 1M
        shift : bool, optional (default=True)
            If True, shifts previous high/low to current candle to remove look-ahead bias

        Returns:
        --------
        Series
            DataFrame with columns:
            - previous_high: previous session's high
            - previous_low: previous session's low
            - broken_high: 1 if current high > previous_high
            - broken_low: 1 if current low < previous_low
        """
        # Ensure datetime index
        ohlc.index = pd.to_datetime(ohlc.index)

        # Initialize arrays
        previous_high = np.zeros(len(ohlc), dtype=np.float32)
        previous_low = np.zeros(len(ohlc), dtype=np.float32)
        broken_high = np.zeros(len(ohlc), dtype=np.int32)
        broken_low = np.zeros(len(ohlc), dtype=np.int32)

        # Resample to higher timeframe
        ohlcv = (
            ohlc.resample(time_frame)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        # Track break status per session
        currently_broken_high = False
        currently_broken_low = False
        last_broken_time = None

        # Process each candle
        for i in range(len(ohlc)):
            # Find previous session in resampled data
            resampled_previous_index = np.where(ohlcv.index < ohlc.index[i])[0]
            if len(resampled_previous_index) <= 1:
                previous_high[i] = np.nan
                previous_low[i] = np.nan
                continue
            resampled_previous_index = resampled_previous_index[
                -2
            ]  # -2 for previous complete session

            # Reset break status on new session
            if last_broken_time != resampled_previous_index:
                currently_broken_high = False
                currently_broken_low = False
                last_broken_time = resampled_previous_index

            # Get previous session levels
            previous_high[i] = ohlcv["high"].iloc[resampled_previous_index]
            previous_low[i] = ohlcv["low"].iloc[resampled_previous_index]

            # Check for breaks
            currently_broken_high = (
                ohlc["high"].iloc[i] > previous_high[i] or currently_broken_high
            )
            currently_broken_low = (
                ohlc["low"].iloc[i] < previous_low[i] or currently_broken_low
            )

            # Set break flags
            broken_high[i] = 1 if currently_broken_high else 0
            broken_low[i] = 1 if currently_broken_low else 0

        result = pd.concat(
            [
                pd.Series(previous_high, name="phl_previous_high"),
                pd.Series(previous_low, name="phl_previous_low"),
                pd.Series(broken_high, name="phl_broken_high"),
                pd.Series(broken_low, name="phl_broken_low"),
            ],
            axis=1,
        )

        # Shift to remove look-ahead bias
        # Previous high/low are known at the beginning of the session
        if shift:
            result = result.shift(1).reset_index(drop=True)

        return result

    @classmethod
    def sessions(
        cls,
        ohlc: DataFrame,
        sessions: list = None,
        time_zone: str = "UTC",
        shift: bool = False,
    ) -> DataFrame:
        """
        Trading Sessions - detects multiple sessions at once
        
        Parameters:
        -----------
        ohlc : DataFrame
            DataFrame with OHLC data and datetime index
        sessions : list, optional
            List of session names. Default: ["Sydney", "Tokyo", "London", "New York"]
        time_zone : str
            Timezone of data (default="UTC")
        shift : bool
            Shift values by 1 to remove look-ahead bias (default=False)
        
        Returns:
        --------
        DataFrame with columns for each session:
            - {session}_active, {session}_high, {session}_low
        """
        if sessions is None:
            sessions = ["Sydney", "Tokyo", "London", "New York"]
        
        # Session definitions
        session_times = {
            "Sydney": ("21:00", "06:00"),
            "Tokyo": ("00:00", "09:00"),
            "London": ("07:00", "16:00"),
            "New York": ("13:00", "22:00"),
            "Asian kill zone": ("00:00", "04:00"),
            "London open kill zone": ("06:00", "09:00"),
            "New York kill zone": ("11:00", "14:00"),
            "London close kill zone": ("14:00", "16:00"),
        }
        
        ohlc.index = pd.to_datetime(ohlc.index)
        
        # Convert timezone
        if time_zone != "UTC":
            tz = time_zone.replace("GMT", "Etc/GMT").replace("UTC", "Etc/GMT")
            ohlc.index = ohlc.index.tz_localize(tz).tz_convert("UTC")
        
        result = pd.DataFrame(index=ohlc.index)
        
        for session in sessions:
            start_str, end_str = session_times[session]
            start = datetime.strptime(start_str, "%H:%M")
            end = datetime.strptime(end_str, "%H:%M")
            
            active = np.zeros(len(ohlc), dtype=np.int32)
            high = np.zeros(len(ohlc), dtype=np.float32)
            low = np.zeros(len(ohlc), dtype=np.float32)
            
            for i in range(len(ohlc)):
                t = datetime.strptime(ohlc.index[i].strftime("%H:%M"), "%H:%M")
                
                # Check if in session
                in_session = (start < end and start <= t <= end) or \
                            (start >= end and (t >= start or t <= end))
                
                if in_session:
                    active[i] = 1
                    high[i] = max(ohlc["high"].iloc[i], high[i-1] if i > 0 else 0)
                    low[i] = min(ohlc["low"].iloc[i], 
                            low[i-1] if i > 0 and low[i-1] != 0 else float("inf"))
                else:
                    if i > 0:
                        high[i] = high[i-1]
                        low[i] = low[i-1]
            
            result[f"{session}_active"] = active
            result[f"{session}_high"] = high
            result[f"{session}_low"] = low
        
        if shift:
            result = result.shift(1).reset_index(drop=True)
        
        return result
    
    @classmethod
    def retracements(
        cls, ohlc: DataFrame, swing_length: int = 10, shift: bool = False
    ) -> Series:
        """
        Price Retracements
        Calculates retracement percentages from swing highs/lows.

        Parameters:
        -----------
        ohlc : DataFrame
            DataFrame containing OHLC data
        swing_length : int, optional (default=10)
            Swing length for internal swing calculation
        shift : bool, optional (default=True)
            If True, shifts retracement values to remove look-ahead bias

        Returns:
        --------
        Series
            DataFrame with columns:
            - direction: 1 for bullish retracement, -1 for bearish
            - current_retracement%: current retracement percentage
            - deepest_retracement%: deepest retracement in current move
        """
        # Calculate swing highs/lows internally
        swing_highs_lows = cls.swing_highs_lows(ohlc, swing_length, shift=False)

        # Initialize arrays
        direction = np.zeros(len(ohlc), dtype=np.int32)
        current_retracement = np.zeros(len(ohlc), dtype=np.float64)
        deepest_retracement = np.zeros(len(ohlc), dtype=np.float64)

        top = 0
        bottom = 0

        # Calculate retracements
        for i in range(len(ohlc)):
            if swing_highs_lows["swing_high_low"][i] == 1:
                direction[i] = 1
                top = swing_highs_lows["swing_high_low_level"][i]
            elif swing_highs_lows["swing_high_low"][i] == -1:
                direction[i] = -1
                bottom = swing_highs_lows["swing_high_low_level"][i]
            else:
                direction[i] = direction[i - 1] if i > 0 else 0

            # Calculate retracement percentages
            if direction[i] == 1 and top != bottom:  # Bullish move
                current_retracement[i] = round(
                    100 - (((ohlc["low"].iloc[i] - bottom) / (top - bottom)) * 100), 1
                )
                deepest_retracement[i] = max(
                    (
                        deepest_retracement[i - 1]
                        if i > 0 and direction[i - 1] == 1
                        else 0
                    ),
                    current_retracement[i],
                )
            elif direction[i] == -1 and bottom != top:  # Bearish move
                current_retracement[i] = round(
                    100 - ((ohlc["high"].iloc[i] - top) / (bottom - top)) * 100, 1
                )
                deepest_retracement[i] = max(
                    (
                        deepest_retracement[i - 1]
                        if i > 0 and direction[i - 1] == -1
                        else 0
                    ),
                    current_retracement[i],
                )

        # Shift arrays by 1 to align properly
        current_retracement = np.roll(current_retracement, 1)
        deepest_retracement = np.roll(deepest_retracement, 1)
        direction = np.roll(direction, 1)

        # Remove initial calculations (less reliable)
        remove_first_count = 0
        for i in range(len(direction)):
            if i + 1 == len(direction):
                break
            if direction[i] != direction[i + 1]:
                remove_first_count += 1
            direction[i] = 0
            current_retracement[i] = 0
            deepest_retracement[i] = 0
            if remove_first_count == 3:
                direction[i + 1] = 0
                current_retracement[i + 1] = 0
                deepest_retracement[i + 1] = 0
                break

        result = pd.concat(
            [
                pd.Series(direction, name="retracement_direction"),
                pd.Series(current_retracement, name="retracement_current_retracement%"),
                pd.Series(deepest_retracement, name="retracement_deepest_retracement%"),
            ],
            axis=1,
        )

        # Shift to remove look-ahead bias
        # Retracements use swing points which have swing_length look-ahead
        if shift:
            shift_amount = swing_length
            result = result.shift(shift_amount).reset_index(drop=True)

        return result

    @classmethod
    def algorithmic_order_block(
        cls, ohlc: DataFrame, swing_length: int = 5, shift: bool = False
    ) -> Series:
        """
        Algorithmic Order Block (AOB)
        Detects order blocks that create Break of Structure (BOS).

        Parameters:
        -----------
        ohlc : DataFrame
            DataFrame containing OHLC data
        swing_length : int, optional (default=10)
            Swing length for internal swing calculation
        shift : bool, optional (default=True)
            If True, shifts AOB signals to remove look-ahead bias

        Returns:
        --------
        Series
            DataFrame with columns:
            - type: 1 for bullish AOB, -1 for bearish
            - top: top of AOB range
            - bottom: bottom of AOB range
            - strength: strength indicator (0-100)
        """
        # Calculate swing highs/lows internally
        swing_highs_lows = cls.swing_highs_lows(ohlc, swing_length, shift=False)

        n = len(ohlc)
        swing_hl = swing_highs_lows["swing_high_low"].values
        swing_high_low_level = swing_highs_lows["swing_high_low_level"].values

        # Initialize arrays
        aob_type = np.full(n, np.nan, dtype=np.float32)
        top = np.full(n, np.nan, dtype=np.float32)
        bottom = np.full(n, np.nan, dtype=np.float32)
        strength = np.full(n, np.nan, dtype=np.float32)

        # Find swing points
        swing_indices = np.where(~np.isnan(swing_hl))[0]

        # Process swings to find AOBs
        for i in range(1, len(swing_indices)):
            curr_idx = swing_indices[i]
            prev_idx = swing_indices[i - 1]

            # Bullish AOB: after swing low, before swing high
            if swing_hl[prev_idx] == -1 and swing_hl[curr_idx] == 1:
                # Find lowest low between swing low and swing high
                segment = ohlc.iloc[prev_idx : curr_idx + 1]
                min_idx = segment["low"].idxmin()
                min_idx_pos = segment.index.get_loc(min_idx) + prev_idx

                # Check if this creates BOS (break of previous high)
                if i >= 2:
                    prev_high_idx = swing_indices[i - 2]
                    if swing_hl[prev_high_idx] == 1:
                        prev_high = swing_high_low_level[prev_high_idx]
                        if ohlc["high"].iloc[curr_idx] > prev_high:
                            aob_type[min_idx_pos] = 1
                            top[min_idx_pos] = ohlc["high"].iloc[min_idx_pos]
                            bottom[min_idx_pos] = ohlc["low"].iloc[min_idx_pos]

                            # Calculate strength based on volume and candle size
                            vol_strength = (
                                ohlc["volume"].iloc[min_idx_pos]
                                / ohlc["volume"].rolling(20).mean().iloc[min_idx_pos]
                            )
                            range_strength = (
                                top[min_idx_pos] - bottom[min_idx_pos]
                            ) / (
                                ohlc["high"].rolling(20).mean().iloc[min_idx_pos]
                                - ohlc["low"].rolling(20).mean().iloc[min_idx_pos]
                            )
                            strength[min_idx_pos] = min(
                                100, (vol_strength + range_strength) * 50
                            )

            # Bearish AOB: after swing high, before swing low
            elif swing_hl[prev_idx] == 1 and swing_hl[curr_idx] == -1:
                # Find highest high between swing high and swing low
                segment = ohlc.iloc[prev_idx : curr_idx + 1]
                max_idx = segment["high"].idxmax()
                max_idx_pos = segment.index.get_loc(max_idx) + prev_idx

                # Check if this creates BOS (break of previous low)
                if i >= 2:
                    prev_low_idx = swing_indices[i - 2]
                    if swing_hl[prev_low_idx] == -1:
                        prev_low = swing_high_low_level[prev_low_idx]
                        if ohlc["low"].iloc[curr_idx] < prev_low:
                            aob_type[max_idx_pos] = -1
                            top[max_idx_pos] = ohlc["high"].iloc[max_idx_pos]
                            bottom[max_idx_pos] = ohlc["low"].iloc[max_idx_pos]

                            # Calculate strength
                            vol_strength = (
                                ohlc["volume"].iloc[max_idx_pos]
                                / ohlc["volume"].rolling(20).mean().iloc[max_idx_pos]
                            )
                            range_strength = (
                                top[max_idx_pos] - bottom[max_idx_pos]
                            ) / (
                                ohlc["high"].rolling(20).mean().iloc[max_idx_pos]
                                - ohlc["low"].rolling(20).mean().iloc[max_idx_pos]
                            )
                            strength[max_idx_pos] = min(
                                100, (vol_strength + range_strength) * 50
                            )

        result = pd.concat(
            [
                pd.Series(aob_type, name="aob_type"),
                pd.Series(top, name="aob_top"),
                pd.Series(bottom, name="aob_bottom"),
                pd.Series(strength, name="aob_strength"),
            ],
            axis=1,
        )

        # Shift to remove look-ahead bias
        if shift:
            shift_amount = 1  # shift by 1 candle to avoid look-ahead
            result = result.shift(shift_amount).reset_index(drop=True)

        return result

    @classmethod
    def breaker_block(
        cls, ohlc: DataFrame, swing_length: int = 10, shift: bool = False
    ) -> Series:
        """
        Breaker Block (BB)
        Detects price levels that stop/break trends.

        Parameters:
        -----------
        ohlc : DataFrame
            DataFrame containing OHLC data
        swing_length : int, optional (default=10)
            Swing length for internal swing calculation
        shift : bool, optional (default=True)
            If True, shifts breaker block signals to remove look-ahead bias

        Returns:
        --------
        Series
            DataFrame with columns:
            - type: 1 for bullish breaker, -1 for bearish
            - level: breaker level price
            - strength: rejection strength (0-100)
        """
        # Calculate swing highs/lows internally
        swing_highs_lows = cls.swing_highs_lows(ohlc, swing_length, shift=False)

        n = len(ohlc)
        swing_hl = swing_highs_lows["swing_high_low"].values
        swing_high_low_level = swing_highs_lows["swing_high_low_level"].values

        # Initialize arrays
        bb_type = np.full(n, np.nan, dtype=np.float32)
        level = np.full(n, np.nan, dtype=np.float32)
        strength = np.full(n, np.nan, dtype=np.float32)

        # Find swing points
        swing_indices = np.where(~np.isnan(swing_hl))[0]

        # Process swings to find breaker blocks
        for i in range(2, len(swing_indices)):
            curr_idx = swing_indices[i]
            prev_idx = swing_indices[i - 1]
            prev2_idx = swing_indices[i - 2]

            # Bullish breaker: price breaks high but fails to continue
            if (
                swing_hl[prev2_idx] == 1
                and swing_hl[prev_idx] == -1
                and swing_hl[curr_idx] == 1
            ):
                # Check if price broke previous high but failed
                prev_high = swing_high_low_level[prev2_idx]
                current_high = swing_high_low_level[curr_idx]

                if current_high > prev_high:
                    # Find the candle that broke the high
                    for j in range(prev_idx, curr_idx):
                        if ohlc["high"].iloc[j] > prev_high:
                            bb_type[j] = -1  # Bearish breaker (stops bullish move)
                            level[j] = ohlc["high"].iloc[j]

                            # Calculate strength based on rejection wick
                            body_size = abs(
                                ohlc["close"].iloc[j] - ohlc["open"].iloc[j]
                            )
                            wick_size = ohlc["high"].iloc[j] - max(
                                ohlc["open"].iloc[j], ohlc["close"].iloc[j]
                            )
                            if body_size > 0:
                                strength[j] = min(100, (wick_size / body_size) * 100)
                            break

            # Bearish breaker: price breaks low but fails to continue
            elif (
                swing_hl[prev2_idx] == -1
                and swing_hl[prev_idx] == 1
                and swing_hl[curr_idx] == -1
            ):
                # Check if price broke previous low but failed
                prev_low = swing_high_low_level[prev2_idx]
                current_low = swing_high_low_level[curr_idx]

                if current_low < prev_low:
                    # Find the candle that broke the low
                    for j in range(prev_idx, curr_idx):
                        if ohlc["low"].iloc[j] < prev_low:
                            bb_type[j] = 1  # Bullish breaker (stops bearish move)
                            level[j] = ohlc["low"].iloc[j]

                            # Calculate strength based on rejection wick
                            body_size = abs(
                                ohlc["close"].iloc[j] - ohlc["open"].iloc[j]
                            )
                            wick_size = (
                                min(ohlc["open"].iloc[j], ohlc["close"].iloc[j])
                                - ohlc["low"].iloc[j]
                            )
                            if body_size > 0:
                                strength[j] = min(100, (wick_size / body_size) * 100)
                            break

        result = pd.concat(
            [
                pd.Series(bb_type, name="bb_type"),
                pd.Series(level, name="bb_level"),
                pd.Series(strength, name="bb_strength"),
            ],
            axis=1,
        )

        # Shift to remove look-ahead bias
        if shift:
            shift_amount = 1  # shift by 1 candle to avoid look-ahead
            result = result.shift(shift_amount).reset_index(drop=True)

        return result

    @classmethod
    def mitigation_block(cls, ohlc: DataFrame, shift: bool = False) -> Series:
        """
        Mitigation Block (MB)
        Detects order blocks at FVG mitigation points.

        Parameters:
        -----------
        ohlc : DataFrame
            DataFrame containing OHLC data
        shift : bool, optional (default=True)
            If True, shifts mitigation block signals to remove look-ahead bias

        Returns:
        --------
        Series
            DataFrame with columns:
            - type: 1 for bullish mitigation, -1 for bearish
            - top: top of mitigation block
            - bottom: bottom of mitigation block
            - fvg_index: index of related FVG
        """
        # Calculate FVG data internally
        fvg_data = cls.fvg(ohlc, shift=False)

        n = len(ohlc)
        fvg = fvg_data["fvg"].values
        mitigated_idx = fvg_data["fvg_mitigated_index"].values

        # Initialize arrays
        mb_type = np.full(n, np.nan, dtype=np.float32)
        top = np.full(n, np.nan, dtype=np.float32)
        bottom = np.full(n, np.nan, dtype=np.float32)
        fvg_index = np.full(n, np.nan, dtype=np.float32)

        # Find mitigated FVGs
        for i in range(n):
            if not np.isnan(mitigated_idx[i]) and mitigated_idx[i] > 0:
                mit_idx = int(mitigated_idx[i])

                # Bullish FVG mitigation (price returns to fill gap)
                if fvg[i] == 1 and mit_idx < n:
                    mb_type[mit_idx] = 1
                    top[mit_idx] = ohlc["high"].iloc[mit_idx]
                    bottom[mit_idx] = ohlc["low"].iloc[mit_idx]
                    fvg_index[mit_idx] = i

                # Bearish FVG mitigation
                elif fvg[i] == -1 and mit_idx < n:
                    mb_type[mit_idx] = -1
                    top[mit_idx] = ohlc["high"].iloc[mit_idx]
                    bottom[mit_idx] = ohlc["low"].iloc[mit_idx]
                    fvg_index[mit_idx] = i

        result = pd.concat(
            [
                pd.Series(mb_type, name="mb_type"),
                pd.Series(top, name="mb_top"),
                pd.Series(bottom, name="mb_bottom"),
                pd.Series(fvg_index, name="mb_fvg_index"),
            ],
            axis=1,
        )

        # ROUND MB values to match CSV exactly
        result["mb_top"] = result["mb_top"].round(2)
        result["mb_bottom"] = result["mb_bottom"].round(2)
        # Shift to remove look-ahead bias
        # Mitigation block uses FVG which has 2 candle look-ahead
        if shift:
            shift_amount = 2
            result = result.shift(shift_amount).reset_index(drop=True)

        return result

    @classmethod
    def bpr(
        cls, ohlc: DataFrame, lookback_periods: int = 20, shift: bool = False
    ) -> Series:
        """
        Balanced Price Range (BPR)
        Identifies price ranges where market is balanced.

        Parameters:
        -----------
        ohlc : DataFrame
            DataFrame containing OHLC data
        lookback_periods : int, optional (default=20)
            Periods to look back for balance calculation
        shift : bool, optional (default=True)
            If True, shifts BPR signals to remove look-ahead bias

        Returns:
        --------
        Series
            DataFrame with columns:
            - top: top of balanced range
            - bottom: bottom of balanced range
            - strength: balance strength (0-100)
        """
        n = len(ohlc)

        # Initialize arrays
        bpr_top = np.full(n, np.nan, dtype=np.float32)
        bpr_bottom = np.full(n, np.nan, dtype=np.float32)
        strength = np.full(n, np.nan, dtype=np.float32)

        # Calculate BPR for each point
        for i in range(lookback_periods, n):
            # Get recent price action
            recent_highs = ohlc["high"].iloc[i - lookback_periods : i].values
            recent_lows = ohlc["low"].iloc[i - lookback_periods : i].values

            # Calculate price distribution
            price_range = np.linspace(np.min(recent_lows), np.max(recent_highs), 100)
            density = np.zeros(100)

            # Build price density histogram
            for price in np.concatenate([recent_highs, recent_lows]):
                idx = min(
                    99,
                    int(
                        (price - np.min(recent_lows))
                        / (np.max(recent_highs) - np.min(recent_lows) + 0.0001)
                        * 100
                    ),
                )
                density[idx] += 1

            # Find high density areas (balanced ranges)
            threshold = np.mean(density) * 1.5
            high_density_indices = np.where(density > threshold)[0]

            if len(high_density_indices) > 0:
                # Find largest contiguous high density area
                groups = []
                current_group = [high_density_indices[0]]

                for idx in high_density_indices[1:]:
                    if idx == current_group[-1] + 1:
                        current_group.append(idx)
                    else:
                        groups.append(current_group)
                        current_group = [idx]
                groups.append(current_group)

                # Get the largest group
                largest_group = max(groups, key=len) if groups else []

                # Minimum size for valid BPR
                if len(largest_group) >= 5:
                    bpr_bottom[i] = np.min(recent_lows) + (largest_group[0] / 100) * (
                        np.max(recent_highs) - np.min(recent_lows)
                    )
                    bpr_top[i] = np.min(recent_lows) + (largest_group[-1] / 100) * (
                        np.max(recent_highs) - np.min(recent_lows)
                    )

                    # Calculate balance strength
                    range_size = bpr_top[i] - bpr_bottom[i]
                    total_range = np.max(recent_highs) - np.min(recent_lows)
                    time_in_range = len(largest_group) / lookback_periods

                    if total_range > 0:
                        strength[i] = min(
                            100,
                            (1 - (range_size / total_range)) * 50
                            + (time_in_range * 50),
                        )

        result = pd.concat(
            [
                pd.Series(bpr_top, name="bpr_top"),
                pd.Series(bpr_bottom, name="bpr_bottom"),
                pd.Series(strength, name="bpr_strength"),
            ],
            axis=1,
        )

        # Shift to remove look-ahead bias
        # BPR uses only past data, but needs 1 candle shift for proper alignment
        if shift:
            result = result.shift(1).reset_index(drop=True)

        return result

    @classmethod
    def liquidity_swing_hl(
        cls, ohlc: DataFrame, swing_length: int = 10, shift: bool = False
    ) -> Series:
        """
        Swing High/Low Liquidity
        Detects liquidity at swing highs/lows and tracks sweeps.

        Parameters:
        -----------
        ohlc : DataFrame
            DataFrame containing OHLC data
        swing_length : int, optional (default=10)
            Swing length for internal swing calculation
        shift : bool, optional (default=True)
            If True, shifts liquidity signals to remove look-ahead bias

        Returns:
        --------
        Series
            DataFrame with columns:
            - type: 1 for buy-side (swing high), -1 for sell-side (swing low)
            - level: the liquidity level
            - swing_index: index of the swing high/low
            - swept: index where liquidity was swept
        """
        # Calculate swing highs/lows internally
        swing_highs_lows = cls.swing_highs_lows(ohlc, swing_length, shift=False)

        n = len(ohlc)
        swing_hl = swing_highs_lows["swing_high_low"].values
        swing_high_low_level = swing_highs_lows["swing_high_low_level"].values

        # Initialize arrays
        liq_type = np.full(n, np.nan, dtype=np.float32)
        level = np.full(n, np.nan, dtype=np.float32)
        swing_idx = np.full(n, np.nan, dtype=np.float32)
        swept = np.full(n, np.nan, dtype=np.float32)

        # Get swing indices
        swing_indices = np.where(~np.isnan(swing_hl))[0]
        swept_mask = np.zeros(len(swing_indices), dtype=bool)

        # Check for liquidity sweeps
        for i in range(n):
            for j, swing_i in enumerate(swing_indices):
                if swing_i >= i or swept_mask[j]:
                    continue

                # Check swing high sweep
                if (
                    swing_hl[swing_i] == 1
                    and ohlc["high"].iloc[i] > swing_high_low_level[swing_i]
                ):
                    liq_type[i] = 1
                    level[i] = swing_high_low_level[swing_i]
                    swing_idx[i] = swing_i
                    swept[i] = i
                    swept_mask[j] = True

                # Check swing low sweep
                if (
                    swing_hl[swing_i] == -1
                    and ohlc["low"].iloc[i] < swing_high_low_level[swing_i]
                ):
                    liq_type[i] = -1
                    level[i] = swing_high_low_level[swing_i]
                    swing_idx[i] = swing_i
                    swept[i] = i
                    swept_mask[j] = True

        result = pd.concat(
            [
                pd.Series(liq_type, name="lshl_type"),
                pd.Series(level, name="lshl_level"),
                pd.Series(swing_idx, name="lshl_swing_index"),
                pd.Series(swept, name="lshl_swept"),
            ],
            axis=1,
        )

        # Shift to remove look-ahead bias
        # it uses only past data, but needs 1 candle shift for proper alignment
        if shift:
            result = result.shift(1).reset_index(drop=True)
