"""
Time & Multi-Timeframe Features (Category: TIME)

This module provides the TimeFeatures class, which extracts temporal 
periodicity (Hour, Day) using trigonometric encodings (Sine/Cosine) 
and decomposes price signals into frequency components using FFT.
"""

import numpy as np

class TimeFeatures:
    """
    Calculates time-based features and seasonal frequency components.
    
    Includes:
    1. Direct temporal extraction (Hour, Day, Weekend).
    2. Cyclical time encodings to preserve continuity (23:59 close to 00:01).
    3. Fourier analysis (FFT) to detect 12h and 24h market cycles.

    Attributes:
        close (np.ndarray): Price series used for frequency analysis.
        n (int): Total number of samples.
        eps (float): Stability constant.
        datetime (List[datetime]): Array of timestamps corresponding to data.
    """

    def __init__(self, close: np.ndarray, n: int, eps: float, datetime: np.ndarray):
        """
        Initializes the TimeFeatures calculator.

        Args:
            close (np.ndarray): Closing prices.
            n (int): Data length.
            eps (float): Stability epsilon.
            datetime (np.ndarray): Timestamps (assumed to be pandas or datetime objects).
        """
        self.close = close
        self.n = n
        self.eps = eps
        self.datetime = datetime

    def hour_of_day(self) -> np.ndarray:
        """
        TIME01: Hour of Day.
        Extracts the hour component (0-23). 
        Useful for capturing intraday seasonality (Asian vs London vs NY sessions).
        """
        hour = np.full(self.n, np.nan)

        if self.datetime is None:
            return hour

        for i, ts in enumerate(self.datetime):
            try:
                # Extracts raw hour (integer)
                hour[i] = ts.hour
            except AttributeError:
                pass
        return hour

    def day_of_week(self) -> np.ndarray:
        """
        TIME02: Day of Week.
        Extracts the day of the week (0=Monday, 6=Sunday).
        Captures weekly patterns (e.g., weekend liquidity drops, Monday gaps).
        """
        dow = np.full(self.n, np.nan)

        if self.datetime is None:
            return dow

        for i, ts in enumerate(self.datetime):
            try:
                # 0-6 range mapping
                dow[i] = ts.weekday()
            except AttributeError:
                pass
        return dow

    def time_sine(self) -> np.ndarray:
        """
        TIME03: Cyclical Hour Sine Encoding.
        Formula: sin(2 * pi * Hour / 24)
        Ensures the model treats 23:00 and 01:00 as similar (unlike raw integers).
        """
        hour = self.hour_of_day()
        # Maps the 24-hour cycle onto a unit circle sine component
        return np.sin(2 * np.pi * hour / 24.0)

    def time_cosine(self) -> np.ndarray:
        """
        TIME04: Cyclical Hour Cosine Encoding.
        Formula: cos(2 * pi * Hour / 24)
        Provides the second dimension for unique unit circle representation.
        """
        hour = self.hour_of_day()
        # Maps the 24-hour cycle onto a unit circle cosine component
        return np.cos(2 * np.pi * hour / 24.0)

    def is_weekend(self) -> np.ndarray:
        """
        TIME05: Weekend Binary Flag.
        1.0 if Saturday (5) or Sunday (6). 
        Significant for crypto markets which trade 24/7 with different weekend dynamics.
        """
        dow = self.day_of_week()
        return (dow >= 5).astype(float)

    def fourier_component_24h(self, window: int = 100) -> np.ndarray:
        """
        TIME06: 24-Hour Fourier Component.
        Extracts the magnitude of the frequency component corresponding to 24 units.
        If data is 1-hour candles, this detects daily cyclical patterns.
        """
        fft_mag = np.full(self.n, np.nan)

        for i in range(window, self.n):
            price_w = self.close[i - window:i]

            if not np.any(np.isnan(price_w)):
                # Compute Fast Fourier Transform
                fft_res = np.fft.fft(price_w)
                
                # Formula: Frequency = index / window. For period T, index k = window / T.
                # Detects the specific frequency '1/24'
                idx_24h = int(window / 24)
                
                if idx_24h < len(fft_res):
                    # Absolute magnitude of the complex component
                    fft_mag[i] = np.abs(fft_res[idx_24h]) / window
        return fft_mag

    def fourier_component_12h(self, window: int = 100) -> np.ndarray:
        """
        TIME07: 12-Hour Fourier Component.
        Magnitude of the 'twice-daily' frequency component.
        Detects higher-frequency rhythms (e.g., overlapping market sessions).
        """
        fft_mag = np.full(self.n, np.nan)

        for i in range(window, self.n):
            price_w = self.close[i - window:i]

            if not np.any(np.isnan(price_w)):
                fft_res = np.fft.fft(price_w)
                # Index for period T=12
                idx_12h = int(window / 12)
                
                if idx_12h < len(fft_res):
                    fft_mag[i] = np.abs(fft_res[idx_12h]) / window
        return fft_mag
