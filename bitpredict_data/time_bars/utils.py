import pandas as pd

def custom_round_series(s: pd.Series) -> pd.Series:
    """
    Round numeric series according to the rules:
    1. If value > 0.01, round to 4 decimals.
    2. If value < 0.01, round starting from the first non-zero digit
       in the decimal part, keeping up to 5 significant decimal digits.
    """
    def round_value(x):
        if pd.isna(x):
            return x
        if abs(x) >= 0.01:
            return round(x, 4)
        else:
            # Convert to string with high precision
            s = f"{x:.12f}"
            # Remove leading "0."
            decimals = s.split('.')[1]
            # Find index of first non-zero digit
            for i, d in enumerate(decimals):
                if d != '0':
                    # Keep next 4 digits after first non-zero digit
                    end_index = i + 5
                    rounded = decimals[:end_index]
                    result = float('0.' + rounded)
                    return result
            return 0.0  # if all decimals are 0
    return s.apply(round_value)
