import pandas as pd
from bitpredict.common.ta.smc.core import smc as _smc
from bitpredict.common.logging import get_logger
from bitpredict.common.constants import OHLCV_COLUMNS

logger = get_logger(__name__)


SMC_MODULES = [
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
    "liquidity_swing_hl",
]


class SMCBase:

    def run(self, df, *names, prefix=True, fill_na=True, shift=False):
        """
        Run one or more SMC indicators and merge their results with the
        original OHLCV data.

        Parameters
        ----------
        df : pandas.DataFrame
            OHLCV data indexed by time or containing a `datetime` column.
        *names : str
            Indicator names to execute. Use "all" to run every SMC module.
        prefix : bool, default True
            Prefix indicator columns with their indicator name.
        fill_na : bool, default True
            Replace NaN values in the final output with 0.

        Returns
        -------
        pandas.DataFrame
            Original OHLCV data merged with indicator outputs. The `datetime`
            column is always present and appears first.
        """

        # Ensure a datetime column exists
        if "datetime" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df["datetime"] = df.index
            else:
                raise ValueError(
                    "DataFrame must have a 'datetime' column or a DatetimeIndex"
                )

        # Sort by datetime
        if isinstance(df.index, pd.DatetimeIndex):
            df.sort_index(inplace=True)
        else:
            df.sort_values("datetime", inplace=True)

        # Validate the minimum required schema
        missing_cols = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing columns: {missing_cols}")

        # Resolve indicator list
        if not names or (len(names) == 1 and names[0].lower() == "all"):
            names = SMC_MODULES
            logger.info("Running all SMC modules")

        # Start with the base OHLCV data
        result = df.reset_index(drop=True)
        executed = []

        for name in names:
            fn = getattr(_smc, name, None)
            if fn is None:
                logger.warning(f"Indicator '{name}' not found, skipping")
                continue

            out = fn(df, shift=shift)
            if out is None or out.empty or out.isna().all().all():
                logger.warning(f"Indicator '{name}' returned no usable data")
                continue

            if prefix:
                out.columns = [f"smc_{name}" for name in out.columns]

            result = pd.concat(
                [result, out.reset_index(drop=True)],
                axis=1,
            )
            executed.append(name)

        if fill_na:
            result.fillna(0, inplace=True)

        # Keep datetime as the leading column for downstream consumers
        cols = result.columns.tolist()
        cols.insert(0, cols.pop(cols.index("datetime")))
        result = result[cols]

        logger.info(f"Merged {len(executed)} indicators | Shape: {result.shape}")

        return result
