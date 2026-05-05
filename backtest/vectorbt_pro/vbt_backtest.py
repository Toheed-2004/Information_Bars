"""
Optimized VBT Pro backtest with vectorized ledger creation
"""
import warnings
import numpy as np
import pandas as pd
from common.constants import OHLCV_COLUMNS
from backtest.vectorbt_pro.utils import calculate_atr_for_trailing_stop
import vectorbtpro as vbt
from vectorbtpro import Portfolio
from vectorbtpro.portfolio import enums as pf_enums


class VBTBacktestOptimized:
    def __init__(self, df_signals, df_ohlcv, backtest_params, config_data=None, df_bars=None, pf_object=None):
        self.params = self._validate_backtest_params(backtest_params)

        self.risk_management = self.params['risk_management']
        self.starting_balance = self.params['starting_balance']

        # TP/SL from static config or top-level params
        static_risk = self.risk_management.get('static', {})
        if static_risk.get('enabled', False):
            self.take_profit_percent = static_risk.get('take_profit', 0.0) / 100
            self.stop_loss_percent = static_risk.get('stop_loss', 0.0) / 100
        else:
            tp = self.params.get('take_profit', 0.0)
            sl = self.params.get('stop_loss', 0.0)
            if tp == 0.0 and sl == 0.0:
                warnings.warn(
                    "No TP/SL configured (static disabled, no top-level take_profit/stop_loss). "
                    "Positions will only exit on direction change.",
                    UserWarning
                )
            self.take_profit_percent = tp / 100
            self.stop_loss_percent = sl / 100

        self.transaction_fee_percent = (self.params['transaction_fee'] / 100.0) * self.params['leverage']
        self.leverage = self.params['leverage']
        self.position_size = self.params['position_size']
        self.slippage = self.params['slippage'] / 100
        self.create_ledger = self.params['create_ledger']
        # config_data holds bar metadata (exchange, symbol, timeframe, bar_type)
        self.config_data = config_data or {}
        # df_bars: pre-loaded bar OHLCV — avoids a DB fetch in ATR calculation
        self.df_bars = df_bars

        # self.df_signals = self._process_signals(df_signals)
        self.existing_pf = pf_object
        if df_signals.empty:
            self.df_signals = pd.DataFrame(columns=['datetime', 'signals']).set_index('datetime')
        else:
            self.df_signals = self._process_signals(df_signals)

        self.df_ohlcv = self._process_ohlcv(df_ohlcv)

        self._handle_empty_signals_with_existing_pf()

        self._prepare_vbt_data()

    def _validate_backtest_params(self, params):
        """Validate and set default parameters"""
        defaults = {
            'starting_balance': 10000,
            'transaction_fee': 0.1,
            'leverage': 1.0,
            'position_size': 0.5,
            'slippage': 0.05,
            'create_ledger': True,
            'direction': 'both',
            'risk_management': {
                'static': {
                    'enabled': False,
                    'take_profit': 3.0,
                    'stop_loss': 2.0
                },
                'time_stop': {
                    'enabled': False,
                    'max_duration': '12h',
                },
                'atr_stop': {
                    'enabled': False,
                    'period': 14,
                    'multiplier': 2.0,
                    'tp_enabled': False,
                    'tp_multiplier': 3.0
                },
                'chandelier_stop': {
                    'enabled': False,
                    'period': 22,
                    'multiplier': 3.0,
                    'tp_enabled': False,
                    'tp_multiplier': 4.0
                },
                'trailing_stop': {
                    'enabled': False,
                    'activation_pct': 1.0,
                    'trail_pct': 0.5
                }
            },
            'zero_signal_mode': 'hold_position'
        }

        validated_params = defaults.copy()

        for k, v in params.items():
            if k == 'risk_management' and isinstance(v, dict):
                for stop_type, stop_cfg in v.items():
                    if stop_type in validated_params['risk_management'] and isinstance(stop_cfg, dict):
                        validated_params['risk_management'][stop_type].update(stop_cfg)
                    else:
                        validated_params['risk_management'][stop_type] = stop_cfg
            else:
                validated_params[k] = v

        valid_modes = ['close_position', 'hold_position']
        if validated_params['zero_signal_mode'] not in valid_modes:
            raise ValueError(f"zero_signal_mode must be one of: {valid_modes}")
        
        valid_directions = ['long_only', 'short_only', 'both']
        if validated_params['direction'] not in valid_directions:
            raise ValueError(f"direction must be one of: {valid_directions}")

        return validated_params

    def _process_signals(self, df_signals):
        """Process and validate signals DataFrame"""
        if df_signals is None or len(df_signals) == 0:
            raise ValueError("df_signals cannot be None or empty")

        if 'datetime' not in df_signals.columns:
            raise ValueError("df_signals missing required column: datetime")

        signal_cols = [col for col in df_signals.columns if col != 'datetime']
        if len(signal_cols) == 0:
            raise ValueError("df_signals must have at least one signal column besides 'datetime'")

        df_signals = df_signals.copy()
        df_signals["datetime"] = pd.to_datetime(df_signals["datetime"])
        return df_signals.set_index("datetime")

    def _handle_empty_signals_with_existing_pf(self):
        """Create dummy zero signals if df_signals is empty but we have an existing portfolio."""
        
        # Check if we need to create dummy signals
        if not (self.df_signals.empty and self.existing_pf is not None):
            return
        
        # Get the last timestamp from the existing portfolio
        last_pf_timestamp = self.existing_pf.wrapper.index[-1]

        last_timestamp_minute = self.existing_pf.wrapper.index[-1]
        trades = self.existing_pf.trades.records
        if len(trades) > 0 and trades.iloc[-1]['status'] == 0:
            last_closed_idx = trades.iloc[-1]['exit_idx']
            last_timestamp_minute = self.existing_pf.wrapper.index[int(last_closed_idx)]
        
        # Find all OHLCV timestamps after the last portfolio timestamp

        self.df_ohlcv = self.df_ohlcv.loc[self.df_ohlcv.index > last_timestamp_minute]
        self.df_signals = self.df_ohlcv.loc[self.df_ohlcv.index > last_timestamp_minute, []].assign(signals=0)

        
    def _process_ohlcv(self, df_ohlcv):
        """Process and validate OHLCV DataFrame"""
        if df_ohlcv is None or len(df_ohlcv) == 0:
            raise ValueError("df_ohlcv cannot be None or empty")

        for col in OHLCV_COLUMNS:
            if col not in df_ohlcv.columns:
                raise ValueError(f"df_ohlcv missing required column: {col}")

        # Convert datetime in-place if needed (no copy to save memory)
        if not pd.api.types.is_datetime64_any_dtype(df_ohlcv["datetime"]):
            df_ohlcv["datetime"] = pd.to_datetime(df_ohlcv["datetime"])
        return df_ohlcv.set_index("datetime")

    def _prepare_vbt_data(self):
        """Prepare OHLCV data and signals for VBT Pro"""
        self._validate_minute_data()
        # 🔥 CRITICAL: Use reference instead of copy to save 158 MB per trial
        self.minute_data = self.df_ohlcv

        direction = self.params['direction']
        if direction == 'long_only':
            self.df_signals[self.df_signals == -1] = 0
        elif direction == 'short_only':
            self.df_signals[self.df_signals == 1] = 0

        self._create_signals()

    def _validate_minute_data(self):
        """Validate that 1-minute OHLCV data is provided.

        df_ohlcv must always be 1m data regardless of the bar type used for signals.
        df_signals can be at any bar resolution (1h, dollar bars, volume bars, etc.).
        """
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in self.df_ohlcv.columns:
                raise ValueError(f"df_ohlcv missing required column: {col}")

        # if len(self.df_ohlcv) < 2:
        #     raise ValueError("df_ohlcv must have at least 2 rows")

        ohlcv_deltas = self.df_ohlcv.index.to_series().diff().dropna()
        if ohlcv_deltas.min() > pd.Timedelta('1min'):
            raise ValueError(
                "df_ohlcv must be 1-minute data for accurate stop simulation. "
                f"Detected minimum bar interval: {ohlcv_deltas.min()}. "
                "Pass 1m OHLCV as df_ohlcv. df_signals can be any bar type."
            )

    def _create_signals(self):
        """Convert prediction signals to VBT Pro entry/exit signals"""
        self.signal_columns = list(self.df_signals.columns)

        if len(self.signal_columns) == 0:
            raise ValueError("No signal columns found in df_signals")

        zero_mode = self.params['zero_signal_mode']

        if zero_mode == 'hold_position':
            processed_signals = {}
            for col in self.signal_columns:
                non_zero_mask = self.df_signals[col] != 0
                valid_signals = self.df_signals.loc[non_zero_mask, col]
                processed_signals[col] = valid_signals.reindex(
                    self.minute_data.index, method='ffill'
                ).fillna(0)
            pred_direction = pd.DataFrame(processed_signals, index=self.minute_data.index)
        else:
            pred_direction = self.df_signals[self.signal_columns].reindex(
                self.minute_data.index, method='ffill'
            ).fillna(0)

        long_entries, short_entries = self._create_position_aware_signals()

        if zero_mode == 'close_position':
            long_exits = ((pred_direction == 0) & (pred_direction.shift(1) == 1)) | \
                         ((pred_direction == -1) & (pred_direction.shift(1) == 1))
            short_exits = ((pred_direction == 0) & (pred_direction.shift(1) == -1)) | \
                          ((pred_direction == 1) & (pred_direction.shift(1) == -1))
        else:
            long_exits = (pred_direction == -1) & (pred_direction.shift(1) == 1)
            short_exits = (pred_direction == 1) & (pred_direction.shift(1) == -1)

        if len(self.signal_columns) == 1:
            col = self.signal_columns[0]
            self.long_entries = long_entries[col]
            self.long_exits = long_exits[col]
            self.short_entries = short_entries[col]
            self.short_exits = short_exits[col]
            self.pred_direction = pred_direction[col]
        else:
            self.long_entries = long_entries
            self.long_exits = long_exits
            self.short_entries = short_entries
            self.short_exits = short_exits
            self.pred_direction = pred_direction

        # Only filter signals if resuming from existing portfolio
        if self.existing_pf is not None:
            last_pf_timestamp = self.existing_pf.wrapper.index[-1]

            last_timestamp_minute = self.existing_pf.wrapper.index[-1]
            trades = self.existing_pf.trades.records
            if len(trades) > 0 and trades.iloc[-1]['status'] == 0:
                last_closed_idx = trades.iloc[-1]['exit_idx']
                last_timestamp_minute = self.existing_pf.wrapper.index[int(last_closed_idx)]
            
            self.long_entries = long_entries.loc[long_entries.index > last_timestamp_minute]
            self.long_exits = long_exits.loc[long_exits.index > last_timestamp_minute]
            self.short_entries = short_entries.loc[short_entries.index > last_timestamp_minute]
            self.short_exits = short_exits.loc[short_exits.index > last_timestamp_minute]

    def _create_position_aware_signals(self):
        """Create entry signals at original signal timestamps (vectorized)."""
        
        direction = self.params['direction']

        long_entries = pd.DataFrame(False, index=self.minute_data.index, columns=self.signal_columns)
        short_entries = pd.DataFrame(False, index=self.minute_data.index, columns=self.signal_columns)

        non_zero_mask = (self.df_signals[self.signal_columns] != 0).any(axis=1)
        original_signals = self.df_signals[non_zero_mask]

        valid_mask = original_signals.index.isin(self.minute_data.index)
        skipped = (~valid_mask).sum()
        if skipped > 0:
            warnings.warn(
                f"{skipped} signal timestamp(s) not found in OHLCV index and will be skipped. "
                "Ensure signal and OHLCV datetimes are aligned.",
                UserWarning
            )

        valid_signals = original_signals[valid_mask]

        for col in self.signal_columns:
            long_entries.loc[valid_signals.index[valid_signals[col] == 1], col] = True
            short_entries.loc[valid_signals.index[valid_signals[col] == -1], col] = True
        # Get the last timestamp from the existing portfolio
        

        return long_entries, short_entries

    def _fetch_atr(self, period):
        """Fetch and align ATR at signal bar frequency to the minute data index.

        If df_bars was provided at construction, it is used directly (avoids a DB call).
        Otherwise bar data is fetched from the DB via calculate_atr_for_trailing_stop.
        """
        exchange = self.config_data.get('exchange') or self.params.get('exchange')
        timeframe = self.config_data.get('timeframe') or self.params.get('timeframe')
        symbol = self.config_data.get('symbol') or self.params.get('symbol')
        bar_type = self.config_data.get('bar_type') or self.params.get('bar_type')
        atr_ind = calculate_atr_for_trailing_stop(
            self.df_signals, period=period,
            exchange=exchange, bar_type=bar_type, timeframe=timeframe, symbol=symbol,
            df_bars=self.df_bars
        )
        # Ensure ATR index is UTC to match minute_data index
        if atr_ind.index.tz is not None:
            # Already has timezone, convert to UTC
            atr_ind = atr_ind.copy()
            atr_ind.index = atr_ind.index.tz_convert('UTC')
        else:
            # Naive, assume UTC and localize
            atr_ind = atr_ind.copy()
            atr_ind.index = atr_ind.index.tz_localize('UTC')
        
        # Forward-fill to every minute bar
        return atr_ind.reindex(self.minute_data.index, method='ffill')

    def run(self):
        """Run the backtest using VBT Pro.

        VBT evaluates sl_stop, tp_stop, tsl_stop, and td_stop simultaneously.
        Whichever triggers first on any bar closes the position.

        Each slot (sl_stop / tp_stop / tsl_stop) can only hold one value.
        When multiple enabled stop types compete for the same slot, the more
        dynamic one wins (ATR > Static for SL/TP; Chandelier > Trailing for TSL)
        and a warning is issued.
        """
        sl_stop = None
        tp_stop = None
        tsl_stop = None
        tsl_th = None
        td_stop = None

        # Track which mechanism occupies each slot — used by ledger action column
        self._sl_mechanism = None
        self._tp_mechanism = None
        self._tsl_mechanism = None

        risk_mgmt = self.risk_management

        # STATIC TP/SL — fills sl_stop and tp_stop with scalar percentages
        if risk_mgmt['static']['enabled']:
            if self.stop_loss_percent > 0:
                sl_stop = self.stop_loss_percent
                self._sl_mechanism = 'Static'
            if self.take_profit_percent > 0:
                tp_stop = self.take_profit_percent
                self._tp_mechanism = 'Static'

        # TRAILING STOP — fills tsl_stop
        if risk_mgmt['trailing_stop']['enabled']:
            tsl_stop = risk_mgmt['trailing_stop']['trail_pct'] / 100
            tsl_th = risk_mgmt['trailing_stop']['activation_pct'] / 100
            self._tsl_mechanism = 'Trailing'

        # TIME-BASED STOP — max_duration converted to bar count
        if risk_mgmt['time_stop']['enabled']:
            max_duration = risk_mgmt['time_stop']['max_duration']
            bar_interval = self.minute_data.index.to_series().diff().median()
            if pd.isna(bar_interval) or bar_interval.total_seconds() == 0:
                raise ValueError("Cannot determine bar interval for time_stop conversion.")
            td_stop = max(1, int(pd.Timedelta(max_duration) / bar_interval))

        # ATR STOP — fills sl_stop (and optionally tp_stop) with ATR-based series
        if risk_mgmt['atr_stop']['enabled']:
            atr_aligned = self._fetch_atr(risk_mgmt['atr_stop']['period'])

            if self._sl_mechanism is not None:
                warnings.warn(
                    f"ATR SL conflicts with {self._sl_mechanism} SL for sl_stop — ATR takes priority.",
                    UserWarning
                )
            sl_stop = (atr_aligned * risk_mgmt['atr_stop']['multiplier']) / self.minute_data['close']
            self._sl_mechanism = 'ATR'

            if risk_mgmt['atr_stop'].get('tp_enabled', False):
                if self._tp_mechanism is not None:
                    warnings.warn(
                        f"ATR TP conflicts with {self._tp_mechanism} TP for tp_stop — ATR takes priority.",
                        UserWarning
                    )
                tp_stop = (atr_aligned * risk_mgmt['atr_stop']['tp_multiplier']) / self.minute_data['close']
                self._tp_mechanism = 'ATR'

        # CHANDELIER STOP — fills tsl_stop (and optionally tp_stop) with ATR-based series
        if risk_mgmt['chandelier_stop']['enabled']:
            atr_aligned = self._fetch_atr(risk_mgmt['chandelier_stop']['period'])

            if self._tsl_mechanism is not None:
                warnings.warn(
                    f"Chandelier conflicts with {self._tsl_mechanism} for tsl_stop — Chandelier takes priority.",
                    UserWarning
                )
            tsl_stop = (atr_aligned * risk_mgmt['chandelier_stop']['multiplier']) / self.minute_data['close']
            self._tsl_mechanism = 'Chandelier'

            if risk_mgmt['chandelier_stop'].get('tp_enabled', False):
                if self._tp_mechanism is not None:
                    warnings.warn(
                        f"Chandelier TP conflicts with {self._tp_mechanism} TP for tp_stop — Chandelier takes priority.",
                        UserWarning
                    )
                tp_stop = (atr_aligned * risk_mgmt['chandelier_stop']['tp_multiplier']) / self.minute_data['close']
                self._tp_mechanism = 'Chandelier'

        portfolio_kwargs = {
            "close": self.minute_data['close'],
            "open": self.minute_data['open'],
            "high": self.minute_data['high'],
            "low": self.minute_data['low'],
            "price": self.minute_data['open'],
            "long_entries": self.long_entries,
            "long_exits": self.long_exits,
            "short_entries": self.short_entries,
            "short_exits": self.short_exits,
            "size": self.position_size,
            "size_type": 'percent',
            "init_cash": self.starting_balance,
            "fees": self.transaction_fee_percent,
            "slippage": self.slippage,
            "sl_stop": sl_stop,
            "tp_stop": tp_stop,
            "tsl_stop": tsl_stop,
            "tsl_th": tsl_th,
            "use_stops": True,
            "delta_format": pf_enums.DeltaFormat.Percent,
            "stop_entry_price": pf_enums.StopEntryPrice.Open,
            "stop_exit_price": pf_enums.StopExitPrice.Stop,
            "accumulate": False,
            "upon_long_conflict": pf_enums.ConflictMode.Ignore,
            "upon_short_conflict": pf_enums.ConflictMode.Ignore,
            "upon_dir_conflict": pf_enums.DirectionConflictMode.Opposite,
            "leverage": self.leverage
        }

        if td_stop is not None:
            portfolio_kwargs["td_stop"] = td_stop
            portfolio_kwargs["time_delta_format"] = 'rows'

        pf_object = self.existing_pf
        if pf_object is not None:
            portfolio_kwargs["stack"] = False
            new_pf = pf_object.update(**portfolio_kwargs)

            last_idx = pf_object.wrapper.index[-1]
            new_pf = new_pf.loc[new_pf.wrapper.index > last_idx]
            self.pf = vbt.PF.row_stack(      
                (pf_object, new_pf), 
                chained=True, 
                preparer=new_pf.preparer                        
                )
        else:
            self.pf = Portfolio.from_signals(**portfolio_kwargs, attach_preparer=True)

        df_ledger = self._create_ledger_vectorized() if self.create_ledger else None

        return self.pf, df_ledger

    def _create_ledger_vectorized(self):
        """Vectorized ledger creation for single/multi-strategy portfolios"""
        if getattr(self.pf.wrapper, 'ncol', 1) > 1:
            ledgers = {}
            for strategy_name in self.pf.wrapper.columns:
                strategy_pf = self.pf.select_col(strategy_name)
                ledgers[strategy_name] = self._create_single_strategy_ledger(strategy_pf)
            return ledgers
        else:
            return self._create_single_strategy_ledger(self.pf)

    def _create_single_strategy_ledger(self, portfolio):
        df = portfolio.trades.records_readable
        
        # Handle portfolio.value which may have MultiIndex
        balance_df = portfolio.value.reset_index()
        
        # Flatten column names if MultiIndex
        if isinstance(balance_df.columns, pd.MultiIndex):
            balance_df.columns = ['_'.join(map(str, col)).strip('_') if isinstance(col, tuple) else col 
                                  for col in balance_df.columns]
        
        # Rename the value column to "balance"
        value_col = [col for col in balance_df.columns if col not in ['datetime', 'index']][0]
        balance_df = balance_df.rename(columns={value_col: "balance"})
        
        # Ensure datetime column exists
        if 'datetime' not in balance_df.columns and balance_df.columns[0] != 'datetime':
            balance_df = balance_df.rename(columns={balance_df.columns[0]: "datetime"})
        
        records = portfolio.orders.readable[['Fill Index', 'Stop Type']]

        ledger = (
            df
            .merge(balance_df, left_on="Exit Index", right_on="datetime", how="left")
            .merge(records, left_on="Exit Index", right_on="Fill Index", how="left")
        )

        # Compute balance at entry (before trade) — used by several columns below
        prev_balance = ledger["balance"] - ledger["PnL"]

        # position_size_pct: leveraged exposure as % of account balance at entry
        entry_pos_value = ledger["Avg Entry Price"] * ledger["Size"]
        ledger["position_size_pct"] = (entry_pos_value / prev_balance * 100).round(2)

        # Fees as % of position value
        exit_pos_value = ledger["Avg Exit Price"] * ledger["Size"]
        ledger["entry_fee_pct"] = (ledger["Entry Fees"] / entry_pos_value * 100).round(4)
        ledger["exit_fee_pct"] = (ledger["Exit Fees"] / exit_pos_value * 100).round(4)

        # trade_return_pct: % return on the capital invested in this trade
        # Use for: win rate, avg win/loss, profit factor, expectancy
        ledger["trade_return_pct"] = (ledger["Return"] * 100).round(4)

        # account_return_pct: actual % change to total account balance per trade
        # Use for: Sharpe, Sortino, Calmar, drawdown
        ledger["account_return_pct"] = (ledger["PnL"] / prev_balance * 100).round(4)

        # cum_account_return: running cumulative % growth of the account
        ledger["cum_account_return"] = (
            (1 + ledger["account_return_pct"] / 100).cumprod() - 1
        ).mul(100).round(4)

        ledger["balance"] = ledger["balance"].round(2)

        # Build action column: "{TP|SL} - {Mechanism}"
        # Mechanism is determined by which slot was populated during run():
        #   sl_stop  → self._sl_mechanism  (e.g. 'Static', 'ATR')
        #   tp_stop  → self._tp_mechanism  (e.g. 'Static', 'ATR', 'Chandelier')
        #   tsl_stop → self._tsl_mechanism (e.g. 'Trailing', 'Chandelier')
        #   td_stop  → always 'Time'
        #   no stop  → 'Direction Change'
        sl_mech = self._sl_mechanism or 'Static'
        tp_mech = self._tp_mechanism or 'Static'
        tsl_mech = self._tsl_mechanism or 'Trailing'

        def _get_mechanism(stop_type):
            if pd.isna(stop_type):
                return 'Direction Change'
            normalized = str(stop_type).strip()
            if normalized in ('SL', 'Stop Loss'):
                return sl_mech
            elif normalized in ('TP', 'Take Profit'):
                return tp_mech
            elif normalized in ('TSL', 'Trailing Stop'):
                return tsl_mech
            elif normalized in ('TD', 'Time Stop'):
                return 'Time'
            return normalized

        mechanism = ledger["Stop Type"].map(_get_mechanism)
        is_tp = ledger["Stop Type"].isin(['TP', 'Take Profit'])
        is_sl = ledger["Stop Type"].isin(['SL', 'Stop Loss'])
        prefix = np.where(is_tp, "TP",
                 np.where(is_sl, "SL",
                 np.where(ledger["trade_return_pct"] >= 0, "TP", "SL")))

        ledger["action"] = pd.Series(prefix, index=ledger.index) + " - " + mechanism

        column_map = {
            "Entry Index": "entry_datetime",
            "Avg Entry Price": "avg_entry_price",
            "Exit Index": "exit_datetime",
            "Avg Exit Price": "avg_exit_price",
            "Direction": "direction",
            "Status": "status",
            "balance": "balance",
        }

        final_cols = [
            "entry_datetime", "entry_fee_pct", "avg_entry_price",
            "exit_datetime", "exit_fee_pct", "avg_exit_price",
            "position_size_pct", "trade_return_pct", "account_return_pct", "cum_account_return",
            "direction", "status", "action", "balance",
        ]

        return ledger.rename(columns=column_map)[final_cols]
