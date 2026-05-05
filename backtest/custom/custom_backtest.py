import pandas as pd
import numpy as np
import warnings
import time
from datetime import timezone

from backtest.custom.templates.static_tp_sl import StaticTpSl
from backtest.custom.templates.trailing_tp_sl import TrailingTpSl

warnings.filterwarnings('ignore')

class Backtest():
	def __init__(self, df_ohlcv, df_signals, config):
		"""
		Initialize Backtest with data and configuration.

		Args:
			df_ohlcv (pd.DataFrame): 1-minute OHLCV data
			df_signals (pd.DataFrame): Prediction signals data
			config (dict): Backtest parameters
		"""
		# Extract configuration parameters with defaults
		self.starting_balance = config.get('starting_balance', 1000)
		take_profit = config.get('take_profit', 100) # Percentage
		stop_loss = config.get('stop_loss', 100) # Percentage
		self.buy_after_minutes = int(config.get('buy_after_minutes', 0))
		transaction_fee = config.get('transaction_fee', 0.05) # Percentage
		self.leverage = config.get('leverage', 1.0)
		self.slippage = config.get('slippage', 0.0)
		self.use_time_stop = config.get('use_time_stop', False)
		self.time_stop_minutes = config.get('time_stop_minutes', 30)
		self.record_same_direction = config.get('record_same_direction', False)
		self.allocation_percent = config.get('allocation_percent', 100) / 100  # 0.5 for 50%, 1.0 for 100%
		self.tp_sl_type = config.get('tp_sl_type', 'trailing')  # Options: 'static', 'trailing', 'donchian', 'chandelier', 'fibonacci'
		
		self.trailing_stop_loss_percent = config.get('trailing_stop_loss_percent', 2) / 100
		self.trailing_stop_activation_percent = config.get('trailing_stop_activation_percent', 0) / 100
		self.donchian_window = config.get('donchian_window', 20)
		self.chandelier_window = config.get('chandelier_window', 22)
		self.chandelier_multiplier = config.get('chandelier_multiplier', 4.0)
		self.fib_window = config.get('fib_window', 50)
		self.fib_sl_level = config.get('fib_sl_level', 0.618)
		
		# For database lookups
		self.exchange = config.get('exchange', 'binance')
		self.symbol = config.get('symbol', 'btc')
		self.timeframe = config.get('timeframe', '1h')
		
		# Ensure signal column is identified
		if 'signal' not in df_signals.columns:
			signal_cols = [c for c in df_signals.columns if c != 'datetime']
			if signal_cols:
				df_signals = df_signals.rename(columns={signal_cols[0]: 'signal'})

		# Normalize datetime in signals to naive UTC
		if df_signals['datetime'].dtype == 'int64' or df_signals['datetime'].dtype == 'float64':
			df_signals['datetime'] = pd.to_datetime(df_signals['datetime'], unit='s', utc=True)
		else:
			df_signals['datetime'] = pd.to_datetime(df_signals['datetime'], utc=True)
		
		# Standardize to naive for numpy comparison
		df_signals['datetime'] = df_signals['datetime'].dt.tz_localize(None)

		self.index_pred_datetime = df_signals.columns.get_loc("datetime")
		self.index_pred_direction = df_signals.columns.get_loc("signal")
		self.np_model_predctions = df_signals.to_numpy()

		# Initialize accounting
		self.pnl_percent_all = 0
		self.current_balance = self.starting_balance
		self.breaking_balance = self.current_balance * 0.5
		
		self.buy_price = 0
		self.sell_price = 0
		self.close_price = 0
		
		self.take_profit_percent = take_profit / 100
		self.stop_loss_percent = stop_loss / 100
		self.transaction_fee_percent = transaction_fee * self.leverage
		self.in_position = False
		self.current_sl_template = None
		self.config = config # store config for templates
		self.array_to_save = []						
		self.header_names = [
						'datetime',
						'signal',
						'action',
						'buy_price',
						'sell_price',
						'position_size',
						'portfolio',
						'pnl'
					]		

		# Process OHLCV data - Normalize to naive UTC
		if df_ohlcv['datetime'].dtype == 'int64' or df_ohlcv['datetime'].dtype == 'float64':
			df_ohlcv['datetime'] = pd.to_datetime(df_ohlcv['datetime'], unit='s', utc=True)
		elif not pd.api.types.is_datetime64_any_dtype(df_ohlcv['datetime']):
			df_ohlcv['datetime'] = pd.to_datetime(df_ohlcv['datetime'], utc=True)
		else:
			# Ensure it is UTC if already datetime
			df_ohlcv['datetime'] = pd.to_datetime(df_ohlcv['datetime'], utc=True)
			
		df_ohlcv['datetime'] = df_ohlcv['datetime'].dt.tz_localize(None)
		
		# Initialize accounting
		self.np_ohlcv = df_ohlcv.to_numpy()
		self.index_ohlcv_datetime = df_ohlcv.columns.get_loc("datetime")
		self.np_ohlcv_datetime = self.np_ohlcv[:, self.index_ohlcv_datetime]
		self.index_ohlcv_open = df_ohlcv.columns.get_loc("open")
		self.index_ohlcv_high = df_ohlcv.columns.get_loc("high")
		self.index_ohlcv_low = df_ohlcv.columns.get_loc("low")
		self.index_ohlcv_close = df_ohlcv.columns.get_loc("close")

	def buy(self, np_temp):
		self.buy_price = np_temp[self.buy_after_minutes][self.index_ohlcv_open]
		self.sell_price = 0
		self.position_size = self.current_balance * self.allocation_percent

		# Initialize SL/TP Template
		template_map = {
			'static': StaticTpSl,
			'trailing': TrailingTpSl
		}
		template_class = template_map.get(self.tp_sl_type, StaticTpSl)
		self.current_sl_template = template_class(self)


		pnl = self.transaction_fee_percent * -1
		pnl -= self.slippage
		self.current_balance += self.position_size * (pnl/100)
		self.in_position = True

		self.close_price = np_temp[-1][self.index_ohlcv_close]
		
		self.record_trade(np_temp[self.buy_after_minutes][self.index_ohlcv_datetime], 'buy', pnl)

	def pnl_direction_change(self, sell_datetime):
		if self.in_position:
			pnl = 0
			if self.previous_pred_direction > 0:
				pnl = ((self.sell_price - self.buy_price)/self.buy_price) * 100
				pnl *= self.leverage
				pnl = pnl - (self.transaction_fee_percent) 
			else:
				pnl = ((self.buy_price - self.sell_price)/self.buy_price) * 100
				pnl *= self.leverage
				pnl = pnl - (self.transaction_fee_percent) 

			pnl -= self.slippage
			self.current_balance += self.position_size * (pnl/100)
			self.in_position = False

			self.record_trade(sell_datetime, 'sell - direction change', pnl)
		
	def find_tp_sl_index(self, take_profit_amount, stop_loss_amount, np_temp):
		np_temp_high = np_temp[:, self.index_ohlcv_high]
		np_temp_low = np_temp[:, self.index_ohlcv_low]

		if self.current_pred_direction > 0:
			list_minute_high_indices = np.where(np_temp_high >= take_profit_amount)[0]
			list_minute_low_indices = np.where(np_temp_low <= stop_loss_amount)[0]
		else:
			list_minute_high_indices = np.where(np_temp_high >= stop_loss_amount)[0]
			list_minute_low_indices = np.where(np_temp_low <= take_profit_amount )[0]

		if len(list_minute_high_indices) == 0 and len(list_minute_low_indices) == 0:
			return False, -1
		elif len(list_minute_high_indices) > 0 and len(list_minute_low_indices) == 0:
			df_index = list_minute_high_indices[0]
			self.sell_price = np_temp_high[df_index]
			return True, df_index
		
		elif len(list_minute_high_indices) == 0 and len(list_minute_low_indices) > 0:
			df_index = list_minute_low_indices[0]
			self.sell_price = np_temp_low[df_index]
			return True, df_index
		else:
			if list_minute_high_indices[0] < list_minute_low_indices[0]:
				df_index = list_minute_high_indices[0]
				self.sell_price = np_temp_high[df_index]	
				return True, df_index
			else:
				df_index = list_minute_low_indices[0]
				self.sell_price = np_temp_low[df_index]
				return True, df_index
	
	def check_tp_sl(self, np_temp):
		if not self.in_position or self.current_sl_template is None:
			return

		self.current_sl_template.check(np_temp)


	def record_trade(self, datetime, action, pnl):
		self.array_to_save.append( 
								[ datetime, 
									'long' if self.current_pred_direction > 0 else 'short',
									action,  
									self.buy_price,
									self.sell_price,
									self.position_size,
									self.current_balance,
									pnl
								]
							)
		
	def get_interval_min_data(self, index):
		# get minutes data for the current prediction time using numpy
		start_time = np.datetime64(self.np_model_predctions[index][self.index_pred_datetime] )
		# end_time   = np.datetime64(self.np_model_predctions[index+1][self.index_pred_datetime] )
		if index + 1 < len(self.np_model_predctions):
			end_time = np.datetime64(self.np_model_predctions[index + 1][self.index_pred_datetime])
		else:
			# If next index does not exist, use the maximum datetime from the dataset
			end_time = np.max(self.np_ohlcv[:, self.index_ohlcv_datetime])

		# Search/Filter (Optimized with searchsorted)
		start_idx = np.searchsorted(self.np_ohlcv_datetime, start_time, side='left')
		end_idx = np.searchsorted(self.np_ohlcv_datetime, end_time, side='left')

		# Slicing
		np_temp = self.np_ohlcv[start_idx:end_idx]

		return np_temp

	def run(self, df_existing_ledger=None):					
		# first predicted direction
		self.previous_pred_direction = self.current_pred_direction = self.np_model_predctions[0][self.index_pred_direction] 
		break_on_huge_loss = False

		# if df_existing_ledger is not None:
		# 	if str(df_existing_ledger['action'].iloc[-1]) == 'buy' or str(df_existing_ledger['action'].iloc[-1]) == 'same direction':
		# 		self.in_position = True
		# 		self.buy_price = float(df_existing_ledger['buy_price'].iloc[-1])
		# 		self.current_balance = float(df_existing_ledger['balance'].iloc[-1])
		# 		self.previous_pred_direction = -1 if str(df_existing_ledger['signal'].iloc[-1])=='short' else 1
		# 	else:
		# 		self.current_balance = float(df_existing_ledger['balance'].iloc[-1])
		# 		# self.previous_pred_direction = -1 if str(df_existing_ledger['signal'].iloc[-1])=='short' else 1

		
		
		for i in range (0, len(self.np_model_predctions)):
			self.current_i = i # used by templates for lookback data
			self.current_pred_direction = self.np_model_predctions[i][self.index_pred_direction] 

			if self.current_pred_direction == 0:
				if self.previous_pred_direction == 0:
					self.previous_pred_direction = self.current_pred_direction
					continue
				self.current_pred_direction = self.previous_pred_direction

			## get current interval's minute level data
			np_temp = self.get_interval_min_data(i)

			if self.use_time_stop and i != len(self.np_model_predctions)-1: 
				np_temp = np_temp[:-self.time_stop_minutes]

			## if in position, and the new direction is same as the previous direction
			if self.in_position:
				if self.previous_pred_direction == self.current_pred_direction:
					self.previous_pred_direction = self.current_pred_direction
					self.close_price = np_temp[-1][self.index_ohlcv_close]
					if self.record_same_direction:
						current_timestamp = int(np_temp[self.buy_after_minutes][self.index_ohlcv_datetime].replace(tzinfo=timezone.utc).timestamp())
						if df_existing_ledger is not None:
							if not df_existing_ledger['datetime'].iloc[-1] == current_timestamp:
								self.record_trade(np_temp[self.buy_after_minutes][self.index_ohlcv_datetime], 'same direction', 0)
						else:
							self.record_trade(np_temp[self.buy_after_minutes][self.index_ohlcv_datetime], 'same direction', 0)

			### if not in position then buy
			if not self.in_position: 
				self.buy(np_temp)
				self.previous_pred_direction = self.current_pred_direction
				
			### sell -> change in direction
			if self.current_pred_direction != self.previous_pred_direction: 
				self.sell_price = np_temp[self.buy_after_minutes][self.index_ohlcv_open]
				sell_datetime = np_temp[self.buy_after_minutes][self.index_ohlcv_datetime]

				# self.close_price = np_temp[-1][self.index_ohlcv_close]
				self.close_price = self.sell_price

				self.pnl_direction_change(sell_datetime)
				self.previous_pred_direction = self.current_pred_direction

				### buy again after direction change
				if not self.in_position: #buy
					self.buy(np_temp)
					self.previous_pred_direction = self.current_pred_direction

			### check if during the time horizon it hits take profit or stop loss
			self.check_tp_sl(np_temp) 
			
			# if self.use_time_stop:
			if self.use_time_stop and i != len(self.np_model_predctions)-1: 
				if self.in_position: 
					df_temp_selling_index = -1
					self.sell_price = np_temp[df_temp_selling_index][self.index_ohlcv_open]
					sell_datetime = np_temp[df_temp_selling_index][self.index_ohlcv_datetime]
					# self.close_price = np_temp[-1][self.index_ohlcv_close]
					self.close_price = self.sell_price

					self.pnl_direction_change(sell_datetime)

			self.previous_pred_direction = self.current_pred_direction

			if self.current_balance < self.breaking_balance:
				break_on_huge_loss = True
				break
		
		# print(round(time.time() - tik, 2))
		### backtest dataframe, and some required fields
		df_ledger = pd.DataFrame(self.array_to_save, columns = self.header_names)
		if df_ledger.empty:
			return df_ledger, np_temp[-1][self.index_ohlcv_close], 0

		if (len(df_ledger) > 0) and (df_existing_ledger is not None):
			# df_ledger
			start_id = df_existing_ledger['id'].max() + 1
			df_ledger['id'] = range(start_id, start_id + len(df_ledger))
			df_ledger['datetime'] = df_ledger['datetime'].apply(lambda x: int(x.timestamp()))
			df_ledger = pd.concat([df_existing_ledger, df_ledger])
			# print(df_ledger)
			# df_existing_ledger["pnl_sum"] = df_existing_ledger["pnl"].cumsum()

		df_ledger["pnl_sum"] = df_ledger["pnl"].cumsum()
		df_ledger['return'] = (df_ledger['portfolio'].pct_change())
		df_ledger['return'].fillna(0, inplace=True)
		df_ledger['cumulative_return'] = ((1 + df_ledger['return']).cumprod() - 1) 
		df_ledger['return'] = df_ledger['return'] * 100
		df_ledger['cumulative_return'] = df_ledger['cumulative_return'] * 100

		# Calculate the drawdown percentage
		cumulative_max = df_ledger['portfolio'].cummax()
		df_ledger['drawdown'] = ((df_ledger['portfolio'] - cumulative_max) / cumulative_max) * 100   
		# rounding_dict = {'balance': 2, 'pnl': 2, 'pnl_sum': 2, 'drawdown': 2}
		rounding_dict = {'portfolio': 2, 'pnl': 2, 'pnl_sum': 2, 'drawdown': 2, 'return': 2, 'cumulative_return': 2}

		# Round the specified columns
		df_ledger = df_ledger.round(rounding_dict)

		### pnl percent
		pnl_percent = np.round(df_ledger["pnl_sum"].iloc[-1], 2)

		if break_on_huge_loss:
			return df_ledger, -1000, pnl_percent
		else:
			return df_ledger, round(self.current_balance, 2), round(pnl_percent, 2)

	