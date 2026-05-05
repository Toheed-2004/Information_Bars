import numpy as np
from .sl_base import StopLossTemplate

class TrailingTpSl(StopLossTemplate):
    def __init__(self, backtest):
        super().__init__(backtest)
        self.trailing_stop_loss_percent = self.backtest.trailing_stop_loss_percent
        self.trailing_activation_percent = self.backtest.trailing_stop_activation_percent
        
        # Initialize stop price to initial stop loss level (from backtest.stop_loss_percent)
        if self.backtest.current_pred_direction > 0: # Long
            self.trailing_stop_loss_price = self.backtest.buy_price * (1 - self.backtest.stop_loss_percent)
        else: # Short
            self.trailing_stop_loss_price = self.backtest.buy_price * (1 + self.backtest.stop_loss_percent)
            
        self.trailing_activated = True if self.trailing_activation_percent <= 0 else False

    def check(self, np_temp):
        if not self.backtest.in_position:
            return False

        # Calculate static take profit amount once for the interval
        if self.backtest.current_pred_direction > 0:
            tp_price = self.backtest.buy_price * (1 + self.backtest.take_profit_percent)
        else:
            tp_price = self.backtest.buy_price * (1 - self.backtest.take_profit_percent)
        
        # Iterate minute by minute
        for i in range(len(np_temp)):
            high = np_temp[i][self.backtest.index_ohlcv_high]
            low  = np_temp[i][self.backtest.index_ohlcv_low]

            if self.backtest.current_pred_direction > 0:  # LONG
                # CHECK STOP & TP FIRST (No look-ahead bias)
                # Use the stop level that was valid at the START of this minute
                if low <= self.trailing_stop_loss_price:
                    return self.close_trade(i, self.trailing_stop_loss_price, " - trailing", np_temp)

                if high >= tp_price:
                    return self.close_trade(i, tp_price, " - take_profit", np_temp)

                # UPDATE TRAIL (for the next minute)
                if not self.trailing_activated:
                    if high >= self.backtest.buy_price * (1 + self.trailing_activation_percent):
                        self.trailing_activated = True
                
                if self.trailing_activated:
                    new_stop = high * (1 - self.trailing_stop_loss_percent)
                    self.trailing_stop_loss_price = max(self.trailing_stop_loss_price, new_stop)

            else:  # SHORT
                # CHECK STOP & TP FIRST
                if high >= self.trailing_stop_loss_price:
                    return self.close_trade(i, self.trailing_stop_loss_price, " - trailing", np_temp)

                if low <= tp_price:
                    return self.close_trade(i, tp_price, " - take_profit", np_temp)

                # 2UPDATE TRAIL
                if not self.trailing_activated:
                    if low <= self.backtest.buy_price * (1 - self.trailing_activation_percent):
                        self.trailing_activated = True
                
                if self.trailing_activated:
                    new_stop = low * (1 + self.trailing_stop_loss_percent)
                    self.trailing_stop_loss_price = min(self.trailing_stop_loss_price, new_stop)

        return False
