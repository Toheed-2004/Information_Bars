import numpy as np
from .sl_base import StopLossTemplate

class StaticTpSl(StopLossTemplate):
    def check(self, np_temp):
        if not self.backtest.in_position:
            return False

        tp_percent = self.backtest.take_profit_percent
        sl_percent = self.backtest.stop_loss_percent
        buy_price = self.backtest.buy_price
        
        if self.backtest.current_pred_direction > 0: # Long
            tp_price = buy_price * (1 + tp_percent)
            sl_price = buy_price * (1 - sl_percent)
        else: # Short
            tp_price = buy_price * (1 - tp_percent)
            sl_price = buy_price * (1 + sl_percent)

        condition_met, index = self.backtest.find_tp_sl_index(tp_price, sl_price, np_temp)
        
        if condition_met:
            sell_price = self.backtest.sell_price
            
            # Recalculate PNL specifically for static TP/SL to ensure it hits the exact target if within minute
            if self.backtest.previous_pred_direction > 0:
                pnl = (tp_percent * 100) if sell_price >= tp_price else (-sl_percent * 100)
            else:
                pnl = (tp_percent * 100) if sell_price <= tp_price else (-sl_percent * 100)
            
            pnl *= self.backtest.leverage
            pnl -= self.backtest.transaction_fee_percent
            pnl -= self.backtest.slippage
            
            reason = ' - take_profit' if pnl > 0 else ' - stop_loss'
            return self.close_trade(index, sell_price, reason, np_temp, pnl=pnl)
            
        return False
