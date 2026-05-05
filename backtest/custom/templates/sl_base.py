from abc import ABC, abstractmethod

class StopLossTemplate(ABC):
    def __init__(self, backtest):
        """
        Base template for stop loss and take profit logic.
        
        Args:
            backtest: The Backtest instance to access shared state and config.
        """
        self.backtest = backtest
        self.config = backtest.config

    @abstractmethod
    def check(self, np_temp):
        """
        Check if any stop loss or take profit conditions are met.
        
        Args:
            np_temp (np.array): OHLCV data for the current prediction interval.
            
        Returns:
            bool: True if a condition was met and a trade was closed.
        """
        pass

    def close_trade(self, index, sell_price, reason, np_temp, pnl=None):
        """
        Helper to close a trade used by templates.
        """
        if pnl is None:
            if self.backtest.previous_pred_direction > 0: # Long
                pnl = ((sell_price - self.backtest.buy_price) / self.backtest.buy_price) * 100
            else: # Short
                pnl = ((self.backtest.buy_price - sell_price) / self.backtest.buy_price) * 100
            
            pnl *= self.backtest.leverage
            pnl -= self.backtest.transaction_fee_percent
            pnl -= self.backtest.slippage

        self.backtest.current_balance += self.backtest.position_size * (pnl/100)
        self.backtest.in_position = False
        self.backtest.sell_price = sell_price
        self.backtest.close_price = np_temp[-1][self.backtest.index_ohlcv_close]
        
        self.backtest.record_trade(np_temp[index][self.backtest.index_ohlcv_datetime], 'sell' + reason, pnl)
        return True
