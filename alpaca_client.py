from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live.news import NewsDataStream
import config
import logging
from datetime import datetime, timedelta

class AlpacaClient:
    def __init__(self):
        self.trading_client = TradingClient(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, paper=True)
        self.data_client = StockHistoricalDataClient(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY)
        self.news_stream = NewsDataStream(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY)
        
    def get_account(self):
        return self.trading_client.get_account()

    def get_positions(self):
        try:
            return self.trading_client.get_all_positions()
        except Exception:
            return []

    def get_historical_bars(self, symbol, days_back=60):
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Hour, # Changed to Hourly bars for significantly more data
            start=datetime.now() - timedelta(days=days_back)
        )
        bars = self.data_client.get_stock_bars(request_params)
        return bars.df

    def get_current_price(self, symbol):
        # We can get the latest trade or just the most recent bar close
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=datetime.now() - timedelta(days=2) # go back 2 days in case of weekend
        )
        bars = self.data_client.get_stock_bars(request_params)
        if not bars.df.empty:
            return bars.df.iloc[-1]['close']
        return 0.0

    def submit_order(self, symbol, qty, side: OrderSide, market_closed=False):
        """
        Submit a fractional order.
        """
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        
        try:
            order = self.trading_client.submit_order(order_data=market_order_data)
            return order
        except Exception as e:
            logging.error(f"Failed to submit order for {symbol}: {e}")
            return None

    def get_open_orders(self):
        return self.trading_client.get_orders()

    def is_market_open(self):
        clock = self.trading_client.get_clock()
        return clock.is_open

    def start_news_stream(self, news_handler):
        self.news_stream.subscribe_news(news_handler, "*")
        logging.info("Starting Alpaca News WebSocket...")
        self.news_stream.run()
