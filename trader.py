import threading
import time
import logging
from alpaca.trading.enums import OrderSide

logging.basicConfig(
    filename='bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
import config
from alpaca_client import AlpacaClient
from math_model import QuantEngine
from llm_agent import LLMAgent
from risk_manager import RiskManager
from dashboard import DashboardApp
from notifier import send_discord_alert
from order_queue import add_to_queue, process_queued_orders



def main():
    alpaca = AlpacaClient()
    quant = QuantEngine()
    llm = LLMAgent()
    risk = RiskManager(alpaca)
    
    # Initialize Textual App
    dash = DashboardApp(alpaca)

    def dash_log(msg):
        dash.log_message(msg)

    dash_log("System initialized. Starting Hybrid AI Trading Bot...")

    def process_trading_signal(symbol, news_headline=None):
        dash_log(f"Analyzing {symbol}...")
        
        # 1. Math Model
        try:
            historical_data = alpaca.get_historical_bars(symbol)
            math_prediction = quant.predict(historical_data)
            dash_log(f"[{symbol}] LSTM Prediction: {math_prediction:.4f}%")
        except Exception as e:
            error_msg = f"[{symbol}] Math model failed: {e}"
            dash_log(error_msg)
            send_discord_alert(f"⚠️ **CRITICAL ERROR:** {error_msg}")
            return

        # 2. LLM Reasoning
        current_price = alpaca.get_current_price(symbol)
        if current_price == 0:
            dash_log(f"[{symbol}] Could not fetch valid price. Skipping.")
            return

        llm_decision = llm.analyze(symbol, current_price, math_prediction, news_headline)
        action = llm_decision['action']
        dash_log(f"[{symbol}] LLM: {action} ({llm_decision['reasoning']})")

        # 3. Risk Management & Execution
        approved_action, qty, reason = risk.evaluate_and_size_trade(symbol, current_price, action)
        
        if approved_action == "BUY":
            dash_log(f"[{symbol}] Risk Manager approved BUY for {qty} shares.")
            side = OrderSide.BUY
        elif approved_action == "SELL":
            dash_log(f"[{symbol}] Risk Manager approved SELL for {qty} shares.")
            side = OrderSide.SELL
        else:
            if action != approved_action:
                dash_log(f"[bold red][{symbol}] Risk Manager OVERRIDE -> HOLD: {reason}[/bold red]")
            else:
                dash_log(f"[{symbol}] Holding position.")
            return

        is_market_closed = not alpaca.is_market_open()
        if is_market_closed:
            dash_log(f"Market is closed. Queueing {symbol} for re-evaluation at market open.")
            add_to_queue(symbol)
            return

        order = alpaca.submit_order(symbol, qty, side, market_closed=False)
        if order:
            msg = f"Order submitted successfully: {order.id} | {approved_action} {qty} {symbol}"
            dash_log(msg)
            send_discord_alert(f"✅ **TRADE EXECUTED:** {msg}")

    # Async News Handler
    async def news_handler(data):
        symbols = data.symbols
        headline = data.headline
        for sym in symbols:
            if sym in config.SYMBOLS:
                dash_log(f"BREAKING NEWS for {sym}: {headline}")
                # Offload to avoid blocking websocket stream
                threading.Thread(target=process_trading_signal, args=(sym, headline), daemon=True).start()

    # Start WebSocket in background thread
    def start_ws():
        try:
            alpaca.start_news_stream(news_handler)
        except Exception as e:
            dash_log(f"WebSocket Error: {e}")
            send_discord_alert(f"⚠️ **WEBSOCKET ERROR:** {e}")

    ws_thread = threading.Thread(target=start_ws, daemon=True)
    ws_thread.start()

    # Periodic Technical Analysis (when there's no news)
    def periodic_loop():
        # wait a few seconds before starting to let UI render
        time.sleep(2)
        while True:
            dash_log("Running periodic technical analysis...")
            for symbol in config.SYMBOLS:
                process_trading_signal(symbol, news_headline=None)
                time.sleep(3) # Avoid rate limits
            time.sleep(3600) # Run every hour

    periodic_thread = threading.Thread(target=periodic_loop, daemon=True)
    periodic_thread.start()

    # Queue processing loop (runs every 5 minutes)
    def queue_loop():
        time.sleep(5) # Let UI initialize
        while True:
            try:
                if alpaca.is_market_open():
                    process_queued_orders(alpaca, dash_log)
            except Exception as e:
                dash_log(f"Error in queue loop: {e}")
            time.sleep(300) # Check every 5 minutes

    queue_thread = threading.Thread(target=queue_loop, daemon=True)
    queue_thread.start()

    # Main thread runs the Textual App UI
    try:
        dash.run()
    except KeyboardInterrupt:
        print("\nExiting Ultimate AI Trading Bot...")

if __name__ == "__main__":
    main()
