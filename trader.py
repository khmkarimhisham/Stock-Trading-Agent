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
from risk_manager import RiskManager
from dashboard import DashboardApp
from notifier import send_discord_alert
from order_queue import add_to_queue, process_queued_orders



def main():
    alpaca = AlpacaClient()
    quant = QuantEngine()
    risk = RiskManager(alpaca)
    
    # Initialize Textual App
    dash = DashboardApp(alpaca)

    def dash_log(msg):
        dash.log_message(msg)

    dash_log("System initialized. Starting Hybrid AI Trading Bot...")

    def process_trading_signal(symbol):
        dash_log(f"Analyzing {symbol}...")
        
        # 1. Math Model
        try:
            historical_data = alpaca.get_historical_bars(symbol)
            math_prediction = quant.predict(historical_data)
            dash_log(f"[{symbol}] LSTM Prediction: UP: {math_prediction['UP']:.1f}% | DOWN: {math_prediction['DOWN']:.1f}%")
        except Exception as e:
            error_msg = f"[{symbol}] Math model failed: {e}"
            dash_log(error_msg)
            send_discord_alert(f"⚠️ **CRITICAL ERROR:** {error_msg}")
            return

        # 2. LSTM Decision Logic
        current_price = alpaca.get_current_price(symbol)
        if current_price == 0:
            dash_log(f"[{symbol}] Could not fetch valid price. Skipping.")
            return

        up_prob = math_prediction['UP']
        down_prob = math_prediction['DOWN']
        
        if up_prob > 60.0:
            action = "BUY"
            reasoning = f"UP probability ({up_prob:.1f}%) > 60%"
        elif down_prob > 60.0:
            action = "SELL"
            reasoning = f"DOWN probability ({down_prob:.1f}%) > 60%"
        else:
            action = "HOLD"
            reasoning = "Probabilities below 60% threshold"
            
        dash_log(f"[{symbol}] LSTM Action: {action} ({reasoning})")

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

        order = alpaca.submit_order(symbol, qty, side, current_price, market_closed=False)
        if order:
            msg = f"Order submitted successfully: {order.id} | {approved_action} {qty} {symbol}"
            dash_log(msg)
            send_discord_alert(f"✅ **TRADE EXECUTED:** {msg}")

    # (News stream removed for pure HFT LSTM approach)

    # High-Frequency Periodic Loop (every 1 minute)
    def periodic_loop():
        # wait a few seconds before starting to let UI render
        time.sleep(2)
        while True:
            dash_log("Checking for profitable positions to take profit...")
            try:
                positions = alpaca.get_positions()
                for pos in positions:
                    if float(pos.unrealized_pl) > 0:
                        dash_log(f"[{pos.symbol}] Profitable position found. Taking profit!")
                        from alpaca.trading.requests import GetOrdersRequest
                        from alpaca.trading.enums import QueryOrderStatus, OrderType, OrderSide
                        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[pos.symbol])
                        open_orders = alpaca.trading_client.get_orders(req)
                        
                        is_closing = False
                        for order in open_orders:
                            if order.side == OrderSide.SELL and order.type == OrderType.MARKET:
                                is_closing = True
                                break
                                
                        if is_closing:
                            dash_log(f"[{pos.symbol}] Close order already pending. Waiting for fill...")
                            continue
                            
                        for order in open_orders:
                            alpaca.trading_client.cancel_order_by_id(order.id)
                        alpaca.trading_client.close_position(pos.symbol)
                        msg = f"Automatically sold {pos.symbol} for profit!"
                        dash_log(msg)
                        send_discord_alert(f"✅ **PROFIT TAKEN:** {msg}")
            except Exception as e:
                dash_log(f"Error checking positions for profit: {e}")

            dash_log("Running 1-minute high-frequency analysis...")
            for symbol in config.SYMBOLS:
                process_trading_signal(symbol)
                time.sleep(3) # Avoid rate limits
            time.sleep(60) # Run every 1 minute

    periodic_thread = threading.Thread(target=periodic_loop, daemon=True)
    periodic_thread.start()

    # Queue processing loop (runs every 5 minutes)
    def queue_loop():
        time.sleep(5) # Let UI initialize
        while True:
            try:
                if alpaca.is_market_open():
                    process_queued_orders(alpaca, dash_log, process_trading_signal)
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
