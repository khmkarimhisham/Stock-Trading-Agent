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
from model import QuantEngine
from risk_manager import RiskManager
from dashboard import DashboardApp
from notifier import send_discord_alert


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
        dash_log("-------------------------------------------------------------------")
        dash_log(f"Analyzing {symbol}...")
        
        # 1. Math Model
        try:
            historical_data = alpaca.get_historical_bars(symbol, days_back=30)
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
            dash_log(f"Market is closed.")
            return

        order = alpaca.submit_order(symbol, qty, side, current_price, market_closed=False)
        if order:
            msg = f"Order submitted successfully: {order.id} | {approved_action} {qty} {symbol}"
            dash_log(msg)
            send_discord_alert(f"✅ **TRADE EXECUTED:** {msg}")

    # (News stream removed for pure HFT LSTM approach)

    # High-Frequency Periodic Loop
    def tp_sl_loop():
        # wait a few seconds before starting to let UI render
        time.sleep(2)
        while True:
            # Check positions for TP/SL
            try:
                positions = alpaca.get_positions()
                for pos in positions:
                    pl_pct = float(pos.unrealized_plpc)
                    
                    action_msg = None
                    if pl_pct >= config.TAKE_PROFIT_PCT:
                        action_msg = f"Take Profit triggered (+{pl_pct*100:.2f}%)"
                    elif pl_pct <= -config.STOP_LOSS_PCT:
                        action_msg = f"Stop Loss triggered ({pl_pct*100:.2f}%)"
                        
                    if action_msg:
                        dash_log(f"[{pos.symbol}] {action_msg}. Closing position!")
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
                        msg = f"Automatically sold {pos.symbol} - {action_msg}!"
                        dash_log(msg)
                        send_discord_alert(f"✅ **POSITION CLOSED:** {msg}")
            except Exception as e:
                dash_log(f"Error checking positions for TP/SL: {e}")

            time.sleep(5) # Run TP/SL every 5 seconds

    def analysis_loop():
        time.sleep(2)
        while True:
            for symbol in config.SYMBOLS:
                process_trading_signal(symbol)
                time.sleep(3) # Avoid rate limits
            time.sleep(60) # Run every 1 minute

    tp_sl_thread = threading.Thread(target=tp_sl_loop, daemon=True)
    tp_sl_thread.start()

    analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
    analysis_thread.start()

    # Main thread runs the Textual App UI
    try:
        dash.run()
    except KeyboardInterrupt:
        print("\nExiting Ultimate AI Trading Bot...")

if __name__ == "__main__":
    main()
