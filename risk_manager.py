import config
import logging
from alpaca.trading.enums import OrderSide

class RiskManager:
    def __init__(self, alpaca_client):
        self.alpaca = alpaca_client
        self.max_budget = config.MAX_BUDGET_PER_SYMBOL_USD

    def get_current_exposure(self, symbol):
        positions = self.alpaca.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return float(pos.market_value), float(pos.qty)
        return 0.0, 0.0

    def evaluate_and_size_trade(self, symbol, current_price, action):
        """
        Takes an AI action (BUY/SELL/HOLD), checks exposure, 
        and calculates exact fractional shares if BUY.
        Returns (approved_action, qty, reason)
        """
        if action == "HOLD":
            return "HOLD", 0.0, "AI chose HOLD"

        current_exposure, current_qty = self.get_current_exposure(symbol)

        if action == "SELL":
            if current_qty > 0:
                # Sell everything we have of this symbol
                return "SELL", current_qty, "Approved"
            else:
                return "HOLD", 0.0, "Nothing to sell"

        if action == "BUY":
            remaining_budget = self.max_budget - current_exposure
            
            if remaining_budget <= 1.0: # Prevent tiny <$1 buys
                msg = f"Max budget reached (${self.max_budget})"
                logging.warning(f"RiskManager: Rejecting BUY for {symbol}. {msg}")
                return "HOLD", 0.0, msg
            
            if current_price <= 0:
                return "HOLD", 0.0, "Invalid price"

            # Calculate exact fractional shares to buy based on remaining budget
            qty_to_buy = remaining_budget / current_price
            
            # Round to 4 decimal places (Alpaca fractional support precision)
            qty_to_buy = round(qty_to_buy, 4)
            
            if qty_to_buy < 0.0001:
                return "HOLD", 0.0, "Calculated qty too small"

            return "BUY", qty_to_buy, "Approved"

        return "HOLD", 0.0, "Default fallback"
