import config
import logging
from alpaca.trading.enums import OrderSide

class RiskManager:
    def __init__(self, alpaca_client):
        self.alpaca = alpaca_client
        self.total_budget = config.TOTAL_TRADING_BUDGET_USD * config.FINANCIAL_LEVERAGE

    def get_exposures(self, symbol):
        positions = self.alpaca.get_positions()
        total_exposure = 0.0
        symbol_qty = 0.0
        symbol_exposure = 0.0
        symbol_plpc = 0.0
        for pos in positions:
            total_exposure += float(pos.market_value)
            if pos.symbol == symbol:
                symbol_qty = float(pos.qty)
                symbol_exposure = float(pos.market_value)
                symbol_plpc = float(pos.unrealized_plpc)
        return total_exposure, symbol_qty, symbol_exposure, symbol_plpc

    def evaluate_and_size_trade(self, symbol, current_price, action):
        """
        Takes an AI action (BUY/SELL/HOLD), checks exposure, 
        and calculates exact fractional shares if BUY.
        Returns (approved_action, qty, reason)
        """
        if action == "HOLD":
            return "HOLD", 0.0, "AI chose HOLD"

        total_exposure, current_qty, symbol_exposure, symbol_plpc = self.get_exposures(symbol)

        if action == "SELL":
            if current_qty > 0:
                if symbol_plpc < 0:
                    msg = "Position is at a loss. Waiting for TP/SL."
                    return "HOLD", 0.0, msg
                # Sell everything we have of this symbol
                return "SELL", current_qty, "Approved"
            else:
                return "HOLD", 0.0, "Nothing to sell"

        if action == "BUY":
            budget_per_symbol = self.total_budget / len(config.SYMBOLS)
            remaining_symbol_budget = budget_per_symbol - symbol_exposure
            
            if remaining_symbol_budget <= 1.0: # Prevent tiny <$1 buys
                msg = f"Symbol budget reached (${budget_per_symbol:.2f})"
                return "HOLD", 0.0, msg
            
            if current_price <= 0:
                return "HOLD", 0.0, "Invalid price"

            # Calculate exact fractional shares to buy based on remaining symbol budget
            qty_to_buy = remaining_symbol_budget / current_price
            
            # Round to 4 decimal places (Alpaca fractional support precision)
            qty_to_buy = round(qty_to_buy, 4)
            
            if qty_to_buy < 0.0001:
                return "HOLD", 0.0, "Calculated qty too small"

            return "BUY", qty_to_buy, "Approved"

        return "HOLD", 0.0, "Default fallback"
