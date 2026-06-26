import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca Credentials
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL")

# Trading Settings
SYMBOLS = ["AAPL","MSFT", "AMZN","TSLA","NVDA","SPY","GOOGL"]

# Risk Management & Budgeting
TOTAL_TRADING_BUDGET_USD = 100.0
FINANCIAL_LEVERAGE = 10.0

STOP_LOSS_PCT = 0.01
TAKE_PROFIT_PCT = 0.001

PROBABILITY_THRESHOLD = 80.0

# Discord Webhook URL (Get this from your Discord Server Settings -> Integrations -> Webhooks)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
