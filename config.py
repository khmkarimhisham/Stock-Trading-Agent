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
# The maximum dollar amount you are willing to allocate across ALL active trades combined.
TOTAL_TRADING_BUDGET_USD = 100.0
FINANCIAL_LEVERAGE = 10.0

# Stop loss and take profit percentages (e.g., 0.05 = 5%)
STOP_LOSS_PCT = 0.10
TAKE_PROFIT_PCT = 0.05

# AI Models Configuration
# (LLM Removed for pure LSTM)


# Discord Webhook URL (Get this from your Discord Server Settings -> Integrations -> Webhooks)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Order Queue configuration
# Expiration time for locally queued orders (e.g., 18 hours = 64800 seconds)
QUEUED_ORDER_EXPIRATION_SECONDS = 64800

