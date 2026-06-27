# Ultimate AI Stock Trading Bot 🚀

A state-of-the-art Quantitative AI algorithmic trading program built in Python. This bot uses advanced quantitative math (PyTorch TCN) to trade stocks autonomously.

It operates on real-time data using Alpaca's REST API, manages strict capital limits with fractional shares, queues off-hours trades automatically, and features a beautiful real-time terminal dashboard.

## 🌟 Key Features

1. **Quantitative AI Engine**: Utilizes technical price predictions through a Stable Temporal Convolutional Network (TCN).
2. **Fractional Share Execution**: You set a strict dollar budget per stock (e.g., $100). The Risk Manager calculates exact fractions (e.g., 0.45 shares) to maximize capital efficiency without exceeding your limit.
3. **Market-On-Open Queueing**: If trades trigger on a weekend or after hours, the bot analyzes it and automatically queues an order for the second the opening bell rings.
4. **Real-Time Terminal Dashboard**: Built with `rich`, providing a clean, emoji-free live view of your portfolio, active positions, and the AI's real-time "thought process."

## 📂 Project Structure

- `config.py` - Core configuration, budget limits, symbols, and API keys.
- `trader.py` - The main execution loop. Run this to start the bot!
- `alpaca_client.py` - Broker integration for fetching market data, checking positions, and executing fractional/queued trades.
- `model.py` - The Quantitative Engine. A PyTorch TCN model with 20 Technical Analysis (TA) indicators.
- `train.py` - The model training script. Use this to optimize the TCN on historical data before live trading.
- `risk_manager.py` - The safety net that enforces budget limits and calculates exact fractional trade sizes.
- `dashboard.py` - The live terminal UI generator.

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10+
- An [Alpaca](https://alpaca.markets/) account (Paper Trading recommended).

### 2. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create your environment variables file:
```bash
cp .env.example .env
```
Open `.env` and paste in your Alpaca Paper Trading `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`.

*(Optional)* Open `config.py` to adjust your max budget per symbol or the stocks you want to trade.

## 🚀 Running the Bot

Start the main orchestrator script:
```bash
python trader.py
```

The bot will initialize, start the background hourly technical analysis loop, and render the live dashboard in your terminal. 

## ⚠️ Disclaimer
This software is for educational purposes. Always test algorithms heavily in Paper Trading environments before risking real capital. The included PyTorch TCN is a structural placeholder and should be formally trained on historical data for optimal quantitative performance.
