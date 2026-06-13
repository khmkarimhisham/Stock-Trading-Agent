# Ultimate AI Stock Trading Bot 🚀

A state-of-the-art Hybrid AI algorithmic trading program built in Python. This bot merges hard quantitative math (PyTorch LSTM) with advanced qualitative reasoning (Local LLM via Ollama) to trade stocks autonomously.

It operates on real-time event-driven data using Alpaca's WebSockets, manages strict capital limits with fractional shares, queues off-hours trades automatically, and features a beautiful real-time terminal dashboard.

## 🌟 Key Features

1. **Hybrid AI Engine**: Combines technical price predictions (LSTM) with real-time news sentiment and logical reasoning (Llama 3).
2. **Local & Private LLM**: Uses [Ollama](https://ollama.com/) to run Llama 3 locally on your machine—zero API costs for language models and complete privacy.
3. **Fractional Share Execution**: You set a strict dollar budget per stock (e.g., $100). The Risk Manager calculates exact fractions (e.g., 0.45 shares) to maximize capital efficiency without exceeding your limit.
4. **Market-On-Open Queueing**: If breaking news happens on a weekend or after hours, the bot analyzes it and automatically queues an order for the second the opening bell rings.
5. **Real-Time Terminal Dashboard**: Built with `rich`, providing a clean, emoji-free live view of your portfolio, active positions, and the AI's real-time "thought process."

## 📂 Project Structure

- `config.py` - Core configuration, budget limits, symbols, and API keys.
- `trader.py` - The main execution loop and WebSocket listener. Run this to start the bot!
- `alpaca_client.py` - Broker integration for fetching market data, checking positions, and executing fractional/queued trades.
- `math_model.py` - The Quantitative Engine. A PyTorch LSTM model with Technical Analysis (TA) indicators.
- `train.py` - The model training script. Use this to optimize the LSTM on historical data before live trading.
- `llm_agent.py` - The Qualitative Engine. Interfaces with your local Ollama server to reason about math signals and breaking news.
- `risk_manager.py` - The safety net that enforces budget limits and calculates exact fractional trade sizes.
- `dashboard.py` - The live terminal UI generator.

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10+
- An [Alpaca](https://alpaca.markets/) account (Paper Trading recommended).
- [Ollama](https://ollama.com/) installed on your machine.

### 2. Pull the Local LLM
Ensure Ollama is running, then pull the Llama 3 model in your terminal:
```bash
ollama run llama3
```
*(You can close the chat interface it opens; the model just needs to be downloaded and available).*

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create your environment variables file:
```bash
cp .env.example .env
```
Open `.env` and paste in your Alpaca Paper Trading `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`.

*(Optional)* Open `config.py` to adjust your max budget per symbol, the stocks you want to trade, or your local Ollama API url.

## 🚀 Running the Bot

Start the main orchestrator script:
```bash
python trader.py
```

The bot will initialize, connect to the Alpaca news WebSocket, start the background hourly technical analysis loop, and render the live dashboard in your terminal. 

## ⚠️ Disclaimer
This software is for educational purposes. Always test algorithms heavily in Paper Trading environments before risking real capital. The included PyTorch LSTM is a structural placeholder and should be formally trained on historical data for optimal quantitative performance.
