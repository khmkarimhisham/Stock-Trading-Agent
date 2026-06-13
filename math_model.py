import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import ta
import os
import logging

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3) # Predict 3 classes (DOWN, FLAT, UP)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class QuantEngine:
    def __init__(self):
        # We now use 12 advanced features
        self.feature_cols = ['close', 'volume', 'rsi', 'macd', 'macd_diff', 
                             'stoch', 'stoch_signal', 'bb_upper', 'bb_lower', 
                             'bb_width', 'atr', 'obv']
        self.model = SimpleLSTM(input_size=len(self.feature_cols))

        # Load pre-trained weights if they exist
        if os.path.exists("lstm_model.pth"):
            try:
                self.model.load_state_dict(torch.load("lstm_model.pth"))
                logging.info("QuantEngine: Loaded trained weights from lstm_model.pth")
            except Exception as e:
                logging.warning(f"QuantEngine: Could not load old weights (architecture changed). Using random weights.")
        else:
            logging.warning("QuantEngine: No lstm_model.pth found. Using randomly initialized weights.")
        
        self.model.eval()

    def engineer_features(self, df):
        if df is None or len(df) < 30:
            return None # Not enough data
            
        df = df.copy()
        # 1. Momentum Indicators
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_diff'] = macd.macd_diff()
        
        stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch'] = stoch.stoch()
        df['stoch_signal'] = stoch.stoch_signal()
        
        # 2. Volatility Indicators
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_pband()
        
        df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        
        # 3. Volume Indicators
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        
        # Drop NaNs created by rolling windows
        df.dropna(inplace=True)
        return df

    def normalize_sequence(self, seq):
        norm_seq = seq.copy()
        last_close = seq[-1, 0]
        if last_close == 0:
            last_close = 1e-8
            
        # close, macd, macd_diff, bb_upper, bb_lower, atr
        price_cols = [0, 3, 4, 7, 8, 10]
        norm_seq[:, price_cols] = seq[:, price_cols] / last_close
        
        # volume
        mean_vol = np.mean(seq[:, 1])
        if mean_vol == 0:
            mean_vol = 1e-8
        norm_seq[:, 1] = seq[:, 1] / mean_vol
        
        # obv
        mean_obv = np.mean(np.abs(seq[:, 11]))
        if mean_obv == 0:
            mean_obv = 1e-8
        norm_seq[:, 11] = seq[:, 11] / mean_obv
        
        # rsi, stoch, stoch_signal
        bounded_cols = [2, 5, 6]
        norm_seq[:, bounded_cols] = (seq[:, bounded_cols] - 50.0) / 50.0
        
        return norm_seq

    def predict(self, df):
        """
        Returns a dictionary of probabilities for DOWN, FLAT, UP
        """
        features_df = self.engineer_features(df)
        if features_df is None or features_df.empty:
            return {"DOWN": 0.0, "FLAT": 100.0, "UP": 0.0}

        # Take the last 10 steps as the sequence for prediction
        seq_length = 10
        if len(features_df) < seq_length:
            return {"DOWN": 0.0, "FLAT": 100.0, "UP": 0.0}

        recent_data = features_df.iloc[-seq_length:][self.feature_cols].values
        recent_data = self.normalize_sequence(recent_data)
        
        x_tensor = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0) # Batch size 1
        
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()
            
        return {
            "DOWN": float(probs[0] * 100),
            "FLAT": float(probs[1] * 100),
            "UP": float(probs[2] * 100)
        }
