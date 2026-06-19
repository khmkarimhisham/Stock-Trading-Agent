import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import ta
import os
import logging

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2) # Predict 2 classes (DOWN, UP)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.layer_norm(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        return out

class QuantEngine:
    def __init__(self):
        # We now use 12 advanced features
        self.feature_cols = ['close', 'volume', 'rsi', 'macd', 'macd_diff', 
                             'stoch', 'stoch_signal', 'bb_upper', 'bb_lower', 
                             'bb_width', 'atr', 'obv', 'ema_9', 'ema_21']
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
        if df is None or len(df) < 80:
            return None # Not enough data
            
        df = df.copy()
        # 1. Momentum & Trend Indicators
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        df['ema_9'] = ta.trend.EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(close=df['close'], window=21).ema_indicator()
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
        
        # 1. Price-based columns (close, bb_upper, bb_lower, ema_9, ema_21)
        # Normalize using the close price mean/std to preserve relative distances
        price_cols = [0, 7, 8, 12, 13]
        mean_price = np.mean(seq[:, 0])
        std_price = np.std(seq[:, 0])
        if std_price < 1e-8:
            std_price = 1e-8
            
        for col in price_cols:
            norm_seq[:, col] = (seq[:, col] - mean_price) / std_price
            
        # 2. Independent continuous columns (volume, macd, macd_diff, bb_width, atr, obv)
        indep_cols = [1, 3, 4, 9, 10, 11]
        for col in indep_cols:
            mean_val = np.mean(seq[:, col])
            std_val = np.std(seq[:, col])
            if std_val < 1e-8:
                std_val = 1e-8
            norm_seq[:, col] = (seq[:, col] - mean_val) / std_val
            
        # 3. Bounded columns [0, 100] (rsi, stoch, stoch_signal)
        bounded_cols = [2, 5, 6]
        norm_seq[:, bounded_cols] = (seq[:, bounded_cols] - 50.0) / 50.0
        
        return norm_seq

    def predict(self, df):
        """
        Returns a dictionary of probabilities for DOWN, FLAT, UP
        """
        features_df = self.engineer_features(df)
        if features_df is None or features_df.empty:
            return {"DOWN": 0.0, "UP": 0.0}

        # Take the last 60 steps as the sequence for prediction
        seq_length = 60
        if len(features_df) < seq_length:
            return {"DOWN": 0.0, "UP": 0.0}

        recent_data = features_df.iloc[-seq_length:][self.feature_cols].values
        recent_data = self.normalize_sequence(recent_data)
        
        x_tensor = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0) # Batch size 1
        
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()
            
        return {
            "DOWN": float(probs[0] * 100),
            "UP": float(probs[1] * 100)
        }
