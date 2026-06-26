import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import ta
import os
import json
import logging

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        
    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out

class StableTCNModel(nn.Module):
    def __init__(self, input_size, num_tickers=1, embedding_dim=8, num_channels=[32, 64, 128, 256], kernel_size=5, dropout=0.5):
        super(StableTCNModel, self).__init__()
        self.ticker_embedding = nn.Embedding(num_tickers, embedding_dim)
        layers = []
        num_levels = len(num_channels)
        in_channels = input_size + embedding_dim
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            layers += [
                CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation_size),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_channels = out_channels
            
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 2)
        
    def forward(self, x, ticker_idx):
        emb = self.ticker_embedding(ticker_idx)
        emb = emb.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = torch.cat([x, emb], dim=-1)
        
        x = x.transpose(1, 2)
        out = self.tcn(x)
        out = out[:, :, -1]
        out = self.fc(out)
        return out

class QuantEngine:
    def __init__(self):
        # We now use 20 advanced features
        self.feature_cols = ['close', 'volume', 'rsi', 'macd', 'macd_diff', 
                             'stoch', 'stoch_signal', 'bb_upper', 'bb_lower', 
                             'bb_width', 'atr', 'obv', 'ema_9', 'ema_21',
                             'williams_r', 'cci', 'roc', 'uo', 'ichimoku_a', 'ichimoku_b']
        
        self.device = torch.device('cpu') # Enforce CPU for inference

        self.ticker_dict = {}
        if os.path.exists("ticker_dict.json"):
            try:
                with open("ticker_dict.json", "r") as f:
                    self.ticker_dict = json.load(f)
            except Exception as e:
                logging.warning(f"QuantEngine: Could not load ticker_dict.json. {e}")

        num_tickers = max(1, len(self.ticker_dict))
        self.model = StableTCNModel(input_size=len(self.feature_cols), num_tickers=num_tickers)

        # Load pre-trained weights if they exist
        if os.path.exists("model.pth"):
            try:
                self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
            except Exception as e:
                logging.warning(f"QuantEngine: Could not load old weights (architecture changed). Using random weights.")
        else:
            logging.warning("QuantEngine: No model.pth found. Using randomly initialized weights.")
        
        self.model.to(self.device)
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
        
        # 4. Additional Indicators
        df['williams_r'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
        df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
        df['roc'] = ta.momentum.ROCIndicator(close=df['close']).roc()
        
        # 5. Advanced Trend & Oscillators
        df['uo'] = ta.momentum.UltimateOscillator(high=df['high'], low=df['low'], close=df['close']).ultimate_oscillator()
        ichimoku = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        
        # Drop NaNs created by rolling windows
        df.dropna(inplace=True)
        return df

    def normalize_sequence(self, seq):
        norm_seq = seq.copy()
        
        # 1. Price-based columns (close, bb_upper, bb_lower, ema_9, ema_21, ichimoku_a, ichimoku_b)
        # Normalize using the close price mean/std to preserve relative distances
        price_cols = [0, 7, 8, 12, 13, 18, 19]
        mean_price = np.mean(seq[:, 0])
        std_price = np.std(seq[:, 0])
        if std_price < 1e-8:
            std_price = 1e-8
            
        for col in price_cols:
            norm_seq[:, col] = (seq[:, col] - mean_price) / std_price
            
        # 2. Independent continuous columns (volume, macd, macd_diff, bb_width, atr, obv, williams_r, cci, roc)
        indep_cols = [1, 3, 4, 9, 10, 11, 14, 15, 16]
        for col in indep_cols:
            mean_val = np.mean(seq[:, col])
            std_val = np.std(seq[:, col])
            if std_val < 1e-8:
                std_val = 1e-8
            norm_seq[:, col] = (seq[:, col] - mean_val) / std_val
            
        # 3. Bounded columns [0, 100] (rsi, stoch, stoch_signal, uo)
        bounded_cols = [2, 5, 6, 17]
        norm_seq[:, bounded_cols] = (seq[:, bounded_cols] - 50.0) / 50.0
        
        return norm_seq

    def predict(self, df, symbol):
        """
        Returns a dictionary of probabilities for DOWN, UP
        """
        features_df = self.engineer_features(df)
        if features_df is None or features_df.empty:
            return {"DOWN": 0.0, "UP": 0.0}

        # Take the last 120 steps as the sequence for prediction
        seq_length = 120
        if len(features_df) < seq_length:
            return {"DOWN": 0.0, "UP": 0.0}

        recent_data = features_df.iloc[-seq_length:][self.feature_cols].values
        recent_data = self.normalize_sequence(recent_data)
        
        x_tensor = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0).to(self.device) # Batch size 1
        
        ticker_id = self.ticker_dict.get(symbol, 0)
        ticker_tensor = torch.tensor([ticker_id], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x_tensor, ticker_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
            
        return {
            "DOWN": float(probs[0] * 100),
            "UP": float(probs[1] * 100)
        }
