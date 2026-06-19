import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from alpaca_client import AlpacaClient
from math_model import QuantEngine, SimpleLSTM
import config
from rich.progress import track



def prepare_data(engine, alpaca_client, symbols):
    import os, pickle
    
    cache_file = "raw_data_cache.pkl"
    raw_data_dict = {}
    
    if os.path.exists(cache_file):
        print(f"Found local cache ({cache_file}). Loading raw data from disk...")
        with open(cache_file, "rb") as f:
            raw_data_dict = pickle.load(f)
    else:
        print("Downloading historical data from Alpaca...")
        for symbol in track(symbols, description="Fetching data"):
            try:
                # Get last 6 months of data for training
                df = alpaca_client.get_historical_bars(symbol, days_back=180)
                raw_data_dict[symbol] = df
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        print(f"Saving data to cache ({cache_file})...")
        with open(cache_file, "wb") as f:
            pickle.dump(raw_data_dict, f)

    train_x, train_y = [], []
    val_x, val_y = [], []
    seq_length = 60
    
    print("Preparing sequences and engineering features...")
    for symbol, df in raw_data_dict.items():
        try:
            # Add technical features
            features_df = engine.engineer_features(df)
            if features_df is None or len(features_df) < seq_length + 1:
                continue
                
            # Extract the 14 robust technical features
            feature_cols = ['close', 'volume', 'rsi', 'macd', 'macd_diff', 
                            'stoch', 'stoch_signal', 'bb_upper', 'bb_lower', 
                            'bb_width', 'atr', 'obv', 'ema_9', 'ema_21']
            data_values = features_df[feature_cols].values
            
            # Calculate target: classify into DOWN (0), UP (1)
            closes = features_df['close'].values
            targets = np.zeros(len(closes))
            for i in range(len(closes) - 5):
                ret = ((closes[i+5] - closes[i]) / closes[i]) * 100.0
                if ret >= 0:
                    targets[i] = 1
                else:
                    targets[i] = 0
                
            symbol_x, symbol_y = [], []
            # Create sequences
            for i in range(len(data_values) - seq_length - 4):
                x_seq = data_values[i : i + seq_length].copy()
                
                # Sequence-level normalization matching the QuantEngine method
                x_seq = engine.normalize_sequence(x_seq)
                
                y_val = targets[i + seq_length - 1] # Target is next 5 mins movement
                
                symbol_x.append(x_seq)
                symbol_y.append(y_val)
                
            # Chronological split for this symbol: 80% train, 20% validation
            split_idx = int(len(symbol_x) * 0.8)
            train_x.extend(symbol_x[:split_idx])
            train_y.extend(symbol_y[:split_idx])
            val_x.extend(symbol_x[split_idx:])
            val_y.extend(symbol_y[split_idx:])
            
        except Exception as e:
            print(f"Error processing data for {symbol}: {e}")
            
    if not train_x:
        raise ValueError("Not enough data fetched to train. Check your internet connection or Alpaca limits.")
        
    train_dataset = TensorDataset(
        torch.tensor(np.array(train_x), dtype=torch.float32),
        torch.tensor(np.array(train_y), dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(np.array(val_x), dtype=torch.float32),
        torch.tensor(np.array(val_y), dtype=torch.long)
    )
    
    return train_dataset, val_dataset

def train_model():
    engine = QuantEngine()
    alpaca = AlpacaClient()
    
    train_dataset, val_dataset = prepare_data(engine, alpaca, config.SYMBOLS)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = SimpleLSTM(input_size=14)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    import copy
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    epochs = 100
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=steps_per_epoch, epochs=epochs)
    
    patience = 15
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    print(f"\nStarting training on {device}...")
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        # Step the scheduler with validation loss
        # (OneCycleLR is stepped per batch instead)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")
        
        # Early Stopping and Model Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}!")
                break
                
    print("\nTraining complete!")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    torch.save(model.state_dict(), "lstm_model.pth")
    print(f"Best model weights saved to 'lstm_model.pth' (Val Loss: {best_val_loss:.6f}). The main bot will now use these trained weights.")

if __name__ == "__main__":
    train_model()
