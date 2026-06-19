import json
import os
import time
import logging
import config
from notifier import send_discord_alert

QUEUE_FILE = "pending_orders.json"

def load_queue():
    if not os.path.exists(QUEUE_FILE):
        return []
    try:
        with open(QUEUE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load queue file: {e}")
        return []

def save_queue(queue):
    try:
        with open(QUEUE_FILE, "w") as f:
            json.dump(queue, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to save queue file: {e}")

def add_to_queue(symbol):
    queue = load_queue()
    
    # Remove any existing pending re-evaluations for the same symbol to avoid duplicates
    queue = [item for item in queue if item["symbol"] != symbol]
    
    queue.append({
        "symbol": symbol,
        "timestamp": time.time()
    })
    save_queue(queue)
    logging.info(f"Queued {symbol} for re-evaluation at market open.")

def process_queued_orders(alpaca_client, log_func, process_signal_func):
    queue = load_queue()
    if not queue:
        return

    log_func(f"Market is open. Re-evaluating {len(queue)} queued symbols...")
    updated_queue = []
    
    ttl = getattr(config, "QUEUED_ORDER_EXPIRATION_SECONDS", 64800)
    current_time = time.time()

    for order in queue:
        symbol = order["symbol"]
        timestamp = order["timestamp"]
        
        # Check expiration of the signal intent
        elapsed = current_time - timestamp
        if elapsed > ttl:
            msg = f"Queued intent for {symbol} has expired (elapsed: {elapsed:.1f}s, TTL: {ttl}s). Skipping re-evaluation."
            log_func(f"[bold red]{msg}[/bold red]")
            logging.warning(msg)
            send_discord_alert(f"⚠️ **QUEUE EXPIRED:** {msg}")
            continue
            
        # Re-evaluate
        log_func(f"Re-evaluating queued symbol: {symbol}...")
        try:
            process_signal_func(symbol)
        except Exception as e:
            msg = f"Failed to re-evaluate {symbol}: {e}"
            log_func(f"[bold red]{msg}[/bold red]")
            logging.error(msg)
            # Keep in queue if it failed for technical reasons, so it tries again
            updated_queue.append(order)

    # Save the updated queue (only contains failed ones, successful/expired ones are removed)
    save_queue(updated_queue)
