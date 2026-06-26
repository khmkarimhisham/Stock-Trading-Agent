import requests
import logging
import config

def send_discord_alert(message: str):
    """Sends an alert to the configured Discord webhook."""
    if not config.DISCORD_WEBHOOK_URL:
        logging.warning("No Discord Webhook URL configured.")
        return
    
    try:
        data = {"content": message}
        response = requests.post(config.DISCORD_WEBHOOK_URL, json=data)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to send Discord alert: {e}")
