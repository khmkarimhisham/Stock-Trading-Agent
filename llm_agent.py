import requests
import json
import config
import logging

class LLMAgent:
    def __init__(self):
        self.api_url = config.OLLAMA_API_URL
        self.model_name = config.LLM_MODEL_NAME

    def analyze(self, symbol, current_price, math_prediction, news_headline=None):
        prompt = f"""
You are an expert quantitative financial analyst AI.
You are evaluating whether to BUY, SELL, or HOLD the stock: {symbol}.
Current Price: ${current_price:.2f}
The mathematical LSTM model predicts an expected hourly price change of: {math_prediction:.4f}% (Positive means expected uptrend, negative means downtrend).
"""
        if news_headline:
            prompt += f"\nBREAKING NEWS: {news_headline}\n"
            prompt += "Evaluate the mathematical prediction in the context of this breaking news.\n"
        else:
            prompt += "\nThere is no recent breaking news. Rely primarily on the mathematical model's trend prediction.\n"

        prompt += """
Respond ONLY with a valid JSON object matching this exact structure, with no markdown formatting or extra text:
{
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": 0-100,
    "reasoning": "1 sentence explanation"
}
"""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                # Parse JSON
                try:
                    data = json.loads(response_text)
                    action = data.get("action", "HOLD").upper()
                    if action not in ["BUY", "SELL", "HOLD"]:
                        action = "HOLD"
                    return {
                        "action": action,
                        "confidence": data.get("confidence", 0),
                        "reasoning": data.get("reasoning", "Parsed from LLM")
                    }
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse LLM JSON: {response_text}")
                    return {"action": "HOLD", "confidence": 0, "reasoning": "JSON parse error"}
            else:
                logging.error(f"Ollama API Error: {response.status_code}")
                return {"action": "HOLD", "confidence": 0, "reasoning": "API Error"}
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to connect to local Ollama API: {e}")
            return {"action": "HOLD", "confidence": 0, "reasoning": "Connection Error"}
