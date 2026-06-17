import requests
import json
import config
import logging

class LLMAgent:
    def __init__(self):
        self.api_url = config.OLLAMA_API_URL
        self.model_name = config.LLM_MODEL_NAME

    def analyze(self, symbol, current_price, math_prediction, news_headline=None):
        # 1. Base mathematical decision
        up_prob = math_prediction.get('UP', 0)
        down_prob = math_prediction.get('DOWN', 0)
        
        if up_prob > 50.0:
            base_action = "BUY"
            base_reasoning = f"Math model predicts an uptrend ({up_prob:.1f}%)."
            confidence = up_prob
        elif down_prob > 50.0:
            base_action = "SELL"
            base_reasoning = f"Math model predicts a downtrend ({down_prob:.1f}%)."
            confidence = down_prob
        else:
            base_action = "HOLD"
            base_reasoning = f"Math probabilities are neutral (UP: {up_prob:.1f}%, DOWN: {down_prob:.1f}%)."
            confidence = max(up_prob, down_prob)

        # 2. Return math decision if no news
        if not news_headline:
            return {
                "action": base_action,
                "confidence": confidence,
                "reasoning": base_reasoning
            }

        # 3. If news exists, ask LLM to evaluate news against math baseline
        logging.info(f"\nBREAKING NEWS: {news_headline}\n")
        
        prompt = f"""
You are an expert quantitative financial analyst AI.
You are evaluating whether to BUY, SELL, or HOLD the stock: {symbol}.
Current Price: ${current_price:.2f}

The mathematical LSTM model predicts the following probabilities:
- UPTREND: {up_prob:.1f}%
- DOWNTREND: {down_prob:.1f}%
Based on these numbers, the baseline recommendation is {base_action}.

BREAKING NEWS: {news_headline}

Evaluate if this breaking news confirms the baseline recommendation of {base_action}, or if it is significant enough to override it.
Respond ONLY with a valid JSON object matching this exact structure, with no markdown formatting or extra text:
{{
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": 0-100,
    "reasoning": "1 sentence explanation combining math and news"
}}
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
