import time
import re
from typing import Dict, Any
from .base_agent import BaseReflectionsAgent


class ScamDefenseAgent(BaseReflectionsAgent):
    """
    Financial Scam & Phishing Defense Agent.
    Analyzes OCR text and layout cues for counterfeit bank applications, lottery fraud, and phishing overlays.
    """

    def __init__(self):
        super().__init__(
            agent_id="scam_agent",
            name="Financial Scam & Phishing Defense Agent",
            category="Fraud Prevention",
            description="Identifies fake bank login screens, counterfeit UPI payment apps, and lottery scam texts.",
            sensitivity=0.6,
        )
        self.scam_keywords = [
            "lottery winner", "claim prize", "transfer money immediately",
            "fake upi", "verify bank account urgently", "account suspended click link",
            "otp required to claim", "congratulations you won", "send money to unlock"
        ]

    async def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        if not self.enabled:
            return {
                "agent_id": self.agent_id,
                "triggered": False,
                "score": 0.0,
                "action": "NONE",
                "reasoning": "Agent disabled.",
                "latency_ms": 0.0,
            }

        text = payload.get("ocr_text", "").lower()
        is_scam_flag = payload.get("is_scam", False)

        matches = [kw for kw in self.scam_keywords if kw in text]
        matched_count = len(matches)
        
        score = 0.95 if is_scam_flag or matched_count >= 2 else (0.7 if matched_count == 1 else 0.0)
        triggered = score >= self.sensitivity
        action = "WARN" if triggered else "NONE"

        reasoning = (
            f"Financial scam phishing pattern matched ({matched_count} fraudulent keywords: {', '.join(matches[:2])})."
            if triggered
            else "Screen content clean of financial fraud patterns."
        )

        latency_ms = (time.time() - start_time) * 1000.0
        self.eval_count += 1
        if triggered:
            self.threat_count += 1
        self.total_latency_ms += latency_ms
        self.last_eval_time = time.strftime("%H:%M:%S")

        self.log(reasoning, level="WARN" if triggered else "INFO")

        return {
            "agent_id": self.agent_id,
            "triggered": triggered,
            "score": round(score, 4),
            "matches": matches,
            "action": action,
            "reasoning": reasoning,
            "latency_ms": round(latency_ms, 2),
        }
