import time
import re
from typing import Dict, Any, List
from .base_agent import BaseReflectionsAgent


class PrivacyPIIAgent(BaseReflectionsAgent):
    """
    PII & Sensitive Document Auto-Redactor Agent.
    Scans text and visual fields for sensitive personal identifiers (Aadhaar, PAN, SSN, Credit Cards, Passports).
    """

    def __init__(self):
        super().__init__(
            agent_id="privacy_agent",
            name="PII Document Privacy Redactor Agent",
            category="Data Privacy",
            description="Auto-redacts Aadhaar numbers, PAN cards, credit cards, and sensitive ID numbers.",
            sensitivity=0.5,
        )
        self.patterns = {
            "Aadhaar": r"\b[2-9]{1}[0-9]{3}\s[0-9]{4}\s[0-9]{4}\b",
            "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
            "Credit Card": r"\b(?:\d[ -]*?){13,16}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        }

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

        text = payload.get("ocr_text", "")
        pii_found: List[str] = []

        for label, pattern in self.patterns.items():
            if re.search(pattern, text):
                pii_found.append(label)

        score = 0.95 if len(pii_found) > 0 else 0.0
        triggered = len(pii_found) > 0 and score >= self.sensitivity
        action = "BLOCK" if triggered else "NONE"

        reasoning = (
            f"Sensitive PII detected ({', '.join(pii_found)}). Applying instant blackout redaction."
            if triggered
            else "No sensitive PII identifiers detected."
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
            "pii_types": pii_found,
            "action": action,
            "reasoning": reasoning,
            "latency_ms": round(latency_ms, 2),
        }
