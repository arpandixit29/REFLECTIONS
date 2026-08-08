import time
from typing import Dict, Any
from .base_agent import BaseReflectionsAgent


class GraphicWoundAgent(BaseReflectionsAgent):
    """
    Graphic Trauma & Medical Redaction Agent.
    Evaluates imagery for lacerations, open wounds, surgical trauma, and severe bleeding.
    Redacts graphic visual trauma instantly.
    """

    def __init__(self, model_handle=None):
        super().__init__(
            agent_id="wound_agent",
            name="Graphic Wound & Medical Trauma Agent",
            category="Gore Protection",
            description="Detects cuts, open lacerations, and surgical trauma, applying dynamic frosted redaction.",
            sensitivity=0.55,
        )
        self.model_handle = model_handle

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

        wound_score = payload.get("wound_score", 0.0)
        triggered = wound_score >= self.sensitivity
        action = "BLUR" if triggered else "NONE"
        reasoning = (
            f"Graphic wound detected ({wound_score*100:.1f}% confidence >= threshold {self.sensitivity*100:.1f}%)."
            if triggered
            else f"No graphic trauma detected ({wound_score*100:.1f}%)."
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
            "score": round(wound_score, 4),
            "action": action,
            "reasoning": reasoning,
            "latency_ms": round(latency_ms, 2),
        }
