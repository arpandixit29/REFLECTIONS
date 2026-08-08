import time
from typing import Dict, Any
from .base_agent import BaseReflectionsAgent


class NSFWProtectionAgent(BaseReflectionsAgent):
    """
    Autonomous Explicit Content Protection Agent.
    Evaluates imagery for nudity, pornography, explicit adult content, and hentai.
    Triggers dynamic frosted-glass blur redactions.
    """

    def __init__(self, model_handle=None):
        super().__init__(
            agent_id="nsfw_agent",
            name="Explicit Content Protection Agent",
            category="NSFW Defense",
            description="Detects explicit imagery, adult content, and pornography in real time.",
            sensitivity=0.6,
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
                "reasoning": "Agent disabled by user.",
                "latency_ms": 0.0,
            }

        # Retrieve scores from payload or PyTorch model
        nsfw_score = payload.get("nsfw_score", 0.0)
        category_label = payload.get("nsfw_label", "Neutral")

        # Evaluate against sensitivity threshold
        triggered = nsfw_score >= self.sensitivity
        action = "BLUR" if triggered else "NONE"
        reasoning = (
            f"Explicit content detected ({category_label}: {nsfw_score*100:.1f}% confidence >= threshold {self.sensitivity*100:.1f}%)."
            if triggered
            else f"Visual stream clear ({category_label}: {nsfw_score*100:.1f}%)."
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
            "score": round(nsfw_score, 4),
            "label": category_label,
            "action": action,
            "reasoning": reasoning,
            "latency_ms": round(latency_ms, 2),
        }
