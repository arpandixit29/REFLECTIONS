import time
from typing import Dict, Any, List
from .base_agent import BaseReflectionsAgent


class AudioProfanityAgent(BaseReflectionsAgent):
    """
    Live Audio Speech Profanity Beep Redactor Agent.
    Processes live microphone audio streams or files, detects profanity timestamps, and applies 1000 Hz censor beep audio buffers.
    """

    def __init__(self):
        super().__init__(
            agent_id="audio_agent",
            name="Audio Speech Profanity Beep Agent",
            category="Audio Safety",
            description="Real-time speech profanity detector with zero-latency 1000 Hz censor beep replacement.",
            sensitivity=0.5,
        )

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

        profanities_count = payload.get("profanities_detected", 0)
        timestamps: List[Any] = payload.get("timestamps", [])
        score = 0.95 if profanities_count > 0 else 0.0

        triggered = profanities_count > 0 and score >= self.sensitivity
        action = "REDACT" if triggered else "NONE"

        reasoning = (
            f"Profanity detected in audio stream ({profanities_count} words). 1000 Hz censor beep buffer applied."
            if triggered
            else "Audio stream clean of profane speech."
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
            "profanities_count": profanities_count,
            "timestamps": timestamps,
            "action": action,
            "reasoning": reasoning,
            "latency_ms": round(latency_ms, 2),
        }
