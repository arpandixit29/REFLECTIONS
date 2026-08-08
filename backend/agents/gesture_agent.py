import time
from typing import Dict, Any, List
from .base_agent import BaseReflectionsAgent


class GestureDefenseAgent(BaseReflectionsAgent):
    """
    Offensive Hand Sign & Gesture Defense Agent.
    Tracks 3D keypoints from MediaPipe to detect middle finger gestures and profane hand signs.
    """

    def __init__(self):
        super().__init__(
            agent_id="gesture_agent",
            name="Offensive Gesture Defense Agent",
            category="Gesture Safety",
            description="Tracks 3D hand keypoints to blur middle finger gestures during live video streams.",
            sensitivity=0.6,
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

        bad_gestures: List[Dict[str, Any]] = payload.get("bad_gestures", [])
        score = payload.get("gesture_score", 1.0 if len(bad_gestures) > 0 else 0.0)

        triggered = len(bad_gestures) > 0 or score >= self.sensitivity
        action = "BLUR" if triggered else "NONE"
        reasoning = (
            f"Offensive gesture detected ({len(bad_gestures)} instances found)."
            if triggered
            else "Hand gestures verified safe."
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
            "boxes": bad_gestures,
            "action": action,
            "reasoning": reasoning,
            "latency_ms": round(latency_ms, 2),
        }
