import time
from typing import Dict, Any, List
from .base_agent import BaseReflectionsAgent


class ThreatWeaponAgent(BaseReflectionsAgent):
    """
    Threat Object & Weapon Safety Agent.
    Evaluates COCO-SSD / YOLO object detections for sharp threats, knives, scissors, and weapons.
    Targeted bounding box redactions only on dangerous objects.
    """

    def __init__(self):
        super().__init__(
            agent_id="weapon_agent",
            name="Threat Object & Weapon Agent",
            category="Threat Protection",
            description="Identifies knives, scissors, and weapons, blurring threat areas specifically.",
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

        detected_threats: List[Dict[str, Any]] = payload.get("dangerous_objects", [])
        max_score = max([t.get("score", 0.0) for t in detected_threats], default=0.0)

        triggered = len(detected_threats) > 0 and max_score >= self.sensitivity
        action = "BLUR" if triggered else "NONE"
        threat_names = ", ".join([t.get("label", "weapon") for t in detected_threats])
        
        reasoning = (
            f"Physical threat detected ({threat_names}: max confidence {max_score*100:.1f}%)."
            if triggered
            else "No weapons or sharp threat objects detected."
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
            "score": round(max_score, 4),
            "threats": detected_threats,
            "action": action,
            "reasoning": reasoning,
            "latency_ms": round(latency_ms, 2),
        }
