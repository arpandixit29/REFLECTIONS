import time
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class BaseReflectionsAgent(ABC):
    """
    Abstract Base Class for all Reflections Autonomous Safety Agents.
    Every agent maintains state, performance metrics, sensitivity threshold,
    and returns standard structured evaluations.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        category: str,
        description: str,
        sensitivity: float = 0.5,
    ):
        self.agent_id = agent_id
        self.name = name
        self.category = category
        self.description = description
        self.enabled = True
        self.sensitivity = sensitivity
        
        # Telemetry metrics
        self.eval_count = 0
        self.threat_count = 0
        self.total_latency_ms = 0.0
        self.last_eval_time = None
        self.logs: List[Dict[str, Any]] = []

    def toggle(self, state: Optional[bool] = None) -> bool:
        """Toggle agent ON/OFF status."""
        if state is None:
            self.enabled = not self.enabled
        else:
            self.enabled = state
        self.log(f"Agent state changed to {'ENABLED' if self.enabled else 'DISABLED'}")
        return self.enabled

    def set_sensitivity(self, value: float) -> float:
        """Adjust agent sensitivity threshold (0.0 to 1.0)."""
        self.sensitivity = max(0.0, min(1.0, value))
        self.log(f"Sensitivity threshold set to {self.sensitivity:.2f}")
        return self.sensitivity

    def log(self, message: str, level: str = "INFO", details: Optional[Dict[str, Any]] = None):
        """Record an execution step into agent reflections log."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "level": level,
            "message": message,
            "details": details or {},
        }
        self.logs.append(entry)
        if len(self.logs) > 100:
            self.logs.pop(0)

    @abstractmethod
    async def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate frame or stream payload.
        Returns dict containing:
        - agent_id: str
        - triggered: bool
        - score: float (0.0 to 1.0)
        - action: str (NONE, BLUR, BLOCK, WARN, REDACT)
        - reasoning: str
        - latency_ms: float
        """
        pass

    def get_telemetry(self) -> Dict[str, Any]:
        """Return real-time diagnostic telemetry for UI dashboard."""
        avg_latency = (
            (self.total_latency_ms / self.eval_count)
            if self.eval_count > 0
            else 0.0
        )
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "enabled": self.enabled,
            "sensitivity": self.sensitivity,
            "eval_count": self.eval_count,
            "threat_count": self.threat_count,
            "avg_latency_ms": round(avg_latency, 2),
            "last_eval_time": self.last_eval_time,
            "recent_logs": self.logs[-5:],
        }
