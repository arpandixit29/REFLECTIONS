import time
import asyncio
from typing import Dict, Any, List
from .base_agent import BaseReflectionsAgent
from .nsfw_agent import NSFWProtectionAgent
from .wound_agent import GraphicWoundAgent
from .gesture_agent import GestureDefenseAgent
from .weapon_agent import ThreatWeaponAgent
from .deepfake_agent import DeepfakeForensicsAgent
from .scam_agent import ScamDefenseAgent
from .privacy_agent import PrivacyPIIAgent
from .audio_agent import AudioProfanityAgent


class MasterSupervisorAgent(BaseReflectionsAgent):
    """
    Master Reflections Supervisor & Orchestrator Agent.
    Coordinates parallel execution of all 8 specialized safety sub-agents,
    synthesizes cross-domain threat vectors, computes consensus action policy,
    and maintains live reflections execution history.
    """

    def __init__(self, models_dict: Dict[str, Any] = None):
        super().__init__(
            agent_id="master_supervisor",
            name="Master Supervisor Orchestrator Agent",
            category="Multi-Agent Supervisor",
            description="Coordinates all 8 domain safety agents in parallel, aggregating threat consensus & policy.",
            sensitivity=0.5,
        )
        models_dict = models_dict or {}
        
        # Instantiate sub-agents
        self.sub_agents: Dict[str, BaseReflectionsAgent] = {
            "nsfw_agent": NSFWProtectionAgent(model_handle=models_dict.get("nsfw")),
            "wound_agent": GraphicWoundAgent(model_handle=models_dict.get("wound")),
            "gesture_agent": GestureDefenseAgent(),
            "weapon_agent": ThreatWeaponAgent(),
            "deepfake_agent": DeepfakeForensicsAgent(),
            "scam_agent": ScamDefenseAgent(),
            "privacy_agent": PrivacyPIIAgent(),
            "audio_agent": AudioProfanityAgent(),
        }
        self.reflections_log: List[Dict[str, Any]] = []

    def get_agent(self, agent_id: str) -> BaseReflectionsAgent:
        return self.sub_agents.get(agent_id)

    def toggle_sub_agent(self, agent_id: str, state: bool = None) -> bool:
        agent = self.sub_agents.get(agent_id)
        if agent:
            return agent.toggle(state)
        return False

    def set_sub_agent_sensitivity(self, agent_id: str, sensitivity: float) -> float:
        agent = self.sub_agents.get(agent_id)
        if agent:
            return agent.set_sensitivity(sensitivity)
        return 0.5

    def get_all_agent_statuses(self) -> List[Dict[str, Any]]:
        statuses = [self.get_telemetry()]
        for agent in self.sub_agents.values():
            statuses.append(agent.get_telemetry())
        return statuses

    async def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        
        if not self.enabled:
            return {
                "supervisor": "DISABLED",
                "overall_action": "NONE",
                "triggered_agents": [],
                "agent_results": {},
                "latency_ms": 0.0,
            }

        # Run all active sub-agents concurrently
        tasks = []
        agent_keys = []
        for key, agent in self.sub_agents.items():
            if agent.enabled:
                tasks.append(agent.evaluate(payload))
                agent_keys.append(key)

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        agent_results = {}
        triggered_agents = []
        highest_score = 0.0
        priority_action = "NONE"
        action_hierarchy = {"BLOCK": 4, "BLUR": 3, "REDACT": 3, "WARN": 2, "NONE": 1}

        for key, res in zip(agent_keys, results_list):
            if isinstance(res, Exception):
                agent_results[key] = {"triggered": False, "error": str(res)}
                continue

            agent_results[key] = res
            if res.get("triggered", False):
                triggered_agents.append(key)
                score = res.get("score", 0.0)
                if score > highest_score:
                    highest_score = score
                
                act = res.get("action", "NONE")
                if action_hierarchy.get(act, 0) > action_hierarchy.get(priority_action, 0):
                    priority_action = act

        total_latency_ms = (time.time() - start_time) * 1000.0
        self.eval_count += 1
        if len(triggered_agents) > 0:
            self.threat_count += 1
        self.total_latency_ms += total_latency_ms
        self.last_eval_time = time.strftime("%H:%M:%S")

        supervisor_reasoning = (
            f"Multi-Agent Consensus: {len(triggered_agents)} agents triggered [{', '.join(triggered_agents)}]. Final Action Policy: {priority_action}."
            if len(triggered_agents) > 0
            else "Multi-Agent Consensus: Stream verified 100% CLEAR across all active safety agents."
        )

        reflections_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "latency_ms": round(total_latency_ms, 2),
            "triggered_agents": triggered_agents,
            "overall_action": priority_action,
            "highest_score": round(highest_score, 4),
            "reasoning": supervisor_reasoning,
        }
        self.reflections_log.append(reflections_entry)
        if len(self.reflections_log) > 200:
            self.reflections_log.pop(0)

        self.log(supervisor_reasoning, level="WARN" if len(triggered_agents) > 0 else "INFO")

        return {
            "supervisor_status": "ACTIVE",
            "overall_action": priority_action,
            "highest_score": round(highest_score, 4),
            "triggered_agents": triggered_agents,
            "agent_results": agent_results,
            "reasoning": supervisor_reasoning,
            "latency_ms": round(total_latency_ms, 2),
        }
