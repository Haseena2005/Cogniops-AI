"""
CogniOps AI - Data Models and Schema Definitions.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class AgentStepLog:
    agent_name: str
    status: str
    duration_sec: float
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievedContext:
    doc_id: str
    title: str
    snippet: str
    score: float
    doc_type: str  # 'policy' or 'past_decision'
    source_file: str

@dataclass
class StrategyCandidate:
    id: str
    title: str
    description: str
    action_type: str
    estimated_cost: float
    resource_level: str
    policy_compliant: bool = True
    violated_policy: Optional[str] = None
    violation_reason: Optional[str] = None

@dataclass
class ScenarioOutcome:
    strategy_id: str
    strategy_title: str
    optimistic_value: float
    expected_value: float
    pessimistic_value: float
    expected_utility: float
    risk_std_dev: float
    worst_case_downside: float
    cost: float
    net_roi: float
    rank: int = 1

@dataclass
class DecisionRecommendation:
    situation_id: str
    recommended_strategy_id: str
    strategy_title: str
    executive_summary: str
    plain_language_explanation: str
    key_drivers: List[Dict[str, Any]]
    policy_citations: List[str]
    kg_facts: List[str]
    rejected_alternatives_rationale: List[Dict[str, str]]
    confidence_score: float
    expected_utility: float
    system_latency_sec: float
