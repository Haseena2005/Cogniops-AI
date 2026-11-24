"""
CogniOps AI - Planner Agent.
Generates 3-4 candidate operational strategies per business situation.
Applies rule-based policy guardrails, pruning non-compliant strategies and recording detailed rejection rationales.
"""
from typing import Dict, Any, List, Tuple
from core.models import StrategyCandidate

class PlannerAgent:
    def __init__(self):
        self.name = "Planner Agent"

    def generate_candidate_strategies(self, situation_id: str) -> List[StrategyCandidate]:
        """Generates realistic strategy options tailored to the enterprise situation."""
        if situation_id == "supplier_delay":
            return [
                StrategyCandidate(
                    id="STRAT-01",
                    title="Dual-Source Split (Activate Nordic Precision)",
                    description="Reroute 45% of pending component orders to secondary certified supplier Nordic Precision. Involves setup fee and slightly higher unit pricing.",
                    action_type="Vendor Diversification",
                    estimated_cost=28000.0,
                    resource_level="Medium"
                ),
                StrategyCandidate(
                    id="STRAT-02",
                    title="Charter Emergency Air Freight & Priority Expedite",
                    description="Pay expedited surcharge to fly remaining backlog lots directly from Apex Global Logistics via specialized air cargo lane.",
                    action_type="Expedited Logistics",
                    estimated_cost=22000.0,
                    resource_level="Low"
                ),
                StrategyCandidate(
                    id="STRAT-03",
                    title="Exceed Air Freight Budget for Total Rush Replacement",
                    description="Charter private dedicated cargo jets across all 8 product variants simultaneously with unconstrained priority overtime.",
                    action_type="Extreme Expedited Freight",
                    estimated_cost=36000.0,  # Exceeds POL-004 $25,000 cap!
                    resource_level="High"
                ),
                StrategyCandidate(
                    id="STRAT-04",
                    title="Passive Delay Absorption (Baseline / Inaction)",
                    description="Maintain existing backlog queue with Apex Global Logistics. Notify customers of unavoidable delivery delays without financial intervention.",
                    action_type="Passive Baseline",
                    estimated_cost=0.0,
                    resource_level="None"
                )
            ]

        elif situation_id == "sales_decline":
            return [
                StrategyCandidate(
                    id="STRAT-01",
                    title="Targeted 12% Promotional Incentive & Channel Blitz",
                    description="Launch focused 60-day promotional pricing rebate with distributor partner co-op marketing in Region-East.",
                    action_type="Targeted Marketing & Rebate",
                    estimated_cost=28000.0,
                    resource_level="Medium"
                ),
                StrategyCandidate(
                    id="STRAT-02",
                    title="Aggressive 25% Blanket Price Slash Across Region-East",
                    description="Institute deep price cuts across all product lines to instantly underprice regional competitors and recapture volume.",
                    action_type="Price Discounting",
                    estimated_cost=65000.0, # Violates POL-002 Max 15% discount limit!
                    resource_level="High"
                ),
                StrategyCandidate(
                    id="STRAT-03",
                    title="Regional Partner Enablement & Solution Bundling",
                    description="Package top-selling hardware with cybersecurity software and host technical enablement sessions for systems integrators.",
                    action_type="Value Bundling",
                    estimated_cost=21000.0,
                    resource_level="Medium"
                ),
                StrategyCandidate(
                    id="STRAT-04",
                    title="Market Inaction / Monitor Q4 Organic Recovery",
                    description="Avoid commercial expenditures; observe whether Q4 seasonality self-corrects the regional revenue contraction.",
                    action_type="Passive Baseline",
                    estimated_cost=0.0,
                    resource_level="None"
                )
            ]

        elif situation_id == "inventory_shortage":
            return [
                StrategyCandidate(
                    id="STRAT-01",
                    title="Inter-Warehouse Stock Transfer (WH-COASTAL to WH-CENTRAL)",
                    description="Transfer 280 units of PROD-01 from Coastal buffer to Central hub via dedicated regional freight.",
                    action_type="Inventory Rebalancing",
                    estimated_cost=12000.0,
                    resource_level="Low"
                ),
                StrategyCandidate(
                    id="STRAT-02",
                    title="Selective Order Allocation & Enterprise Priority Rationing",
                    description="Reserve available units strictly for high-ACV Enterprise contracts; offer minor credit vouchers to delayed SMB orders.",
                    action_type="Rationing Protocol",
                    estimated_cost=7500.0,
                    resource_level="Low"
                ),
                StrategyCandidate(
                    id="STRAT-03",
                    title="Deplete WH-COASTAL Entirely Below Minimum Safety Stock",
                    description="Drain 90% of Coastal warehouse reserves to flood Central hub, leaving Coastal region with near-zero safety buffer.",
                    action_type="Unbalanced Depletion",
                    estimated_cost=15000.0, # Violates POL-003 safety buffer rules!
                    resource_level="Medium"
                ),
                StrategyCandidate(
                    id="STRAT-04",
                    title="Wait for Next Regular Replenishment Cycle",
                    description="Do not rebalance warehouses. Accept immediate stockout cancellations on pending orders until next batch arrival.",
                    action_type="Passive Baseline",
                    estimated_cost=0.0,
                    resource_level="None"
                )
            ]

        elif situation_id == "customer_churn_spike":
            return [
                StrategyCandidate(
                    id="STRAT-01",
                    title="Dedicated Executive Sponsor Assignment & 8% ACV Credit",
                    description="Assign VP sponsor to top 15 distressed enterprise clients with goodwill contract credit and expedited engineering support.",
                    action_type="Executive Intervention",
                    estimated_cost=35000.0,
                    resource_level="Medium"
                ),
                StrategyCandidate(
                    id="STRAT-02",
                    title="Premium SLA Upgrade & Priority Ticket Routing",
                    description="Upgrade all at-risk accounts to 24/7 platinum tier response SLAs free of charge for 6 months.",
                    action_type="Service Enhancement",
                    estimated_cost=22000.0,
                    resource_level="Low"
                ),
                StrategyCandidate(
                    id="STRAT-03",
                    title="Uncapped Retention Concession (30% ACV Payout)",
                    description="Offer emergency 30% contract rebates across all complaining accounts to guarantee renewals.",
                    action_type="Uncapped Concession",
                    estimated_cost=95000.0, # Violates POL-005 max 10% ACV credit limit!
                    resource_level="High"
                ),
                StrategyCandidate(
                    id="STRAT-04",
                    title="Standard Automated CS Feedback Loop",
                    description="Dispatch automated CSAT survey emails with no direct commercial concession or dedicated human outreach.",
                    action_type="Passive Baseline",
                    estimated_cost=500.0,
                    resource_level="None"
                )
            ]
        
        return []

    def apply_guardrails(
        self,
        candidates: List[StrategyCandidate],
        retrieved_policies: List[Dict[str, Any]]
    ) -> Tuple[List[StrategyCandidate], List[StrategyCandidate]]:
        """
        Validates candidate strategies against corporate policy guardrails.
        Filters out non-compliant strategies and returns: (surviving_strategies, rejected_strategies).
        """
        surviving = []
        rejected = []

        for strat in candidates:
            compliant = True
            violated_policy = None
            reason = None

            # Guardrail Rule 1: Emergency air freight cap (POL-004: max $25,000)
            if strat.action_type == "Extreme Expedited Freight" or (strat.action_type == "Expedited Logistics" and strat.estimated_cost > 25000.0):
                compliant = False
                violated_policy = "POL-004: Emergency Expedited Freight Expenditure Limits"
                reason = f"Proposed logistics cost (${strat.estimated_cost:,.0f}) breaches the non-negotiable policy limit of $25,000 per procurement incident."

            # Guardrail Rule 2: Promotional discount limit (POL-002: max 15.0%)
            elif "25%" in strat.title or "Blanket Price Slash" in strat.title:
                compliant = False
                violated_policy = "POL-002: Maximum Promotional Discounting & Pricing Guardrails"
                reason = "Proposed 25% price reduction breaches Rule 1 (maximum standard discount cannot exceed 15.0% without executive authorization)."

            # Guardrail Rule 3: Minimum safety buffer depletion (POL-003)
            elif strat.action_type == "Unbalanced Depletion" or "Deplete WH-COASTAL Entirely" in strat.title:
                compliant = False
                violated_policy = "POL-003: Minimum Inventory Safety Stock & Replenishment Policy"
                reason = "Violates Rule 3: Rebalancing must not deplete donor warehouse below 40% of its regional baseline safety buffer."

            # Guardrail Rule 4: Customer retention concession cap (POL-005: max 10% ACV)
            elif "30% ACV" in strat.title or strat.action_type == "Uncapped Concession":
                compliant = False
                violated_policy = "POL-005: Enterprise Account SLA Breach & Customer Retention Protocol"
                reason = "Violates Rule 2: Retention credits authorized up to 10.0% of ACV; proposed 30% credit constitutes gross margin impairment."

            strat.policy_compliant = compliant
            strat.violated_policy = violated_policy
            strat.violation_reason = reason

            if compliant:
                surviving.append(strat)
            else:
                rejected.append(strat)

        return surviving, rejected
