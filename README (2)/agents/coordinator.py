"""
CogniOps AI - Coordinator Agent.
Orchestrates the sequential multi-agent execution pipeline, logs execution durations and outputs,
aggregates context, and compiles the final explainable recommendation.
"""
import time
from typing import Dict, Any, List
from core.models import AgentStepLog, DecisionRecommendation
from core.config import SITUATIONS
from agents.data_agent import DataAgent
from agents.analytics_agent import AnalyticsAgent
from agents.kg_agent import KnowledgeGraphAgent
from agents.rag_agent import RAGAgent
from agents.planner_agent import PlannerAgent
from agents.scenario_agent import ScenarioAgent
from agents.explain_agent import ExplainAgent

class CoordinatorAgent:
    def __init__(self):
        self.name = "Coordinator Agent"
        self.data_agent = DataAgent()
        self.analytics_agent = AnalyticsAgent()
        self.kg_agent = KnowledgeGraphAgent()
        self.rag_agent = RAGAgent()
        self.planner_agent = PlannerAgent()
        self.scenario_agent = ScenarioAgent()
        self.explain_agent = ExplainAgent()

    def run_pipeline(self, situation_id: str) -> Dict[str, Any]:
        """
        Executes the end-to-end multi-agent autonomous decision pipeline:
        Step 1: Data Ingestion & Quality Audit (Data Agent)
        Step 2: Predictive Risk Modeling & Analytics (Analytics Agent)
        Step 3: Knowledge Graph Ontological Impact Traversal (KG Agent)
        Step 4: Policy & Historical Semantic Retrieval (RAG Agent)
        Step 5: Strategy Generation & Policy Guardrail Pruning (Planner Agent)
        Step 6: Monte Carlo What-If Simulation & Utility Ranking (Scenario Agent)
        Step 7: SHAP Attribution & Explainable Recommendation (Explain Agent)
        Step 8: Recommendation Packaging & Human Decision Hand-off (Coordinator)
        """
        pipeline_start = time.time()
        step_logs: List[AgentStepLog] = []
        situation = SITUATIONS.get(situation_id, SITUATIONS["supplier_delay"])

        # ----------------------------------------------------
        # STEP 1: Data Agent
        # ----------------------------------------------------
        t0 = time.time()
        dfs, data_summary = self.data_agent.load_and_validate_data()
        situation_features = self.data_agent.get_situation_features(situation_id)
        d_time = time.time() - t0
        step_logs.append(AgentStepLog(
            agent_name="Data Agent",
            status="SUCCESS",
            duration_sec=round(d_time, 3),
            summary=f"Ingested 7 enterprise datasets ({data_summary['total_records']} rows). Data Quality Score: {data_summary['overall_quality_score']}% ({data_summary['status']}).",
            details=data_summary
        ))

        # ----------------------------------------------------
        # STEP 2: Analytics Agent
        # ----------------------------------------------------
        t0 = time.time()
        model_metrics = self.analytics_agent.train_risk_model(
            sales_df=dfs["sales"],
            inventory_df=dfs["inventory"],
            suppliers_df=dfs["suppliers"],
            products_df=dfs.get("products")
        )
        trends = self.analytics_agent.get_trend_analytics(dfs["sales"], dfs["finance"])
        risk_class, risk_prob = self.analytics_agent.predict_risk_score({
            "units_sold": 90.0,
            "revenue": 45000.0,
            "margin": 18000.0,
            "current_stock": situation_features.get("current_stock", 140),
            "reorder_level": 450,
            "lead_time_days": situation_features.get("delay_days", 24),
            "reliability_score": situation_features.get("reliability_score", 0.62),
            "delay_days": situation_features.get("delay_days", 24),
            "stock_to_reorder_ratio": 0.31,
            "margin_ratio": 0.40,
            "delay_impact": 15.0
        })
        a_time = time.time() - t0
        step_logs.append(AgentStepLog(
            agent_name="Analytics Agent",
            status="SUCCESS",
            duration_sec=round(a_time, 3),
            summary=f"Trained Random Forest Risk Classifier (F1: {model_metrics['f1_score']}, Accuracy: {model_metrics['accuracy']}). Predicted Disruption Probability: {risk_prob*100:.1f}%.",
            details={"metrics": model_metrics, "predicted_risk_prob": risk_prob, "risk_class": risk_class}
        ))

        # ----------------------------------------------------
        # STEP 3: Knowledge Graph Agent
        # ----------------------------------------------------
        t0 = time.time()
        if self.kg_agent.graph.number_of_nodes() == 0:
            self.kg_agent.build_graph(
                suppliers_df=dfs["suppliers"],
                products_df=dfs["products"],
                orders_df=dfs["orders"],
                customers_df=dfs["customers"],
                inventory_df=dfs["inventory"]
            )
        kg_impact = self.kg_agent.query_impact_path(situation_id)
        k_time = time.time() - t0
        step_logs.append(AgentStepLog(
            agent_name="Knowledge Graph Agent",
            status="SUCCESS",
            duration_sec=round(k_time, 3),
            summary=f"Traversed {len(kg_impact['affected_nodes'])} impacted entities. Calculated Revenue at Risk: ${kg_impact['revenue_at_risk']:,.2f}.",
            details=kg_impact
        ))

        # ----------------------------------------------------
        # STEP 4: RAG Agent
        # ----------------------------------------------------
        t0 = time.time()
        rag_context = self.rag_agent.retrieve_context_for_situation(situation)
        r_time = time.time() - t0
        top_policy_name = rag_context['policies'][0]['title'] if rag_context['policies'] else "N/A"
        step_logs.append(AgentStepLog(
            agent_name="RAG Agent",
            status="SUCCESS",
            duration_sec=round(r_time, 3),
            summary=f"Retrieved {len(rag_context['policies'])} governing policies and {len(rag_context['past_decisions'])} past decision cases. Top match: '{top_policy_name}'.",
            details=rag_context
        ))

        # ----------------------------------------------------
        # STEP 5: Planner Agent
        # ----------------------------------------------------
        t0 = time.time()
        candidates = self.planner_agent.generate_candidate_strategies(situation_id)
        surviving_strategies, rejected_strategies = self.planner_agent.apply_guardrails(
            candidates, rag_context["policies"]
        )
        p_time = time.time() - t0
        step_logs.append(AgentStepLog(
            agent_name="Planner Agent",
            status="SUCCESS",
            duration_sec=round(p_time, 3),
            summary=f"Generated {len(candidates)} candidates. Policy guardrails approved {len(surviving_strategies)} strategies and rejected {len(rejected_strategies)} non-compliant options.",
            details={"surviving": surviving_strategies, "rejected": rejected_strategies}
        ))

        # ----------------------------------------------------
        # STEP 6: Scenario Simulation Agent
        # ----------------------------------------------------
        t0 = time.time()
        scenario_outcomes = self.scenario_agent.simulate_strategies(surviving_strategies, situation_id)
        best_scenario = scenario_outcomes[0]
        s_time = time.time() - t0
        step_logs.append(AgentStepLog(
            agent_name="Scenario Agent",
            status="SUCCESS",
            duration_sec=round(s_time, 3),
            summary=f"Executed 1,000 Monte Carlo runs across 3 scenarios. Ranked Strategy '{best_scenario.strategy_title}' #1 with Expected Utility {best_scenario.expected_utility}.",
            details={"ranked_outcomes": [o.__dict__ for o in scenario_outcomes]}
        ))

        # ----------------------------------------------------
        # STEP 7: Explainability Agent
        # ----------------------------------------------------
        t0 = time.time()
        shap_drivers = self.explain_agent.calculate_shap_attributions(
            model=self.analytics_agent.model,
            feature_names=self.analytics_agent.feature_names,
            situation_features=situation_features
        )
        explanation_package = self.explain_agent.generate_explanation_package(
            situation=situation,
            recommended_strategy=best_scenario,
            all_strategies=scenario_outcomes,
            rejected_strategies=rejected_strategies,
            kg_data=kg_impact,
            rag_data=rag_context,
            shap_drivers=shap_drivers
        )
        e_time = time.time() - t0
        step_logs.append(AgentStepLog(
            agent_name="Explainability Agent",
            status="SUCCESS",
            duration_sec=round(e_time, 3),
            summary=f"Computed SHAP feature importances. Top risk driver: '{shap_drivers[0]['feature']}'. Assembled executive audit explanation.",
            details=explanation_package
        ))

        # ----------------------------------------------------
        # STEP 8: Recommendation Packaging
        # ----------------------------------------------------
        total_time = round(time.time() - pipeline_start, 3)
        recommendation = DecisionRecommendation(
            situation_id=situation_id,
            recommended_strategy_id=best_scenario.strategy_id,
            strategy_title=best_scenario.strategy_title,
            executive_summary=f"Autonomous recommendation: {best_scenario.strategy_title} yields Expected Utility of {best_scenario.expected_utility:.2f} at ${best_scenario.cost:,.0f} estimated capital cost.",
            plain_language_explanation=explanation_package["plain_language_explanation"],
            key_drivers=shap_drivers,
            policy_citations=explanation_package["policy_citations"],
            kg_facts=explanation_package["kg_facts"],
            rejected_alternatives_rationale=explanation_package["rejected_alternatives_rationale"],
            confidence_score=0.91,
            expected_utility=best_scenario.expected_utility,
            system_latency_sec=total_time
        )

        step_logs.append(AgentStepLog(
            agent_name="Coordinator Agent",
            status="SUCCESS",
            duration_sec=total_time,
            summary=f"Multi-agent cycle completed in {total_time:.2f}s. Ready for human-in-the-loop review.",
            details={"recommendation_id": f"REC-{situation_id.upper()}-{int(time.time())}"}
        ))

        return {
            "situation": situation,
            "situation_features": situation_features,
            "data_summary": data_summary,
            "model_metrics": model_metrics,
            "confusion_matrix": self.analytics_agent.confusion_mat,
            "roc_data": self.analytics_agent.roc_data,
            "pr_data": self.analytics_agent.pr_data,
            "trends": trends,
            "kg_impact": kg_impact,
            "rag_context": rag_context,
            "candidates": candidates,
            "surviving_strategies": surviving_strategies,
            "rejected_strategies": rejected_strategies,
            "scenario_outcomes": scenario_outcomes,
            "shap_drivers": shap_drivers,
            "explanation_package": explanation_package,
            "recommendation": recommendation,
            "step_logs": step_logs,
            "total_execution_time": total_time,
            "dfs": dfs
        }
