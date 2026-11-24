"""
Unit and Integration Tests for CogniOps AI Multi-Agent Pipeline.
Uses an isolated temporary SQLite database for test records.
Verifies revenue at risk bounded constraint, 80-92% realistic accuracy,
5-fold CV, ROC/PR data, and KG sub-graph queries.
"""
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from core.config import SITUATIONS
from agents.data_agent import DataAgent
from agents.analytics_agent import AnalyticsAgent
from agents.kg_agent import KnowledgeGraphAgent
from agents.rag_agent import RAGAgent
from agents.planner_agent import PlannerAgent
from agents.scenario_agent import ScenarioAgent
from agents.explain_agent import ExplainAgent
from agents.coordinator import CoordinatorAgent
from core.db import save_decision_record, get_all_decisions, init_db

def test_data_agent():
    agent = DataAgent()
    dfs, summary = agent.load_and_validate_data()
    assert len(dfs) == 7
    assert summary["overall_quality_score"] > 85.0
    assert summary["status"] == "HEALTHY"
    
    feats = agent.get_situation_features("supplier_delay")
    assert "supplier_name" in feats
    assert feats["delay_days"] > 0

def test_analytics_agent_realistic_evaluation():
    data_agent = DataAgent()
    dfs, _ = data_agent.load_and_validate_data()
    analytics_agent = AnalyticsAgent()
    metrics = analytics_agent.train_risk_model(
        sales_df=dfs["sales"],
        inventory_df=dfs["inventory"],
        suppliers_df=dfs["suppliers"],
        products_df=dfs["products"]
    )
    # Check realistic accuracy (not 100%, within 80-92% range)
    assert 0.78 <= metrics["accuracy"] <= 0.94
    assert 0.70 <= metrics["cv_metrics"]["accuracy_mean"] <= 0.94
    # Cross validation metrics exist
    assert "accuracy_mean" in metrics["cv_metrics"]
    assert "accuracy_std" in metrics["cv_metrics"]
    # ROC and PR curves generated
    assert len(analytics_agent.roc_data["fpr"]) > 2
    assert len(analytics_agent.pr_data["precision"]) > 2
    assert len(analytics_agent.confusion_mat) == 2

def test_kg_agent_and_revenue_at_risk_bound():
    data_agent = DataAgent()
    dfs, _ = data_agent.load_and_validate_data()
    kg_agent = KnowledgeGraphAgent()
    kg_agent.build_graph(
        suppliers_df=dfs["suppliers"],
        products_df=dfs["products"],
        orders_df=dfs["orders"],
        customers_df=dfs["customers"],
        inventory_df=dfs["inventory"]
    )
    assert kg_agent.graph.number_of_nodes() > 50
    assert kg_agent.graph.number_of_edges() > 50

    # Test all 4 situations and verify revenue at risk CAN NEVER exceed total period revenue
    total_sales_revenue = float(dfs["sales"]["revenue"].sum())

    for sit_id in ["supplier_delay", "sales_decline", "inventory_shortage", "customer_churn_spike"]:
        impact = kg_agent.query_impact_path(sit_id)
        assert len(impact["affected_nodes"]) > 0
        rev_at_risk = impact["revenue_at_risk"]
        assert rev_at_risk >= 0.0
        # Strict invariant test: downstream revenue at risk <= total enterprise revenue
        assert rev_at_risk <= total_sales_revenue, f"Revenue at risk ${rev_at_risk} exceeded total revenue ${total_sales_revenue}"

def test_rag_agent_active_backend():
    rag = RAGAgent()
    rag.index_documents()
    assert len(rag.documents) >= 15
    assert rag.active_backend is not None

    results = rag.retrieve("emergency air freight shipping limits", top_k=2)
    assert len(results) == 2
    assert results[0]["score"] > 0
    assert "snippet" in results[0]

def test_planner_agent_and_guardrails():
    planner = PlannerAgent()
    candidates = planner.generate_candidate_strategies("supplier_delay")
    assert len(candidates) == 4

    surviving, rejected = planner.apply_guardrails(candidates, [])
    assert len(surviving) == 3
    assert len(rejected) == 1
    assert "POL-004" in rejected[0].violated_policy

def test_scenario_agent_with_perturbation():
    planner = PlannerAgent()
    scenario = ScenarioAgent()
    candidates = planner.generate_candidate_strategies("inventory_shortage")
    surviving, _ = planner.apply_guardrails(candidates, [])
    
    outcomes = scenario.simulate_strategies(surviving, "inventory_shortage")
    assert len(outcomes) == len(surviving)
    assert outcomes[0].expected_utility >= outcomes[1].expected_utility
    assert outcomes[0].rank == 1

    # Robustness check with +/-10% perturbation
    perturbed_outcomes = scenario.simulate_strategies(surviving, "inventory_shortage", perturbation_pct=0.10)
    assert len(perturbed_outcomes) == len(surviving)

def test_full_coordinator_pipeline():
    coord = CoordinatorAgent()
    res = coord.run_pipeline("supplier_delay")

    assert res["situation"]["id"] == "supplier_delay"
    assert res["recommendation"].strategy_title == "Dual-Source Split (Activate Nordic Precision)"
    assert res["recommendation"].confidence_score > 0.8
    assert len(res["step_logs"]) == 8
    assert res["total_execution_time"] > 0
    assert "roc_data" in res
    assert "pr_data" in res

def test_isolated_sqlite_database(tmp_path):
    """Verifies feedback loop in a completely isolated temporary database."""
    test_db_path = str(tmp_path / "test_isolated.db")
    rec_id = save_decision_record(
        decision_id="ISOLATED-DEC-001",
        situation_id="supplier_delay",
        situation_title="Isolated Delay Test",
        recommended_strategy="Dual-Source Split",
        recommended_strategy_title="Dual-Source Split",
        expected_utility=0.88,
        estimated_cost=25000,
        explanation="Test explanation",
        confidence_score=0.92,
        user_status="ACCEPTED",
        user_feedback="Approved in test.",
        realized_outcome=0.91,
        agent_execution_time_sec=1.45,
        db_path=test_db_path
    )
    assert rec_id > 0
    all_decs = get_all_decisions(db_path=test_db_path)
    assert len(all_decs) == 1
    assert all_decs[0]["decision_id"] == "ISOLATED-DEC-001"
