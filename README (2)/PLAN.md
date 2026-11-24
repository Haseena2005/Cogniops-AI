# CogniOps AI Architecture and Implementation Plan

## Overview
CogniOps AI is a multi-agent explainable decision framework designed for autonomous enterprise planning.
It coordinates 8 specialized agents to transform raw enterprise signals into explainable, constraint-validated, and risk-simulated strategy recommendations.

## Pipeline Architecture
1. **Data Agent (`agents/data_agent.py`)**: Data ingestion, cleaning, schema validation, anomaly detection, data quality score.
2. **Analytics Agent (`agents/analytics_agent.py`)**: Feature extraction, trend analysis, predictive risk modeling (RandomForest / GradientBoosting), test evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
3. **Knowledge Graph Agent (`agents/kg_agent.py`)**: Entity-relationship ontology (Supplier, Product, Order, Customer, Region, Warehouse) with multi-hop impact traversal and revenue-at-risk calculation. Visualized with Plotly interactive graph.
4. **RAG Agent (`agents/rag_agent.py`)**: Vector/TF-IDF semantic indexing of enterprise policy documents and historical decisions with similarity scoring.
5. **Planner Agent (`agents/planner_agent.py`)**: Candidate strategy generation (3-4 actions) + policy constraint filter / guardrail evaluator with rejected rationales.
6. **Scenario Agent (`agents/scenario_agent.py`)**: Monte Carlo simulation (1,000 runs) across Optimistic, Expected, and Pessimistic worlds, expected utility computation, and Pareto ranking.
7. **Explainability Agent (`agents/explain_agent.py`)**: SHAP feature importance attribution + structured natural language rationale citing KG facts and retrieved policies.
8. **Coordinator Agent (`agents/coordinator.py`)**: Orchestrates the sequential and dependency graph, measures step latency, aggregates state, and accepts human decision actions (Accept/Modify/Reject).
9. **Decision Store & Feedback (`core/db.py`)**: SQLite audit trail recording decisions, feedback status, realized outcomes, and reliability tracking over time.

## Evaluation & Baseline
- Comparison against Conventional BI / Manual Heuristic baseline (static rule-of-thumb picking cheapest default action + fixed analyst latency).
- Metrics: Prediction Accuracy, Risk Recall, Simulator Ground-truth Agreement, Kendall-tau Ranking Consistency, Explanation Completeness Score, System Decision Time.
