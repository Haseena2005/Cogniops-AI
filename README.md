# CogniOps AI: A Multi-Agent Explainable Framework for Autonomous Enterprise Decision Planning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Complete%20Prototype-success.svg)]()
[![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)]()

> **Final Year B.E. Computer Science Capstone Project**  
> *Demonstrating an end-to-end multi-agent explainable decision intelligence pipeline for enterprise planning.*

---

## 📌 Executive Summary

Enterprise decision-making frequently suffers from siloed databases, uncalibrated risk models, and black-box decision tools. **CogniOps AI** implements an orchestrated, 8-agent explainable decision intelligence architecture:
1. **Data Ingestion & Quality Auditing** (`DataAgent`)
2. **Predictive Risk Modeling & ML Evaluation** (`AnalyticsAgent` - 5-fold CV, 80/20 test split, ROC/PR curves)
3. **Ontological Supply Chain Knowledge Graph** (`KnowledgeGraphAgent` - 2-hop affected sub-graph with bounded revenue at risk)
4. **Policy Guardrails & Historical RAG Retrieval** (`RAGAgent` - sentence-transformers + FAISS / TF-IDF fallback)
5. **Candidate Strategy Generation & Policy Pruning** (`PlannerAgent`)
6. **1,000-Iteration Monte Carlo What-If Simulation** (`ScenarioAgent`)
7. **TreeSHAP Feature Attribution & Multimodal Rationale** (`ExplainAgent`)
8. **Human-in-the-Loop Feedback & Closed-Loop Reliability** (`CoordinatorAgent` + SQLite)

All operations run **100% locally and offline** with zero paid API dependencies required.

---

## 🏗️ 9-Block Architecture & Modules

```
                      ┌───────────────────────────┐
                      │    Enterprise Ingestion   │
                      │  (Sales, Inv, Supp, Cust) │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │        Data Agent         │
                      │ (Quality Audit: 98.8%)    │
                      └─────────────┬─────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Analytics Agent │      │     KG Agent     │      │    RAG Agent     │
│ (RF Risk Model)  │      │ (Multi-hop Path) │      │ (Policy Retrieval│
└────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                                   ▼
                      ┌───────────────────────────┐
                      │       Planner Agent       │
                      │(Candidate Gen + Guardrails│
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │       Scenario Agent      │
                      │(1,000 Monte Carlo Runs)   │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │      Explain Agent        │
                      │ (SHAP + Natural Language) │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │     Coordinator Agent     │
                      │ (Human Accept/Modify/Rej) │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │   SQLite Decision Store   │
                      │ (Closed-loop Reliability) │
                      └───────────────────────────┘
```

---

## ⚡ Quickstart: Running Locally

### 1. Requirements & Setup
```bash
# Clone or navigate to the project directory
cd "cogniops AI"

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Enterprise Data (Seed 42)
```bash
python data/generate_data.py
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Unit & Integration Tests

All 8 agents, revenue-at-risk bounded invariant, realistic accuracy range, and isolated SQLite testing are covered by pytest:
```bash
pytest tests/test_pipeline.py -v
```

---

## 📸 Exporting Report Figures (Publication-Ready)

To generate all high-resolution PNG charts (dpi=200) for your project report:
```bash
python export_figures.py
```
The figures are saved in `/report_figures`:
- `fig_overview.png`: KPI executive health dashboard
- `fig_confusion_matrix.png`: Confusion matrix of disruption classifier (held-out 80/20 test split)
- `fig_roc_curve.png`: ROC curve and Precision-Recall curve
- `fig_knowledge_graph_subgraph.png`: 2-hop affected sub-graph with entity legend
- `fig_rag_context.png`: Retrieved governance policies, similarity scores, and highlights
- `fig_agent_workflow.png`: Execution latency timeline across all 8 agents
- `fig_scenario_comparison.png`: Payoff comparison across Optimistic, Expected, and Pessimistic worlds
- `fig_shap_explanation.png`: TreeSHAP feature attributions
- `fig_recommendation_card.png`: Final recommendation card, expected utility, and policies satisfied
- `fig_evaluation_comparison.png`: CogniOps AI vs. Threshold-Alert BI Rule & Cheapest Heuristic (20 situations)
- `fig_feedback_trend.png`: Realized outcome vs. expected utility closed-loop reliability

---

## 📊 Empirical Evaluation Benchmark (20 Situations)

| Evaluation Dimension | CogniOps AI Framework | Threshold-Alert BI Rule | Cheapest-Action Heuristic | Significance / Insight |
|---|---|---|---|---|
| **Prediction Accuracy** | **84.4%** (CV: 81.6±1.2%) | 61.2% | 52.0% | Realistic noise calibration; no overfitting |
| **Risk Detection Recall** | **65.3%** (CV: 59.3±3.0%) | 48.5% | 35.0% | Substantially reduces undetected stockouts |
| **Ground-Truth Agreement (20 Situations)** | **100.0% (20/20)** | 0.0% (0/20) | 0.0% (0/20) | Ground-truth simulated independently (10k runs) |
| **Ranking Consistency (Kendall's $\tau$)** | **$\tau = 1.000$** across 5 seeds | $\tau = 0.310$ | $\tau = 0.400$ | Evaluated across distinct random seeds |
| **Explanation Completeness** | **100.0%** | 25.0% | 10.0% | 4-point governance & causal audit |
| **Measured System Latency** | **~1.85 seconds** | Instantaneous | Instantaneous | Autonomous multi-agent cycle |

> **Documented Operational Assumption:** Conventional cross-department manual coordination time is documented as an assumed operational baseline of 4.0 hours (14,400s), not a benchmark computation.  
> **Synthetic Data Declaration:** All datasets (sales, inventory, orders, customers, finance) are synthetically generated (Seed 42). Proof of concept.
