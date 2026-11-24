# CogniOps AI: Project Presentation Notes & Viva Preparation Guide

**Project Title:** CogniOps AI: A Multi-Agent Explainable Framework for Autonomous Enterprise Decision Planning  
**Degree:** Bachelor of Engineering (B.E.) in Computer Science & Engineering  
**Academic Focus:** Decision Intelligence, Multi-Agent Systems (MAS), Applied Machine Learning, Explainable AI (XAI), Operations Research.

---

## 1. Project Overview & Motivation
Enterprise decision-making currently suffers from three fundamental bottlenecks:
1. **Siloed Relational Data:** ERP, CRM, and Supply Chain databases operate in isolation.
2. **Black-box Predictive Models:** Machine learning risk models alert managers about probabilities without explaining causal drivers or recommending actionable interventions.
3. **Fragile Rule-Based Automation:** Conventional BI dashboards rely on static heuristics or manual spreadsheet reviews, which fail when faced with stochastic market fluctuations and regulatory policies.

**CogniOps AI** addresses these gaps through an orchestrated multi-agent framework that ingests raw enterprise data, forecasts disruption risks, extracts topological exposure via Knowledge Graphs, retrieves governance guardrails via RAG, simulates candidate actions under stochastic Monte Carlo regimes, and provides multimodal TreeSHAP explanations for human sign-off.

---

## 2. Technical Stack
- **Dashboard & Interface:** Streamlit (Wide-layout, dark theme, interactive Plotly visualizations).
- **Core Analytics & ML:** scikit-learn (Random Forest with 5-fold Stratified Cross-Validation, ROC & PR curves), NumPy, Pandas.
- **Explainable AI (XAI):** SHAP (TreeSHAP feature importance attributions).
- **Ontology & Knowledge Graph:** NetworkX (multi-relational directed graph: Supplier, Product, Order, Customer, Region, Warehouse).
- **RAG & Policy Engine:** sentence-transformers (`all-MiniLM-L6-v2`) + FAISS index; deterministic TF-IDF Cosine Similarity fallback.
- **Decision Audit Store:** SQLite (Closed-loop feedback tracking $\Delta = \text{realized} - \text{expected}$).
- **LLM Layer:** Unified interface (`core/llm_client.py`) supporting Google Gemini if `GEMINI_API_KEY` is present, or a deterministic offline template generator.

---

## 3. Dataset Architecture & Real-World Disruption Scenarios
Generated using `data/generate_data.py` with reproducible Seed 42:
- **Sales Data (`sales.csv`):** 24 months, 5 geographical regions, 8 distinct industrial products ($60.37M total revenue).
- **Inventory Data (`inventory.csv`):** Current stock levels, reorder points, safety buffers, lead times across Central and Coastal hubs.
- **Supplier Directory (`suppliers.csv`):** Vendor reliability metrics, historical delays, expedite cost multipliers.
- **Customer Database (`customers.csv`):** 120 client accounts partitioned into Enterprise, Mid-Market, SMB, and Government tiers.
- **Orders & Transactions (`orders.csv`):** 250 orders linking clients, products, and fulfillment status.
- **Financial Summary (`finance.csv`):** Monthly COGS, OPEX, and EBITDA.
- **Policy Repository (`data/policies/`):** 9 operational policies (POL-001 to POL-009) enforcing discount ceilings, air freight caps ($25k), and safety stock boundaries.

### Injected Enterprise Disruptions:
1. **Supplier Delay Crisis (Apex Global Logistics):** Delay spikes from 3 to 24 days, jeopardizing $74,800 in downstream client orders.
2. **Regional Sales Contraction (Region-East):** Revenue drops 24.3% over 90 days.
3. **Critical Stockout Threat (OmniSensor X1):** Stock plummets to 140 units (reorder threshold: 450).
4. **Enterprise Churn Spike:** Account churn escalates to 38.0% among enterprise clients.

---

## 4. How Each Agent Works

1. **Data Agent (`agents/data_agent.py`):**
   - Validates data completeness, uniqueness, and schema conformance.
   - Computes a composite Data Quality Score (98.8% Healthy).
2. **Analytics Agent (`agents/analytics_agent.py`):**
   - Trains a Random Forest classifier with realistic enterprise noise and stochastic label overlap.
   - Evaluates performance via 5-fold Stratified Cross-Validation (81.6% ± 1.2%) and held-out 80/20 test split (84.4% accuracy, 65.3% recall, 0.817 ROC-AUC).
3. **Knowledge Graph Agent (`agents/kg_agent.py`):**
   - Traverses 2-hop topological dependencies (Supplier $\rightarrow$ Product $\rightarrow$ Order $\rightarrow$ Customer).
   - Computes downstream revenue at risk, strictly bounded below total period revenue.
4. **RAG Agent (`agents/rag_agent.py`):**
   - Semantically indexes governing policies and historical precedent cases.
   - Returns top matches with similarity scores, source file citations, and highlighted excerpts.
5. **Planner Agent (`agents/planner_agent.py`):**
   - Synthesizes 3-4 viable operational strategies per situation.
   - Evaluates rule-based guardrails and removes non-compliant strategies (recording explicit rejection rationales).
6. **Scenario Agent (`agents/scenario_agent.py`):**
   - Runs 1,000 Monte Carlo runs across Optimistic ($p=0.25$), Expected ($p=0.55$), and Pessimistic ($p=0.20$) macroeconomic states.
   - Computes Expected Utility $U = \sum p_k u_k$, 5th-percentile downside Value-at-Risk (VaR), and net ROI.
7. **Explainability Agent (`agents/explain_agent.py`):**
   - Extracts TreeSHAP values for operational drivers.
   - Generates multimodal explanations citing specific Knowledge Graph facts and policy IDs.
8. **Coordinator Agent (`agents/coordinator.py`):**
   - Sequentially orchestrates the pipeline, logs timing (avg ~1.8s), packages recommendation, and interfaces with the SQLite human feedback loop.

---

## 5. Comparative Evaluation Methodology
CogniOps AI was evaluated across **20 distinct enterprise situation variants** against two fair, computed competitors:
- **Competitor A (Threshold-Alert BI Rule):** Reactive alerting triggering status quo or passive delay absorption.
- **Competitor B (Cheapest-Action Heuristic):** Always selects the lowest initial dollar cost action.
- **Ground Truth:** Independently computed by a 10,000-run Monte Carlo simulation using independent seed (9999) and perturbed payoff distributions.

### Benchmark Highlights:
- **Accuracy:** 84.4% (CogniOps) vs 61.2% (Threshold BI) vs 52.0% (Cheapest).
- **Ground-Truth Agreement (20 Situations):** 100% (CogniOps) vs 0% (Threshold BI) vs 0% (Cheapest).
- **Ranking Stability (Kendall's $\tau$):** $\tau = 1.000$ across distinct random seeds ($p < 0.01$).
- **Explanation Completeness:** 100% (4-point checklist) vs 25% (BI) vs 10% (Cheapest).
- **Latency:** ~1.85 seconds autonomous system time (compared to 4.0 hours documented human analyst assumption).

---

## 6. Known Limitations
1. **Synthetic Data Generation:** Relational data is synthetically generated with Seed 42 for academic reproducibility.
2. **Documented Baseline Latency:** The 4.0-hour baseline represents an operational assumption for human cross-departmental alignment, not a computational benchmark.
3. **Graph Scale:** The prototype models an enterprise supply network with ~250 orders; scaling to millions of live transactions would require distributed graph frameworks like Neo4j or Memgraph.

---

## 7. 15 Likely Viva Questions & Model Answers

### Q1: What is CogniOps AI and why is it needed?
**Answer:** CogniOps AI is an explainable multi-agent decision planning framework designed to bridge the gap between predictive ML models and business execution. Standard ML models only predict probabilities (e.g., "78% risk of stockout"), whereas CogniOps AI identifies root causes, checks corporate policies, simulates stochastic outcomes across alternative strategies, and provides an explainable recommendation for human sign-off.

### Q2: Why did you choose a Multi-Agent architecture instead of a single LLM prompt?
**Answer:** A monolithic LLM prompt suffers from hallucination, lack of deterministic constraint validation, and inability to execute heavy numerical tasks like Monte Carlo simulations or SHAP calculations. By separating concerns into 8 dedicated agents (Data, Analytics, KG, RAG, Planner, Scenario, Explain, Coordinator), each module handles what it is best suited for, making the system verifiable, modular, and audit-compliant.

### Q3: How does the Knowledge Graph contribute to decision making?
**Answer:** Relational tables store data in separate schemas. The Knowledge Graph models multi-relational dependencies: Supplier $\rightarrow$ Product $\rightarrow$ Order $\rightarrow$ Customer $\rightarrow$ Region. When a supplier is delayed, the KG performs multi-hop path traversal to discover exactly which customer accounts, high-margin contracts, and revenue amounts are compromised.

### Q4: Why is the revenue at risk bounded, and how was it tested?
**Answer:** In earlier iterations, unconstrained summation could double-count overlapping order lines. We introduced strict bounding in `kg_agent.py` ensuring downstream exposure cannot exceed relevant periodic revenue ($60.37M total annualized). We verified this with a dedicated unit test in `tests/test_pipeline.py`.

### Q5: How did you prevent the machine learning model from having an unrealistic 100% accuracy?
**Answer:** In initial prototypes, labels were deterministic threshold functions of inputs. In our revised `analytics_agent.py`, we introduced a continuous latent risk probability with Gaussian noise ($\mu=0, \sigma=0.14$), creating non-deterministic class overlap. The Random Forest achieves a realistic 84.4% accuracy (81.6% ± 1.2% across 5-fold cross-validation).

### Q6: What is TreeSHAP and why is it used?
**Answer:** SHAP (Shapley Additive Explanations) uses cooperative game theory to quantify the exact contribution of each input feature toward a model's prediction. TreeSHAP is an optimized algorithm for tree ensembles that calculates exact Shapley values in polynomial time, enabling us to pinpoint whether supplier delay, stock ratio, or lead time was the dominant risk driver.

### Q7: How does the RAG Agent function without external paid APIs?
**Answer:** The RAG agent uses sentence-transformers (`all-MiniLM-L6-v2`) with a FAISS index if installed locally, and automatically falls back to scikit-learn's TF-IDF Vectorizer with Cosine Similarity. Both methods run 100% offline and cost $0.

### Q8: How are policy guardrails enforced in the Planner Agent?
**Answer:** The Planner Agent generates candidate actions and checks them against corporate governance rules retrieved by RAG (e.g., POL-004 caps emergency air freight at $25,000; POL-002 caps discounts at 15%). Strategies that breach these rules are pruned before simulation, and their violation reasons are recorded for audit transparency.

### Q9: How does the Scenario Agent evaluate strategies?
**Answer:** It uses Monte Carlo simulation (1,000 iterations per strategy) across three macroeconomic scenarios: Optimistic (weight 0.25), Expected (weight 0.55), and Pessimistic (weight 0.20). It computes composite Expected Utility $U = \sum p_k u_k$, 5th-percentile downside Value-at-Risk (VaR), and net ROI.

### Q10: How did you define the ground truth for your evaluation benchmark?
**Answer:** To avoid self-reinforcing bias, ground truth was not hardcoded to match the planner. We implemented an independent simulation in `evaluation.py` running 10,000 iterations per scenario with a different random seed (9999) and perturbed payoff distributions, evaluating 20 situational variants.

### Q11: What is Kendall's Tau ($\tau$) and what does it measure here?
**Answer:** Kendall's Tau is a rank correlation coefficient measuring ordinal agreement between two rankings. We measured ranking consistency across 5 different random seeds with stochastic input perturbations. The resulting Tau demonstrated high stochastic stability of the Pareto ranking.

### Q12: What are the two fair competitors used in your benchmark?
**Answer:** 
1. **Threshold-Alert BI Rule:** Simulates standard business intelligence dashboards that flag threshold breaches and adopt reactive, status-quo actions.
2. **Cheapest-Action Heuristic:** Always selects the action with the lowest initial cash outlay, ignoring systemic downstream risks.

### Q13: What does the Closed-Loop Feedback Loop do?
**Answer:** Human decision-makers review the autonomous recommendation and select Accept, Modify, or Reject. The decision is saved to an SQLite database (`core/cogniops.db`). When actual business outcomes are realized, the system records the delta ($\Delta = \text{realized} - \text{expected}$), enabling reliability tracking over time.

### Q14: How does the What-If Sandbox work?
**Answer:** The What-If Sandbox provides interactive sliders for supplier delay multiplier, demand shift %, and emergency budget caps. It re-executes the Scenario Agent live, dynamically re-ranking strategies and testing sensitivity under +/-10% parameter shocks.

### Q15: What would you do differently if deploying this into enterprise production?
**Answer:** 
1. Connect live CDC (Change Data Capture) pipelines from ERP systems (e.g., SAP, Oracle) via Apache Kafka.
2. Migrate the graph store from in-memory NetworkX to a scalable distributed graph database (e.g., Neo4j).
3. Integrate enterprise IAM / SSO (OAuth 2.0 / Okta) for multi-tenant role-based access control.
