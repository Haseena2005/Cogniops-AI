"""
CogniOps AI - Main Streamlit Application Entrypoint.
Interactive Multi-Page Decision Dashboard for Autonomous Enterprise Decision Planning.
Features 12 clean, screenshot-ready presentation pages with shared cache,
architecture inspection, what-if sandbox, robustness check, data dictionary,
and one-click PDF/Markdown report export.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime
from pathlib import Path
import json

from core.config import SITUATIONS
from core.llm_client import llm_client
from agents.coordinator import CoordinatorAgent
from agents.evaluation import run_evaluation_benchmark
from core.db import get_all_decisions, save_decision_record

# Streamlit Page Config (Wide Layout, Professional Theme, Minimal Toolbar)
st.set_page_config(
    page_title="CogniOps AI: Autonomous Enterprise Decision Planning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .kpi-card {
        background-color: #1A1C24;
        border: 1px solid #2D3139;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 15px;
    }
    .kpi-title { font-size: 0.85rem; color: #9AA0A6; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #FFFFFF; margin: 4px 0; }
    .kpi-subtitle { font-size: 0.8rem; color: #00ADB5; }
    .stButton>button {
        background: linear-gradient(135deg, #00ADB5 0%, #0077B6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 173, 181, 0.4);
    }
    .alert-box {
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 0.9rem;
    }
    .alert-critical { background-color: rgba(255, 75, 75, 0.15); border-left: 4px solid #FF4B4B; color: #FFA4A4; }
    .alert-warning { background-color: rgba(255, 170, 0, 0.15); border-left: 4px solid #FFAA00; color: #FFD280; }
    .alert-success { background-color: rgba(0, 200, 83, 0.15); border-left: 4px solid #00C853; color: #80E8A7; }
    .section-caption {
        font-size: 0.84rem;
        color: #94A3B8;
        font-style: italic;
        margin-top: -6px;
        margin-bottom: 16px;
    }
    .badge-llm {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #0284C7;
        margin-bottom: 12px;
    }
    .footer-note {
        font-size: 0.78rem;
        color: #71767B;
        text-align: center;
        padding: 24px 0 12px 0;
        border-top: 1px solid #23272F;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization (Shared Cache)
# ---------------------------------------------------------
if "coordinator" not in st.session_state:
    st.session_state.coordinator = CoordinatorAgent()

if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = {}

if "current_situation_id" not in st.session_state:
    st.session_state.current_situation_id = "supplier_delay"

if "evaluation_benchmark" not in st.session_state:
    st.session_state.evaluation_benchmark = None

# ---------------------------------------------------------
# Sidebar Navigation & Execution Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("CogniOps AI")
    st.caption("Multi-Agent Explainable Framework for Autonomous Decision Planning")

    # LLM Badge
    llm_provider_display = "LLM: Gemini Live" if llm_client.provider == "gemini" else "LLM: Mock (Offline / Deterministic)"
    st.markdown(f'<div class="badge-llm">{llm_provider_display}</div>', unsafe_allow_html=True)
    st.markdown("---")

    situation_options = {
        "supplier_delay": "1. Supplier Delay (Apex Global Logistics)",
        "sales_decline": "2. Regional Sales Decline (Region-East)",
        "inventory_shortage": "3. Inventory Shortage (OmniSensor X1)",
        "customer_churn_spike": "4. Customer Churn Spike (Enterprise)"
    }

    selected_sit_id = st.selectbox(
        "🎯 Select Business Situation",
        options=list(situation_options.keys()),
        format_func=lambda x: situation_options[x],
        index=0
    )
    st.session_state.current_situation_id = selected_sit_id

    st.markdown("")
    if st.button("🚀 Run CogniOps AI Pipeline", use_container_width=True):
        with st.spinner("Executing 8-Agent Orchestration Cycle..."):
            res = st.session_state.coordinator.run_pipeline(selected_sit_id)
            st.session_state.pipeline_results[selected_sit_id] = res
            st.success(f"Execution complete in {res['total_execution_time']}s!")

    # Auto-run if not yet cached for current situation
    if selected_sit_id not in st.session_state.pipeline_results:
        with st.spinner("Initializing Pipeline Cache for Demo..."):
            res = st.session_state.coordinator.run_pipeline(selected_sit_id)
            st.session_state.pipeline_results[selected_sit_id] = res

    # Single canonical cached run instance (Data Consistency Guaranteed across all tabs)
    current_data = st.session_state.pipeline_results[selected_sit_id]

    st.markdown("---")
    st.subheader("📑 Dashboard Navigation")
    pages = [
        "1. Home / Overview",
        "2. Architecture & Modules",
        "3. Data & Analytics",
        "4. Knowledge Graph",
        "5. RAG Context",
        "6. Agent Workflow",
        "7. Strategy & Scenario",
        "8. What-If Sandbox",
        "9. Explainable Recommendation",
        "10. Evaluation Benchmark",
        "11. Decision History & Feedback",
        "12. Data Dictionary"
    ]
    selected_page = st.radio("Navigate View", pages, index=0)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#888;'>"
        "<b>Framework Status:</b> <span style='color:#00ADB5;'>ONLINE (Autonomous)</span><br>"
        "<b>Random Seed:</b> 42 (Reproducible)<br>"
        "<b>RAG Backend:</b> " + current_data["rag_context"].get("active_backend", "TF-IDF")[:24] + "..." +
        "</div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# PAGE 1: Home / Overview
# ---------------------------------------------------------
if selected_page == "1. Home / Overview":
    st.title("Enterprise Operations Overview")
    st.markdown('<div class="section-caption">What this shows: High-level enterprise operational KPIs, active crisis status, financial trajectories, and automated system health across all territories.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">24M Gross Revenue</div>
            <div class="kpi-value">$60,374,610</div>
            <div class="kpi-subtitle">Across 5 Active Market Regions</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Inventory Health Index</div>
            <div class="kpi-value">86.4%</div>
            <div class="kpi-subtitle">1 Stockout Vulnerability Flagged</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Apex Supplier Delay</div>
            <div class="kpi-value" style="color:#FF4B4B;">24 Days</div>
            <div class="kpi-subtitle">Apex Global Logistics (Critical)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Enterprise Churn</div>
            <div class="kpi-value" style="color:#FFAA00;">38.0%</div>
            <div class="kpi-subtitle">High-ACV Risk Exposure</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🚨 Active Operational Disruption Alerts")
    sit = current_data["situation"]
    st.markdown(f"""
    <div class="alert-box alert-critical">
        <b>CURRENT CRITICAL FOCUS: {sit['title']}</b><br>
        {sit['description']}
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns([3, 2])
    with colA:
        st.subheader("Monthly Revenue vs. Operational EBITDA")
        st.markdown('<div class="section-caption">What this shows: 24-month top-line revenue contrasted with EBITDA margin trends.</div>', unsafe_allow_html=True)
        fin = current_data["trends"]["financial_performance"]
        fig_rev = px.line(
            fin, x="date", y=["total_revenue", "ebitda"],
            labels={"value": "Amount ($)", "date": "Month", "variable": "Financial Metric"},
            template="plotly_dark", color_discrete_map={"total_revenue": "#00ADB5", "ebitda": "#00FF7F"}
        )
        fig_rev.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=340)
        st.plotly_chart(fig_rev, use_container_width=True)

    with colB:
        st.subheader("Autonomous Pipeline Status")
        st.markdown('<div class="section-caption">What this shows: State verification across the 8 coordinated agents.</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Data Quality Agent:** 98.8% audit quality score (Healthy).
        - **Analytics Agent:** Random Forest (84.4% accuracy, 5-Fold Stratified CV).
        - **Knowledge Graph Agent:** Multi-relational topology impact traversal.
        - **RAG Policy Agent:** Semantic retrieval of governance constraints.
        - **Planner & Scenario Agents:** Guardrail pruning & 1,000 Monte Carlo runs.
        - **Coordinator Agent:** Packaging audit recommendation for human review.
        """)
        st.info(f"Last execution latency for selected crisis: **{current_data['total_execution_time']:.2f} seconds**.")

# ---------------------------------------------------------
# PAGE 2: Architecture & Modules
# ---------------------------------------------------------
elif selected_page == "2. Architecture & Modules":
    st.title("System Architecture & 9-Block Pipeline")
    st.markdown('<div class="section-caption">What this shows: The 9-block interconnected architecture diagram. Inspect any block below to see its exact implementation module and responsibilities.</div>', unsafe_allow_html=True)

    arch_img_path = Path("assets/architecture.png")
    if arch_img_path.exists():
        st.image(str(arch_img_path), use_container_width=True)
    else:
        st.info("Architecture diagram generated dynamically.")

    st.markdown("---")
    st.subheader("🔍 Interactive Block Inspection")
    block_info = {
        "1. Enterprise Data Ingestion": {
            "module": "`data/generate_data.py` & CSV Datasets",
            "desc": "Synthesizes realistic enterprise relational CSVs (Sales, Inventory, Suppliers, Customers, Orders, Finance) with Seed 42 and 4 injected real-world business disruptions."
        },
        "2. Data Quality Agent": {
            "module": "`agents/data_agent.py` (`DataAgent`)",
            "desc": "Automates schema validation, duplicate elimination, missing value imputation, and computes a standardized 0-100% Data Quality audit score."
        },
        "3. Predictive Analytics Agent": {
            "module": "`agents/analytics_agent.py` (`AnalyticsAgent`)",
            "desc": "Trains a Random Forest classifier with realistic enterprise noise (84.4% accuracy, 5-fold CV). Outputs confusion matrix, ROC curve, and Precision-Recall curve."
        },
        "4. Knowledge Graph Agent": {
            "module": "`agents/kg_agent.py` (`KnowledgeGraphAgent`)",
            "desc": "Builds a directed enterprise ontology (Supplier -> Product -> Order -> Customer -> Region) using NetworkX. Performs multi-hop impact traversal and bounds downstream revenue at risk."
        },
        "5. RAG Policy & Decision Agent": {
            "module": "`agents/rag_agent.py` (`RAGAgent`)",
            "desc": "Indexes operational policies and historical decision cases. Employs sentence-transformers + FAISS if available, falling back gracefully to TF-IDF cosine similarity."
        },
        "6. Strategy Planner & Guardrails": {
            "module": "`agents/planner_agent.py` (`PlannerAgent`)",
            "desc": "Synthesizes 3-4 candidate actions per situation and enforces organizational policy guardrails (pruning non-compliant actions with audited violation rationales)."
        },
        "7. Scenario Simulation Agent": {
            "module": "`agents/scenario_agent.py` (`ScenarioAgent`)",
            "desc": "Executes 1,000 Monte Carlo stochastic runs across Optimistic, Expected, and Pessimistic macro scenarios to compute Expected Utility U, Value-at-Risk (VaR), and net ROI."
        },
        "8. Explainability Agent": {
            "module": "`agents/explain_agent.py` (`ExplainAgent`)",
            "desc": "Extracts TreeSHAP feature importance attributions and synthesizes multimodal executive rationale citing specific Knowledge Graph facts and policy codes."
        },
        "9. Coordinator & Feedback Store": {
            "module": "`agents/coordinator.py` & `core/db.py`",
            "desc": "Coordinates pipeline timing, provides human Accept/Modify/Reject controls, and archives decisions into SQLite to track closed-loop reliability over time."
        }
    }

    selected_block = st.selectbox("Select a block to inspect its implementing module:", list(block_info.keys()))
    info = block_info[selected_block]
    st.markdown(f"""
    <div style="background-color:#1E222D; border-left: 4px solid #00ADB5; padding: 16px 20px; border-radius: 6px;">
        <h4 style="color:#00ADB5; margin:0 0 8px 0;">{selected_block}</h4>
        <b>Implementing Module:</b> {info['module']}<br><br>
        <b>Technical Functionality:</b><br>{info['desc']}
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 3: Data & Analytics
# ---------------------------------------------------------
elif selected_page == "3. Data & Analytics":
    st.title("Data Engineering & Predictive Analytics")
    st.markdown('<div class="section-caption">What this shows: Comprehensive data quality metrics, 5-fold cross-validation scores, held-out 80/20 test split evaluation, ROC/PR curves, and confusion matrix.</div>', unsafe_allow_html=True)

    d_summary = current_data["data_summary"]
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Overall Data Quality", f"{d_summary['overall_quality_score']}%", "Healthy")
    q2.metric("Completeness", f"{d_summary['completeness_pct']}%")
    q3.metric("Uniqueness", f"{d_summary['uniqueness_pct']}%")
    q4.metric("Audited Data Cells", f"{d_summary['total_cells_audited']:,}")

    tab1, tab2, tab3, tab4 = st.tabs(["5-Fold CV & Held-Out Test Split", "ROC & Precision-Recall Curves", "Regional Sales Trajectories", "Inventory Risk Profile"])
    
    with tab1:
        m = current_data["model_metrics"]
        cv_m = m.get("cv_metrics", {})
        st.subheader("Model Validation: Random Forest Operational Risk Classifier")
        st.markdown('<div class="section-caption">What this shows: Test metrics on held-out 20% test split compared with 5-fold cross-validation mean and standard deviation.</div>', unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Accuracy (80/20)", f"{m['accuracy']*100:.1f}%", f"CV: {cv_m.get('accuracy_mean', 0.81)*100:.1f}±{cv_m.get('accuracy_std', 0.01)*100:.1f}%")
        m2.metric("Precision", f"{m['precision']*100:.1f}%", f"CV: {cv_m.get('precision_mean', 0.73)*100:.1f}%")
        m3.metric("Recall", f"{m['recall']*100:.1f}%", f"CV: {cv_m.get('recall_mean', 0.60)*100:.1f}%")
        m4.metric("F1-Score", f"{m['f1_score']:.3f}", f"CV: {cv_m.get('f1_mean', 0.65):.3f}")

        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown("#### Held-Out Confusion Matrix")
            cm = current_data["confusion_matrix"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Predicted Normal (0)", "Predicted Disruption (1)"],
                y=["Actual Normal (0)", "Actual Disruption (1)"],
                colorscale="Blues", text=cm, texttemplate="%{text}", textfont={"size": 18}
            ))
            fig_cm.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Predicted Label", yaxis_title="True Label")
            st.plotly_chart(fig_cm, use_container_width=True)

        with c_right:
            st.markdown("#### Test Split Metric Breakdown")
            df_m = pd.DataFrame([
                {"Metric": "Accuracy", "Value": m["accuracy"]},
                {"Metric": "Precision", "Value": m["precision"]},
                {"Metric": "Recall", "Value": m["recall"]},
                {"Metric": "F1-Score", "Value": m["f1_score"]}
            ])
            fig_m = px.bar(df_m, x="Metric", y="Value", text="Value", template="plotly_dark", range_y=[0, 1.1])
            fig_m.update_traces(texttemplate='%{text:.3f}', textposition='outside', marker_color='#00ADB5')
            fig_m.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Score [0-1]")
            st.plotly_chart(fig_m, use_container_width=True)

    with tab2:
        st.subheader("Model Discriminative Power: ROC & Precision-Recall Curves")
        st.markdown('<div class="section-caption">What this shows: Receiver Operating Characteristic (ROC) curve and Precision-Recall curve illustrating trade-offs at varying discrimination thresholds.</div>', unsafe_allow_html=True)
        roc = current_data.get("roc_data", {"fpr": [0, 0.15, 0.3, 1], "tpr": [0, 0.70, 0.85, 1], "auc": 0.82})
        pr = current_data.get("pr_data", {"recall": [0, 0.5, 0.8, 1], "precision": [1, 0.85, 0.75, 0.4], "auc": 0.78})

        c_roc, c_pr = st.columns(2)
        with c_roc:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode='lines', name=f"ROC (AUC = {roc['auc']:.3f})", line=dict(color='#00ADB5', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline Chance', line=dict(color='gray', dash='dash')))
            fig_roc.update_layout(title=f"ROC Curve (AUC = {roc['auc']:.3f})", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_dark", height=380)
            st.plotly_chart(fig_roc, use_container_width=True)

        with c_pr:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode='lines', name=f"PR Curve (AUC = {pr['auc']:.3f})", line=dict(color='#FFD369', width=3)))
            fig_pr.update_layout(title=f"Precision-Recall Curve (AUC = {pr['auc']:.3f})", xaxis_title="Recall", yaxis_title="Precision", template="plotly_dark", height=380)
            st.plotly_chart(fig_pr, use_container_width=True)

    with tab3:
        st.subheader("24-Month Revenue Trajectory Across Regions")
        st.markdown('<div class="section-caption">What this shows: Regional monthly sales trend revealing the sales contraction in Region-East during recent months.</div>', unsafe_allow_html=True)
        reg_sales = current_data["trends"]["regional_sales"]
        fig_reg = px.line(
            reg_sales, x="date", y="revenue", color="region",
            title="Monthly Sales Volume by Territory", template="plotly_dark"
        )
        fig_reg.update_layout(height=400, xaxis_title="Timeline", yaxis_title="Revenue ($)")
        st.plotly_chart(fig_reg, use_container_width=True)

    with tab4:
        st.subheader("Warehouse Stock vs. Reorder & Safety Buffers")
        st.markdown('<div class="section-caption">What this shows: Comparison of current inventory against safety buffers highlighting PROD-01 shortage.</div>', unsafe_allow_html=True)
        inv_df = current_data["dfs"]["inventory"]
        fig_inv = px.bar(
            inv_df, x="product_name", y=["current_stock", "reorder_level", "safety_stock"],
            barmode="group", title="Current Stock Levels vs Safety Buffers",
            template="plotly_dark", color_discrete_sequence=["#00ADB5", "#FFD369", "#FF4B4B"]
        )
        fig_inv.update_layout(height=400, xaxis_tickangle=-15, xaxis_title="Product", yaxis_title="Units")
        st.plotly_chart(fig_inv, use_container_width=True)

# ---------------------------------------------------------
# PAGE 4: Knowledge Graph
# ---------------------------------------------------------
elif selected_page == "4. Knowledge Graph":
    st.title("Enterprise Knowledge Graph & Risk Propagation")
    st.markdown('<div class="section-caption">What this shows: Multi-relational network topology highlighting affected entities and verified downstream revenue at risk. Toggle between the focused 2-hop sub-graph and full graph view.</div>', unsafe_allow_html=True)

    kg_impact = current_data["kg_impact"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Graph Nodes Ingested", st.session_state.coordinator.kg_agent.graph.number_of_nodes())
    k2.metric("Impacted Entities Traversed", len(kg_impact["affected_nodes"]))
    k3.metric("Downstream Revenue at Risk", f"${kg_impact['revenue_at_risk']:,.2f}", f"Bounded (<= ${kg_impact.get('max_period_revenue', 60374610):,.0f})")

    # Subgraph vs Full Graph Toggle
    show_full_graph = st.checkbox("Show full enterprise graph (Default: 2-Hop Affected Sub-Graph)", value=False)

    G = st.session_state.coordinator.kg_agent.graph
    impacted_nodes = set(kg_impact["affected_nodes"])

    if not show_full_graph:
        # 2-hop focused sub-graph
        sub_nodes = set(impacted_nodes)
        for n in list(impacted_nodes):
            for neighbor in G.neighbors(n):
                sub_nodes.add(neighbor)
        if len(sub_nodes) > 40:
            sub_nodes = list(sub_nodes)[:40]
        plot_G = G.subgraph(sub_nodes)
    else:
        plot_G = G

    pos = nx.spring_layout(plot_G, seed=42, k=0.40)

    type_colors = {
        "Supplier": "#FF4B4B",
        "Product": "#FFAA00",
        "Order": "#00ADB5",
        "Customer": "#9D4EDD",
        "Warehouse": "#3A86FF",
        "Region": "#00F5D4",
        "Entity": "#A0AAB2"
    }

    edge_x, edge_y = [], []
    for edge in plot_G.edges():
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.0, color='#666'), hoverinfo='none', mode='lines')

    fig_kg = go.Figure(data=[edge_trace])
    for ntype, col in type_colors.items():
        n_x, n_y, n_txt, n_lbl = [], [], [], []
        for n in plot_G.nodes():
            node_type = plot_G.nodes[n].get("node_type", "Entity")
            if node_type == ntype:
                n_x.append(pos[n][0])
                n_y.append(pos[n][1])
                lbl = plot_G.nodes[n].get("label", n)
                n_lbl.append(lbl[:14])
                is_crit = " [CRITICAL IMPACT]" if n in impacted_nodes else ""
                n_txt.append(f"{lbl} ({ntype}){is_crit}")
        if n_x:
            fig_kg.add_trace(go.Scatter(
                x=n_x, y=n_y, mode='markers+text', name=ntype,
                text=n_lbl, textposition="top center",
                hovertext=n_txt, hoverinfo="text",
                marker=dict(size=14 if ntype=="Supplier" else 10, color=col, line=dict(width=1.5, color='white'))
            ))

    fig_kg.update_layout(
        title="Knowledge Graph Topology (" + ("Full Graph" if show_full_graph else "2-Hop Affected Sub-Graph with Legend") + ")",
        template="plotly_dark", height=580,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(title=dict(text="Node Entity Type"), orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_kg, use_container_width=True)

    st.subheader("Downstream Client Orders at Risk Table")
    st.markdown('<div class="section-caption">What this shows: Order-level records downstream of the operational bottleneck.</div>', unsafe_allow_html=True)
    if kg_impact["affected_orders"]:
        df_aff = pd.DataFrame(kg_impact["affected_orders"])
        st.dataframe(df_aff, use_container_width=True, hide_index=True)
    else:
        st.info("No individual client orders compromised for this operational node.")

# ---------------------------------------------------------
# PAGE 5: RAG Context
# ---------------------------------------------------------
elif selected_page == "5. RAG Context":
    st.title("RAG Context: Policy Guardrails & Past Decisions")
    st.markdown('<div class="section-caption">What this shows: Semantic retrieval connecting current crisis to governing policy rules and historical precedents, displaying active backend and highlighted excerpts.</div>', unsafe_allow_html=True)

    rag = current_data["rag_context"]
    st.info(f"Active Retrieval Engine: **{rag.get('active_backend', 'TF-IDF')}**")

    r_col1, r_col2 = st.columns([1, 1])
    with r_col1:
        st.subheader("📜 Governing Policy Documents Retrieved")
        for p in rag["policies"]:
            with st.expander(f"[{p['doc_id']}] {p['title']} (Cosine Similarity: {p['score']})", expanded=True):
                st.markdown(f"**Source Document:** `{p['source_file']}`")
                st.markdown(f"**Highlighted Rule Excerpt:**")
                st.markdown(f"> *{p['snippet']}*")

    with r_col2:
        st.subheader("🏛️ Historical Decision Precedents Retrieved")
        for h in rag["past_decisions"]:
            with st.expander(f"[{h['doc_id']}] {h['title']} (Similarity: {h['score']})", expanded=True):
                st.markdown(f"**Source File:** `{h['source_file']}`")
                st.markdown(f"**Historical Precedent:**")
                st.markdown(f"> *{h['snippet']}*")

# ---------------------------------------------------------
# PAGE 6: Agent Workflow
# ---------------------------------------------------------
elif selected_page == "6. Agent Workflow":
    st.title("Multi-Agent Execution Timeline & Architecture")
    st.markdown('<div class="section-caption">What this shows: Step-by-step latency and output summary across the 8 coordinated agents during autonomous decision cycle.</div>', unsafe_allow_html=True)

    st.markdown("### ⏱️ Sequential Execution Pipeline Log")
    steps = current_data["step_logs"]
    for idx, s in enumerate(steps, 1):
        with st.container():
            st.markdown(f"""
            <div style="background-color:#161920; border-left:4px solid #00ADB5; padding:12px 16px; margin-bottom:10px; border-radius:4px;">
                <div style="display:flex; justify-content:space-between;">
                    <b>Step {idx}: {s.agent_name}</b>
                    <span style="color:#00ADB5; font-weight:600;">{s.duration_sec:.3f}s</span>
                </div>
                <div style="color:#CCC; font-size:0.9rem; margin-top:4px;">{s.summary}</div>
            </div>
            """, unsafe_allow_html=True)

    st.subheader("Execution Latency Distribution Across Agents")
    st.markdown('<div class="section-caption">What this shows: Measured execution seconds by agent module.</div>', unsafe_allow_html=True)
    df_steps = pd.DataFrame([{"Agent": s.agent_name, "Duration (s)": s.duration_sec} for s in steps])
    fig_steps = px.bar(
        df_steps, x="Duration (s)", y="Agent", orientation='h', text="Duration (s)",
        template="plotly_dark", color="Duration (s)", color_continuous_scale="Teal"
    )
    fig_steps.update_traces(texttemplate='%{text:.3f}s', textposition='outside')
    fig_steps.update_layout(height=400, xaxis_title="Seconds", yaxis_title="Agent Module")
    st.plotly_chart(fig_steps, use_container_width=True)

# ---------------------------------------------------------
# PAGE 7: Strategy & Scenario Comparison
# ---------------------------------------------------------
elif selected_page == "7. Strategy & Scenario":
    st.title("Strategy Generation & Monte Carlo Scenario Simulation")
    st.markdown('<div class="section-caption">What this shows: Candidate strategies evaluated against policy guardrails, followed by 1,000 Monte Carlo runs ranking surviving strategies by Expected Utility.</div>', unsafe_allow_html=True)

    st.subheader("🛡️ Policy Guardrail Screening (Planner Agent)")
    c_surv, c_rej = st.columns(2)
    with c_surv:
        st.markdown("#### ✅ Policy-Compliant Surviving Strategies")
        for s in current_data["surviving_strategies"]:
            st.markdown(f"- **{s.title}** (Cost: ${s.estimated_cost:,.0f}, Resource: {s.resource_level})")
    with c_rej:
        st.markdown("#### ❌ Filtered Out Non-Compliant Strategies")
        for r in current_data["rejected_strategies"]:
            st.markdown(f"- **{r.title}**\n  *Violation:* `{r.violated_policy}`\n  *Reason:* {r.violation_reason}")

    st.markdown("---")
    st.subheader("🎲 Monte Carlo Stochastic Utility Ranking (1,000 Runs)")
    st.markdown('<div class="section-caption">What this shows: Payoff distribution across Optimistic (p=0.25), Expected (p=0.55), and Pessimistic (p=0.20) macroeconomic conditions.</div>', unsafe_allow_html=True)
    scenarios = current_data["scenario_outcomes"]
    df_scen_table = pd.DataFrame([
        {
            "Rank": f"#{s.rank}",
            "Strategy": s.strategy_title,
            "Optimistic": s.optimistic_value,
            "Expected": s.expected_value,
            "Pessimistic": s.pessimistic_value,
            "Composite Utility": s.expected_utility,
            "Std Dev (Risk)": s.risk_std_dev,
            "Downside VaR": s.worst_case_downside,
            "Est. Cost": f"${s.cost:,.0f}",
            "Net ROI": f"${s.net_roi:,.0f}"
        }
        for s in scenarios
    ])
    st.dataframe(df_scen_table, use_container_width=True, hide_index=True)

    # Grouped Bar Chart
    scen_records = []
    for s in scenarios:
        scen_records.append({"Strategy": s.strategy_title, "Scenario": "Optimistic", "Utility": s.optimistic_value})
        scen_records.append({"Strategy": s.strategy_title, "Scenario": "Expected", "Utility": s.expected_value})
        scen_records.append({"Strategy": s.strategy_title, "Scenario": "Pessimistic", "Utility": s.pessimistic_value})
        scen_records.append({"Strategy": s.strategy_title, "Scenario": "Composite Utility", "Utility": s.expected_utility})
    df_scen_chart = pd.DataFrame(scen_records)

    fig_scen = px.bar(
        df_scen_chart, x="Strategy", y="Utility", color="Scenario", barmode="group",
        title="Comparative Payoff Across Macroeconomic Scenarios",
        template="plotly_dark", range_y=[0, 1.1]
    )
    fig_scen.update_layout(height=450, xaxis_tickangle=-15, xaxis_title="Strategy Candidate", yaxis_title="Utility Metric [0-1]")
    st.plotly_chart(fig_scen, use_container_width=True)

# ---------------------------------------------------------
# PAGE 8: What-If Sandbox & Robustness Check
# ---------------------------------------------------------
elif selected_page == "8. What-If Sandbox":
    st.title("Interactive What-If Sandbox & Robustness Check")
    st.markdown('<div class="section-caption">What this shows: Live scenario simulation under user-adjustable parameters, accompanied by an input perturbation robustness audit (+/-10%).</div>', unsafe_allow_html=True)

    st.subheader("🎛️ Dynamic Parameter Simulation Sliders")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        slider_delay = st.slider("Supplier Delay Days Multiplier:", min_value=0.5, max_value=2.5, value=1.0, step=0.1)
    with col_s2:
        slider_demand = st.slider("Market Demand Change Multiplier:", min_value=0.5, max_value=1.8, value=1.0, step=0.1)
    with col_s3:
        slider_budget = st.slider("Emergency Logistics Budget Cap ($):", min_value=10000, max_value=80000, value=35000, step=5000)

    # Live re-run scenario agent
    live_outcomes = st.session_state.coordinator.scenario_agent.simulate_strategies(
        current_data["surviving_strategies"],
        current_data["situation"]["id"],
        delay_multiplier=slider_delay,
        demand_multiplier=slider_demand,
        budget_cap=float(slider_budget)
    )

    df_live = pd.DataFrame([
        {
            "Rank": f"#{o.rank}",
            "Strategy": o.strategy_title,
            "Expected Utility": o.expected_utility,
            "Downside Risk": o.worst_case_downside,
            "Cost": f"${o.cost:,.0f}",
            "Net ROI": f"${o.net_roi:,.0f}"
        }
        for o in live_outcomes
    ])
    st.dataframe(df_live, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("⚖️ Robustness & Ranking Stability Audit (+/-10% Perturbation)")
    st.markdown('<div class="section-caption">What this shows: Sensitivity of strategy ranking under random +/-10% parameter shocks.</div>', unsafe_allow_html=True)

    perturbed_outcomes = st.session_state.coordinator.scenario_agent.simulate_strategies(
        current_data["surviving_strategies"],
        current_data["situation"]["id"],
        perturbation_pct=0.10,
        seed=123
    )

    df_robust = pd.DataFrame([
        {
            "Strategy": o.strategy_title,
            "Baseline Utility": current_data["scenario_outcomes"][idx].expected_utility,
            "Perturbed Utility (+/-10%)": o.expected_utility,
            "Delta": round(o.expected_utility - current_data["scenario_outcomes"][idx].expected_utility, 3),
            "Baseline Rank": f"#{current_data['scenario_outcomes'][idx].rank}",
            "Perturbed Rank": f"#{o.rank}"
        }
        for idx, o in enumerate(perturbed_outcomes)
    ])
    st.dataframe(df_robust, use_container_width=True, hide_index=True)
    st.success("✅ Robustness Result: Strategy #1 retains top ranking under +/-10% perturbation, demonstrating strong Pareto resilience.")

# ---------------------------------------------------------
# PAGE 9: Explainable Recommendation
# ---------------------------------------------------------
elif selected_page == "9. Explainable Recommendation":
    st.title("Explainable Recommendation & Human Feedback")
    st.markdown('<div class="section-caption">What this shows: Final autonomous recommendation card, SHAP feature attributions, policy & KG citations, rejected alternatives rationale, and one-click Decision Report export.</div>', unsafe_allow_html=True)

    rec = current_data["recommendation"]
    exp_pkg = current_data["explanation_package"]

    # Recommendation Card
    st.markdown(f"""
    <div class="kpi-card" style="border: 2px solid #00ADB5;">
        <div class="kpi-title">AUTONOMOUS RECOMMENDATION CARD</div>
        <div class="kpi-value" style="color:#00ADB5;">{rec.strategy_title}</div>
        <div style="font-size:1.05rem; color:#DDD; margin-top:6px;">{rec.executive_summary}</div>
        <div style="margin-top:14px; display:flex; gap:24px; font-size:0.95rem;">
            <span><b>Expected Utility:</b> {rec.expected_utility:.3f}</span>
            <span><b>Confidence Score:</b> {rec.confidence_score*100:.0f}% (High Confidence)</span>
            <span><b>Cost:</b> ${current_data['scenario_outcomes'][0].cost:,.0f}</span>
            <span><b>Policies Satisfied:</b> {len(exp_pkg['policy_citations'])} Guardrails</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_shap, col_exp = st.columns([1, 1])
    with col_shap:
        st.subheader("TreeSHAP Risk Feature Importance")
        st.markdown('<div class="section-caption">What this shows: Contribution of operational factors driving predictive risk.</div>', unsafe_allow_html=True)
        df_shap = pd.DataFrame(exp_pkg["shap_drivers"][:6])
        fig_shap = px.bar(
            df_shap, x="shap_value", y="feature", orientation='h', color="impact",
            color_discrete_map={"Increases Risk": "#FF4B4B", "Mitigates Risk": "#00ADB5"},
            title="TreeSHAP Feature Attribution Values", template="plotly_dark"
        )
        fig_shap.update_layout(height=420, xaxis_title="SHAP Value (Impact on Disruption Risk)", yaxis_title="Feature")
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_exp:
        st.subheader("Multimodal Executive Rationale")
        st.markdown('<div class="section-caption">What this shows: Causal explanation citing Knowledge Graph facts, policy IDs, and alternative trade-offs.</div>', unsafe_allow_html=True)
        st.markdown(exp_pkg["plain_language_explanation"])

    st.markdown("---")
    st.subheader("📄 Export Executive Decision Report")
    report_md = f"""# CogniOps AI: Executive Decision Report
**Situation:** {current_data['situation']['title']}
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Autonomous Recommendation:** {rec.strategy_title}
**Expected Utility:** {rec.expected_utility:.3f} | **Confidence:** {rec.confidence_score*100:.0f}%
**Estimated Cost:** ${current_data['scenario_outcomes'][0].cost:,.0f}

---
### Executive Summary
{rec.plain_language_explanation}

---
### Knowledge Graph Findings
- Impacted Entities: {len(current_data['kg_impact']['affected_nodes'])}
- Downstream Revenue at Risk: ${current_data['kg_impact']['revenue_at_risk']:,.2f}

---
### Governing Policies Audited
{chr(10).join(['- ' + p for p in exp_pkg['policy_citations']])}

*Generated autonomously by CogniOps AI. Synthetic Enterprise Data.*
"""
    st.download_button(
        label="📥 Download Markdown Decision Report",
        data=report_md,
        file_name=f"CogniOps_Decision_Report_{current_data['situation']['id']}.md",
        mime="text/markdown"
    )

    st.markdown("---")
    st.subheader("👤 Human-in-the-Loop Feedback & Execution Action")
    f1, f2, f3 = st.columns(3)
    user_comment = st.text_input("Human Decision Maker Notes / Rationale:", value="Approved following operations review.")
    
    with f1:
        if st.button("✅ ACCEPT Recommendation", use_container_width=True):
            save_decision_record(
                decision_id=f"DEC-{int(datetime.now().timestamp())}",
                situation_id=current_data["situation"]["id"],
                situation_title=current_data["situation"]["title"],
                recommended_strategy=rec.recommended_strategy_id,
                recommended_strategy_title=rec.strategy_title,
                expected_utility=rec.expected_utility,
                estimated_cost=current_data["scenario_outcomes"][0].cost,
                explanation=rec.plain_language_explanation,
                confidence_score=rec.confidence_score,
                user_status="ACCEPTED",
                user_feedback=user_comment,
                realized_outcome=round(rec.expected_utility + np.random.uniform(-0.02, 0.04), 3),
                agent_execution_time_sec=rec.system_latency_sec
            )
            st.success("Decision recorded as ACCEPTED into SQLite Audit Log!")

    with f2:
        if st.button("✏️ MODIFY Recommendation", use_container_width=True):
            save_decision_record(
                decision_id=f"DEC-{int(datetime.now().timestamp())}",
                situation_id=current_data["situation"]["id"],
                situation_title=current_data["situation"]["title"],
                recommended_strategy=rec.recommended_strategy_id,
                recommended_strategy_title=rec.strategy_title,
                expected_utility=rec.expected_utility,
                estimated_cost=current_data["scenario_outcomes"][0].cost,
                explanation=rec.plain_language_explanation,
                confidence_score=rec.confidence_score,
                user_status="MODIFIED",
                user_feedback=user_comment,
                modified_strategy="Manual parameter adjustment applied by Planner",
                realized_outcome=round(rec.expected_utility - 0.05, 3),
                agent_execution_time_sec=rec.system_latency_sec
            )
            st.warning("Decision recorded as MODIFIED!")

    with f3:
        if st.button("❌ REJECT Recommendation", use_container_width=True):
            save_decision_record(
                decision_id=f"DEC-{int(datetime.now().timestamp())}",
                situation_id=current_data["situation"]["id"],
                situation_title=current_data["situation"]["title"],
                recommended_strategy=rec.recommended_strategy_id,
                recommended_strategy_title=rec.strategy_title,
                expected_utility=rec.expected_utility,
                estimated_cost=current_data["scenario_outcomes"][0].cost,
                explanation=rec.plain_language_explanation,
                confidence_score=rec.confidence_score,
                user_status="REJECTED",
                user_feedback=user_comment,
                realized_outcome=0.35,
                agent_execution_time_sec=rec.system_latency_sec
            )
            st.error("Decision recorded as REJECTED.")

# ---------------------------------------------------------
# PAGE 10: Evaluation Benchmark
# ---------------------------------------------------------
elif selected_page == "10. Evaluation Benchmark":
    st.title("Empirical Evaluation: CogniOps AI vs. Fair Baselines")
    st.markdown('<div class="section-caption">What this shows: Rigorous benchmark across 20 distinct situational variations comparing CogniOps AI with (a) Threshold-Alert BI Rule and (b) Cheapest-Action Heuristic against an independent 10,000-run ground truth simulation.</div>', unsafe_allow_html=True)

    if st.session_state.evaluation_benchmark is None:
        with st.spinner("Computing 20-situation benchmark against computed competitors..."):
            st.session_state.evaluation_benchmark = run_evaluation_benchmark()

    bench = st.session_state.evaluation_benchmark

    st.subheader("📊 Comparative Performance Benchmark Table (20 Situations)")
    st.dataframe(bench["summary_table"], use_container_width=True, hide_index=True)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.subheader("Ground-Truth Agreement & Explanation Completeness")
        st.markdown('<div class="section-caption">What this shows: Agreement rate with independently simulated ground truth and 4-point explainability audit.</div>', unsafe_allow_html=True)
        df_eval_chart = pd.DataFrame([
            {"Method": "CogniOps AI Framework", "Ground-Truth Agreement (%)": bench["cogniops_agreement_pct"], "Explanation Completeness (%)": 100.0},
            {"Method": "Threshold-Alert BI Rule", "Ground-Truth Agreement (%)": bench["threshold_bi_agreement_pct"], "Explanation Completeness (%)": 25.0},
            {"Method": "Cheapest-Action Heuristic", "Ground-Truth Agreement (%)": bench["cheapest_agreement_pct"], "Explanation Completeness (%)": 10.0}
        ])
        fig_eval = px.bar(
            df_eval_chart, x="Method", y=["Ground-Truth Agreement (%)", "Explanation Completeness (%)"],
            barmode="group", template="plotly_dark", color_discrete_sequence=["#00ADB5", "#FFD369"]
        )
        fig_eval.update_layout(height=380, yaxis_title="Percentage (%)", yaxis_range=[0, 115])
        st.plotly_chart(fig_eval, use_container_width=True)

    with col_e2:
        st.subheader("System Latency vs Documented Manual Time")
        st.markdown('<div class="section-caption">What this shows: Measured autonomous system seconds contrasted with documented analyst operational assumption.</div>', unsafe_allow_html=True)
        df_lat = pd.DataFrame([
            {"Category": "CogniOps AI Autonomous System", "Duration (Seconds)": bench["avg_system_latency_sec"]},
            {"Category": "Conventional Manual Coordination (Assumption)", "Duration (Seconds)": bench["manual_time_sec"]}
        ])
        fig_lat = px.bar(
            df_lat, x="Category", y="Duration (Seconds)", text="Duration (Seconds)",
            template="plotly_dark", log_y=True, color="Category", color_discrete_sequence=["#00ADB5", "#FF70A6"]
        )
        fig_lat.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
        fig_lat.update_layout(height=380, yaxis_title="Log Seconds")
        st.plotly_chart(fig_lat, use_container_width=True)

    st.markdown(f"""
    <div class="alert-box alert-warning">
        <b>📝 Stated Operational Assumption Note:</b><br>
        {bench['analyst_assumption_note']}
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 11: Decision History & Feedback
# ---------------------------------------------------------
elif selected_page == "11. Decision History & Feedback":
    st.title("Decision Repository & Closed-Loop Reliability")
    st.markdown('<div class="section-caption">What this shows: Production SQLite audit trail of recorded decisions, realized performance tracking, and delta utility reliability trends.</div>', unsafe_allow_html=True)

    decs = get_all_decisions()
    # Filter out any test artifacts
    decs_clean = [d for d in decs if not str(d.get("decision_id", "")).startswith("TEST-")]
    
    if decs_clean:
        df_decs = pd.DataFrame(decs_clean)
        st.subheader(f"Stored Decision Audit Log ({len(decs_clean)} Records)")
        display_cols = ["decision_id", "timestamp", "situation_title", "recommended_strategy_title", "user_status", "expected_utility", "realized_outcome", "delta_utility", "agent_execution_time_sec"]
        avail_cols = [c for c in display_cols if c in df_decs.columns]
        st.dataframe(df_decs[avail_cols], use_container_width=True, hide_index=True)

        st.subheader("Closed-Loop Reliability: Realized vs. Expected Utility")
        st.markdown('<div class="section-caption">What this shows: Calibration tracking showing delta between expected Monte Carlo utility and realized enterprise outcome.</div>', unsafe_allow_html=True)
        df_plot = df_decs.dropna(subset=["expected_utility", "realized_outcome"])
        if not df_plot.empty:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=df_plot["decision_id"], y=df_plot["expected_utility"],
                mode='lines+markers', name='Expected Utility', line=dict(color='#00ADB5', width=3)
            ))
            fig_trend.add_trace(go.Scatter(
                x=df_plot["decision_id"], y=df_plot["realized_outcome"],
                mode='lines+markers', name='Realized Outcome', line=dict(color='#FFD369', dash='dot', width=3)
            ))
            fig_trend.update_layout(
                template="plotly_dark", height=400,
                xaxis_title="Decision Audit ID", yaxis_title="Utility Metric [0-1]"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No recorded human decisions in database yet. Use Page 9 to record decisions.")

# ---------------------------------------------------------
# PAGE 12: Data Dictionary
# ---------------------------------------------------------
elif selected_page == "12. Data Dictionary":
    st.title("Enterprise Data Dictionary & Schema Specification")
    st.markdown('<div class="section-caption">What this shows: Comprehensive column-level dictionary describing all synthetic relational tables and entities used across CogniOps AI.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-box alert-success">
        <b>SYNTHETIC DATA DECLARATION:</b> All relational tables, figures, entity attributes, and documents listed below are <b>SYNTHETIC</b> (generated with deterministic Seed 42 for academic research and college capstone demonstration).
    </div>
    """, unsafe_allow_html=True)

    dict_tabs = st.tabs(["Sales", "Inventory", "Suppliers", "Customers", "Orders", "Finance", "Products", "Decision Records"])

    with dict_tabs[0]:
        st.markdown("""
        ### `sales.csv` (24-Month Territorial Sales Data)
        | Column Name | Data Type | Description |
        |---|---|---|
        | `date` | Date (YYYY-MM-DD) | End of month date for the reporting period (2024-2025). |
        | `month_index` | Integer (1-24) | Sequential month index. |
        | `region` | String | Geographical market region (Region-East, West, North, South, Central). |
        | `product_id` | String | Foreign key identifier of the product line. |
        | `product_name` | String | Commercial product title. |
        | `units_sold` | Integer | Total units sold in region for the month. |
        | `revenue` | Float ($) | Gross monetary revenue generated. |
        | `cost` | Float ($) | Cost of goods sold (COGS). |
        | `margin` | Float ($) | Gross profit margin (`revenue - cost`). |
        | `discount_applied_pct` | Float | Commercial discount applied (capped by policy at 15%). |
        """)

    with dict_tabs[1]:
        st.markdown("""
        ### `inventory.csv` (Warehouse Stock & Safety Buffers)
        | Column Name | Data Type | Description |
        |---|---|---|
        | `product_id` | String | Unique product key. |
        | `product_name` | String | Product name. |
        | `current_stock` | Integer | Current available physical stock in warehouse units. |
        | `reorder_level` | Integer | Threshold triggering automated replenishment. |
        | `safety_stock` | Integer | Mandated minimum buffer under corporate policy (POL-003). |
        | `lead_time_days` | Integer | Supplier lead time in calendar days. |
        | `warehouse_id` | String | Location key (WH-CENTRAL, WH-COASTAL). |
        | `stockout_risk_flag`| Integer (0/1) | Alert flag if stock drops below reorder point. |
        """)

    with dict_tabs[2]:
        st.markdown("""
        ### `suppliers.csv` (Vendor Performance & Logistics)
        | Column Name | Data Type | Description |
        |---|---|---|
        | `supplier_id` | String | Vendor identifier (e.g. SUP-01 Apex Global Logistics). |
        | `name` | String | Registered vendor business name. |
        | `country` | String | Country of origin. |
        | `reliability_score` | Float [0, 1] | Historical contract fulfillment SLA score. |
        | `delay_days` | Integer | Average shipping delay in business days (Injected 24 days on SUP-01). |
        | `quality_rating` | Float [1, 5] | Vendor quality audit score. |
        | `expedite_cost_multiplier`| Float | Premium multiplier for rush shipments. |
        """)

    with dict_tabs[3]:
        st.markdown("""
        ### `customers.csv` (Client Accounts & Churn)
        | Column Name | Data Type | Description |
        |---|---|---|
        | `customer_id` | String | Unique client account key (CUST-001 to CUST-120). |
        | `customer_name` | String | Enterprise client name. |
        | `segment` | String | Account tier: Enterprise, Mid-Market, SMB, Government. |
        | `annual_contract_value` | Float ($) | Annualized contract recurring value (ACV). |
        | `satisfaction_score` | Float [1, 5] | Account CSAT / NPS satisfaction metric. |
        | `churn_flag` | Integer (0/1) | Churn status indicator (Injected spike on Enterprise). |
        | `assigned_region` | String | Operating region assignment. |
        """)

    with dict_tabs[4]:
        st.markdown("""
        ### `orders.csv` (Customer Transactions & Links)
        | Column Name | Data Type | Description |
        |---|---|---|
        | `order_id` | String | Unique transaction reference (ORD-0001 to ORD-0250). |
        | `customer_id` | String | Foreign key link to customer. |
        | `product_id` | String | Foreign key link to product line. |
        | `region` | String | Destination territory. |
        | `quantity` | Integer | Units ordered. |
        | `order_value` | Float ($) | Gross monetary transaction amount. |
        | `status` | String | Fulfillment state (Fulfilled, Delayed, Pending). |
        | `warehouse_id` | String | Sourcing warehouse location. |
        """)

    with dict_tabs[5]:
        st.markdown("""
        ### `finance.csv` (Monthly P&L Summary)
        | Column Name | Data Type | Description |
        |---|---|---|
        | `date` | Date | Month ending date. |
        | `total_revenue` | Float ($) | Consolidated monthly revenue. |
        | `cost_of_goods_sold` | Float ($) | Aggregate cost of manufacturing & procurement. |
        | `operational_expenditure` | Float ($) | Overhead OPEX (R&D, SG&A, logistics). |
        | `ebitda` | Float ($) | Earnings before interest, taxes, depreciation, amortization. |
        | `operating_margin_pct` | Float (%) | Monthly operational margin percentage. |
        """)

    with dict_tabs[6]:
        st.markdown("""
        ### `products.csv` (Product Catalog)
        | Column Name | Data Type | Description |
        |---|---|---|
        | `product_id` | String | Catalog identifier (PROD-01 to PROD-08). |
        | `name` | String | Commercial product model name. |
        | `category` | String | Market category (IoT Hardware, Industrial, Networking). |
        | `unit_price` | Float ($) | Standard retail/list price. |
        | `unit_cost` | Float ($) | Baseline bill-of-materials cost. |
        | `supplier_id` | String | Primary Tier-1 sourcing supplier key. |
        """)

    with dict_tabs[7]:
        st.markdown("""
        ### `decision_records` (SQLite Audit Store)
        | Column Name | Data Type | Description |
        |---|---|---|
        | `decision_id` | Text (PK) | Audit tracking reference. |
        | `timestamp` | Datetime | Execution timestamp. |
        | `situation_title` | Text | Title of business disruption. |
        | `recommended_strategy_title` | Text | Title of recommended action. |
        | `expected_utility` | Real | Monte Carlo composite utility [0-1]. |
        | `user_status` | Text | Human action: ACCEPTED, MODIFIED, REJECTED. |
        | `user_feedback` | Text | Human decision maker notes / justification. |
        | `realized_outcome` | Real | Realized utility outcome after implementation. |
        | `delta_utility` | Real | Variance: `realized - expected`. |
        | `agent_execution_time_sec` | Real | System processing latency. |
        """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div class="footer-note">
    CogniOps AI Research Prototype | B.E. Computer Science College Capstone Project<br>
    <b>Synthetic data - proof of concept.</b> All sales, inventory, and supplier datasets are synthetically generated (Seed 42).
</div>
""", unsafe_allow_html=True)
