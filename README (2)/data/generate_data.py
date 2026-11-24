"""
CogniOps AI - Synthetic Enterprise Data Generator.
Creates realistic enterprise CSV datasets (Seed 42), policy documents, and decision history.
All generated data is strictly SYNTHETIC for demonstration and research benchmarking.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from core.config import DATA_DIR, POLICIES_DIR, HISTORY_DIR, RANDOM_SEED

def generate_enterprise_data():
    np.random.seed(RANDOM_SEED)
    
    # 1. Products (8 products)
    products = [
        {"product_id": "PROD-01", "name": "OmniSensor X1", "category": "IoT Hardware", "unit_price": 450.0, "unit_cost": 210.0, "supplier_id": "SUP-01"},
        {"product_id": "PROD-02", "name": "OptiFlow Valve", "category": "Industrial Valves", "unit_price": 780.0, "unit_cost": 390.0, "supplier_id": "SUP-01"},
        {"product_id": "PROD-03", "name": "CyberShield Gateway", "category": "Edge Security", "unit_price": 1200.0, "unit_cost": 650.0, "supplier_id": "SUP-02"},
        {"product_id": "PROD-04", "name": "AeroDrive Motor 500", "category": "Motors & Drives", "unit_price": 850.0, "unit_cost": 420.0, "supplier_id": "SUP-02"},
        {"product_id": "PROD-05", "name": "DataCore Telemetry Hub", "category": "Networking", "unit_price": 620.0, "unit_cost": 310.0, "supplier_id": "SUP-03"},
        {"product_id": "PROD-06", "name": "ThermalGuard Sensor", "category": "IoT Hardware", "unit_price": 320.0, "unit_cost": 140.0, "supplier_id": "SUP-03"},
        {"product_id": "PROD-07", "name": "VoltPulse Relay", "category": "Power Modules", "unit_price": 260.0, "unit_cost": 115.0, "supplier_id": "SUP-04"},
        {"product_id": "PROD-08", "name": "Synapse Neural Unit", "category": "Embedded AI", "unit_price": 1550.0, "unit_cost": 820.0, "supplier_id": "SUP-04"}
    ]
    df_products = pd.DataFrame(products)
    df_products.to_csv(DATA_DIR / "products.csv", index=False)

    # 2. Suppliers (4 suppliers)
    suppliers = [
        {"supplier_id": "SUP-01", "name": "Apex Global Logistics", "country": "Taiwan", "reliability_score": 0.62, "delay_days": 24, "quality_rating": 4.1, "expedite_cost_multiplier": 1.45}, # INJECTED DELAY
        {"supplier_id": "SUP-02", "name": "Nordic Precision Components", "country": "Germany", "reliability_score": 0.94, "delay_days": 2, "quality_rating": 4.8, "expedite_cost_multiplier": 1.25},
        {"supplier_id": "SUP-03", "name": "Pacifica Electronics Corp", "country": "South Korea", "reliability_score": 0.89, "delay_days": 4, "quality_rating": 4.5, "expedite_cost_multiplier": 1.30},
        {"supplier_id": "SUP-04", "name": "Astra Dynamic Systems", "country": "United States", "reliability_score": 0.91, "delay_days": 3, "quality_rating": 4.7, "expedite_cost_multiplier": 1.35}
    ]
    df_suppliers = pd.DataFrame(suppliers)
    df_suppliers.to_csv(DATA_DIR / "suppliers.csv", index=False)

    # 3. Inventory (8 records with INJECTED SHORTAGE on PROD-01)
    inventory = []
    for p in products:
        pid = p["product_id"]
        if pid == "PROD-01":
            stock = 140  # Below reorder level 450!
            reorder = 450
            lead_time = 32 # High due to SUP-01
        elif pid == "PROD-02":
            stock = 210
            reorder = 300
            lead_time = 28
        else:
            stock = np.random.randint(400, 950)
            reorder = np.random.randint(200, 350)
            lead_time = np.random.randint(7, 16)
            
        inventory.append({
            "product_id": pid,
            "product_name": p["name"],
            "current_stock": stock,
            "reorder_level": reorder,
            "safety_stock": int(reorder * 0.6),
            "lead_time_days": lead_time,
            "warehouse_id": "WH-CENTRAL" if pid in ["PROD-01", "PROD-02", "PROD-03"] else "WH-COASTAL",
            "stockout_risk_flag": 1 if stock < reorder else 0
        })
    df_inventory = pd.DataFrame(inventory)
    df_inventory.to_csv(DATA_DIR / "inventory.csv", index=False)

    # 4. Customers (120 customers, 4 segments)
    segments = ["Enterprise", "Mid-Market", "SMB", "Government"]
    customers = []
    for i in range(1, 121):
        cid = f"CUST-{i:03d}"
        seg = np.random.choice(segments, p=[0.25, 0.35, 0.30, 0.10])
        # INJECTED CHURN SPIKE in Enterprise
        if seg == "Enterprise":
            churn_prob = 0.38 # High churn spike
        else:
            churn_prob = 0.08
            
        churned = 1 if np.random.rand() < churn_prob else 0
        satisfaction = np.random.uniform(2.2, 3.8) if churned else np.random.uniform(3.9, 4.9)
        customers.append({
            "customer_id": cid,
            "customer_name": f"Acme Client {i}",
            "segment": seg,
            "annual_contract_value": float(np.random.randint(25000, 350000)),
            "satisfaction_score": round(satisfaction, 2),
            "churn_flag": churned,
            "assigned_region": np.random.choice(["Region-East", "Region-West", "Region-North", "Region-South", "Region-Central"])
        })
    df_customers = pd.DataFrame(customers)
    df_customers.to_csv(DATA_DIR / "customers.csv", index=False)

    # 5. Sales (24 Months across 5 regions & 8 products)
    regions = ["Region-East", "Region-West", "Region-North", "Region-South", "Region-Central"]
    dates = pd.date_range(start="2024-01-01", periods=24, freq="ME")
    sales_rows = []
    
    for dt in dates:
        month_idx = dt.month + (dt.year - 2024) * 12
        for reg in regions:
            for p in products:
                base_units = np.random.randint(45, 130)
                
                # INJECTED REGIONAL SALES DECLINE in Region-East over recent months (months 21-24)
                if reg == "Region-East" and month_idx >= 21:
                    decline_factor = 1.0 - (0.08 * (month_idx - 20)) # drops ~24%
                    base_units = int(base_units * decline_factor)
                
                # If product is PROD-01 in recent months, supply bottleneck dampens units sold
                if p["product_id"] == "PROD-01" and month_idx >= 22:
                    base_units = int(base_units * 0.65)

                unit_price = p["unit_price"]
                revenue = round(base_units * unit_price, 2)
                cost = round(base_units * p["unit_cost"], 2)
                margin = round(revenue - cost, 2)
                
                sales_rows.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "month_index": month_idx,
                    "region": reg,
                    "product_id": p["product_id"],
                    "product_name": p["name"],
                    "units_sold": base_units,
                    "revenue": revenue,
                    "cost": cost,
                    "margin": margin,
                    "discount_applied_pct": round(np.random.uniform(0.02, 0.12), 3)
                })
    df_sales = pd.DataFrame(sales_rows)
    df_sales.to_csv(DATA_DIR / "sales.csv", index=False)

    # 6. Finance Summary by Month
    finance_rows = []
    for dt, group in df_sales.groupby("date"):
        rev = group["revenue"].sum()
        cst = group["cost"].sum()
        opex = rev * 0.22 # operational overhead
        ebitda = rev - cst - opex
        finance_rows.append({
            "date": dt,
            "total_revenue": round(rev, 2),
            "cost_of_goods_sold": round(cst, 2),
            "operational_expenditure": round(opex, 2),
            "ebitda": round(ebitda, 2),
            "operating_margin_pct": round((ebitda / rev) * 100, 2)
        })
    df_finance = pd.DataFrame(finance_rows)
    df_finance.to_csv(DATA_DIR / "finance.csv", index=False)

    # 7. Customer Orders for Knowledge Graph Links (250 orders)
    orders = []
    for i in range(1, 251):
        cust = df_customers.sample(1, random_state=i).iloc[0]
        prod = df_products.sample(1, random_state=i*2).iloc[0]
        qty = int(np.random.randint(5, 45))
        orders.append({
            "order_id": f"ORD-{i:04d}",
            "customer_id": cust["customer_id"],
            "product_id": prod["product_id"],
            "region": cust["assigned_region"],
            "quantity": qty,
            "order_value": round(qty * prod["unit_price"], 2),
            "status": "Delayed" if prod["supplier_id"] == "SUP-01" and np.random.rand() < 0.65 else "Fulfilled",
            "warehouse_id": "WH-CENTRAL" if prod["product_id"] in ["PROD-01", "PROD-02", "PROD-03"] else "WH-COASTAL"
        })
    df_orders = pd.DataFrame(orders)
    df_orders.to_csv(DATA_DIR / "orders.csv", index=False)

    print("Synthetic CSV datasets successfully generated.")

def generate_policy_documents():
    """Generates 9 realistic enterprise operational policy documents."""
    policies = {
        "POL-001_Supplier_Switching.txt": (
            "POLICY CODE: POL-001\n"
            "TITLE: Supplier Switching & Dual-Sourcing Governance\n"
            "CATEGORY: Procurement & Supply Chain\n"
            "RULES:\n"
            "1. Switching tier-1 component suppliers requires VP Procurement sign-off if financial exposure exceeds $50,000.\n"
            "2. Expedited onboarding of secondary vendors is approved when existing supplier delay exceeds 14 business days.\n"
            "3. Secondary supplier quality rating must be at least 4.2 out of 5.0.\n"
            "4. Expedited shipping surcharge multiplier is strictly capped at 1.5x standard logistics baseline cost."
        ),
        "POL-002_Discounting_Limits.txt": (
            "POLICY CODE: POL-002\n"
            "TITLE: Maximum Promotional Discounting & Pricing Guardrails\n"
            "CATEGORY: Sales & Commercial Operations\n"
            "RULES:\n"
            "1. Maximum standard sales discount cannot exceed 15.0% without CFO authorization.\n"
            "2. Promotional discount campaigns aiming to reverse regional sales drops are limited to a 90-day execution window.\n"
            "3. Price reductions must never cause product gross profit margin to drop below 28.0%.\n"
            "4. Bundled discounting across product lines requires automated margin contribution validation."
        ),
        "POL-003_Safety_Stock_Thresholds.txt": (
            "POLICY CODE: POL-003\n"
            "TITLE: Minimum Inventory Safety Stock & Replenishment Policy\n"
            "CATEGORY: Inventory Management\n"
            "RULES:\n"
            "1. Warehouse stock for category 'IoT Hardware' must maintain at least 60% of reorder level as non-negotiable safety buffer.\n"
            "2. Inter-warehouse transfers between WH-CENTRAL and WH-COASTAL are mandatory when stock falls below safety buffer.\n"
            "3. Autonomous stock rebalancing must not deplete the donor warehouse beyond 40% of its regional baseline.\n"
            "4. Priority fulfillment is enforced for Enterprise segment customer orders when inventory is constrained."
        ),
        "POL-004_Expedited_Logistics_Budget.txt": (
            "POLICY CODE: POL-004\n"
            "TITLE: Emergency Expedited Freight Expenditure Limits\n"
            "CATEGORY: Logistics & Freight\n"
            "RULES:\n"
            "1. Emergency air freight authorizations are capped at $25,000 per single procurement event.\n"
            "2. Expedited freight is only authorized for orders with contract penalties exceeding air freight premiums.\n"
            "3. Cumulative monthly emergency shipping variances must not breach $75,000 across all warehouses.\n"
            "4. Carbon offset requirements apply to all expedited cargo lanes."
        ),
        "POL-005_Customer_SLA_Escalation.txt": (
            "POLICY CODE: POL-005\n"
            "TITLE: Enterprise Account SLA Breach & Customer Retention Protocol\n"
            "CATEGORY: Customer Success & Accounts\n"
            "RULES:\n"
            "1. Any enterprise customer with satisfaction below 3.5 or delayed orders must be assigned an Executive Sponsor within 48 hours.\n"
            "2. Retention concession credits are authorized up to 10.0% of Annual Contract Value (ACV) to prevent churn.\n"
            "3. Contract termination penalties may be waived if delay exceeds 20 business days due to supplier bottleneck.\n"
            "4. Proactive SLA delivery credits must be credited before billing cycles close."
        ),
        "POL-006_Regional_Marketing_Reallocation.txt": (
            "POLICY CODE: POL-006\n"
            "TITLE: Regional Demand Stimulation & Ad-Spend Reallocation\n"
            "CATEGORY: Marketing\n"
            "RULES:\n"
            "1. Marketing reallocation to depressed sales territories is capped at $40,000 per quarter.\n"
            "2. In-region lead generation campaigns must demonstrate a projected payback ratio of at least 2.5x revenue lift.\n"
            "3. Channel marketing subsidies require joint commitment from certified regional distribution partners.\n"
            "4. Budget cannibalization from growing regions must not exceed 12% of donor region marketing capital."
        ),
        "POL-007_Capital_Allocation_Risk_Tolerance.txt": (
            "POLICY CODE: POL-007\n"
            "TITLE: Enterprise Operational Risk Tolerance & Capital Protection\n"
            "CATEGORY: Corporate Governance\n"
            "RULES:\n"
            "1. Any operational strategy with worst-case downside exceeding $100,000 requires Chief Risk Officer approval.\n"
            "2. Expected utility calculations must incorporate a 95% Value-at-Risk (VaR) penalty parameter.\n"
            "3. Autonomous recommendation pipelines must present at least one conservative fallback strategy.\n"
            "4. Human-in-the-loop signoff is mandatory for all irreversible contractual commitments."
        ),
        "POL-008_Vendor_Diversification_Mandate.txt": (
            "POLICY CODE: POL-008\n"
            "TITLE: Single-Source Vendor Mitigation & Resilience Mandate\n"
            "CATEGORY: Strategic Sourcing\n"
            "RULES:\n"
            "1. No single supplier may account for more than 45% of critical bill-of-materials components without dual-sourcing contracts.\n"
            "2. Secondary supplier qualification audits must be refreshed every 180 days.\n"
            "3. Standby allocation agreements must guarantee 30% volume delivery within 10 calendar days of invocation.\n"
            "4. Foreign currency hedging must accompany international component re-routing."
        ),
        "POL-009_Product_End_Of_Life_Substitution.txt": (
            "POLICY CODE: POL-009\n"
            "TITLE: Component Substitution & Product Cross-Compatibility Guidelines\n"
            "CATEGORY: Engineering & Manufacturing\n"
            "RULES:\n"
            "1. Substituting micro-controllers or sensors with alternative part numbers requires QA automated bench-test clearance.\n"
            "2. Direct pin-compatible substitutions are permissible during stockouts if performance variance is under 3%.\n"
            "3. Customer notification is mandatory if component revision changes exterior casing dimensions or firmware signatures.\n"
            "4. Scrap allowances for replaced legacy inventory are capped at $15,000 per incident."
        )
    }

    for filename, content in policies.items():
        with open(POLICIES_DIR / filename, "w", encoding="utf-8") as f:
            f.write(content)

    print("Policy documents written.")

def generate_history_records():
    """Generates 18 realistic past enterprise decision records with realized outcomes."""
    histories = [
        {
            "record_id": "DEC-HIST-001",
            "situation_id": "supplier_delay",
            "situation_title": "Apex Logistics Port Congestion",
            "date": "2024-04-12",
            "applied_strategy": "Dual-Source Split (Activate Nordic Precision)",
            "expected_utility": 0.88,
            "estimated_cost": 28000,
            "realized_outcome": 0.92,
            "outcome_status": "SUCCESS",
            "delta_utility": 0.04,
            "summary": "Transferred 40% of microchip procurement to Nordic Precision; avoided $145,000 order cancellation penalty."
        },
        {
            "record_id": "DEC-HIST-002",
            "situation_id": "supplier_delay",
            "situation_title": "Customs Hold-up on Microchip Lot",
            "date": "2024-06-18",
            "applied_strategy": "Expedited Air Freight + Priority Buffer",
            "expected_utility": 0.81,
            "estimated_cost": 22000,
            "realized_outcome": 0.79,
            "outcome_status": "SUCCESS",
            "delta_utility": -0.02,
            "summary": "Chartered expedited cargo flight to bypass dock strikes; critical line resumed production within 5 days."
        },
        {
            "record_id": "DEC-HIST-003",
            "situation_id": "supplier_delay",
            "situation_title": "Apex Tier-1 Yield Decline",
            "date": "2024-08-04",
            "applied_strategy": "Status Quo (Wait for Batch Completion)",
            "expected_utility": 0.42,
            "estimated_cost": 0,
            "realized_outcome": 0.28,
            "outcome_status": "FAILURE",
            "delta_utility": -0.14,
            "summary": "Management opted to wait; delay extended by 18 days leading to loss of 3 enterprise clients."
        },
        {
            "record_id": "DEC-HIST-004",
            "situation_id": "sales_decline",
            "situation_title": "Mid-Atlantic Market Share Loss",
            "date": "2024-05-10",
            "applied_strategy": "12% Targeted Promotional Incentive & Sales Blitz",
            "expected_utility": 0.79,
            "estimated_cost": 32000,
            "realized_outcome": 0.84,
            "outcome_status": "SUCCESS",
            "delta_utility": 0.05,
            "summary": "Targeted pricing stimulus combined with regional distributor webinars reversed sales drop within 45 days."
        },
        {
            "record_id": "DEC-HIST-005",
            "situation_id": "sales_decline",
            "situation_title": "Eastern Territory Channel Defection",
            "date": "2024-07-22",
            "applied_strategy": "Distributor Margin Rebate Restructuring",
            "expected_utility": 0.75,
            "estimated_cost": 25000,
            "realized_outcome": 0.73,
            "outcome_status": "SUCCESS",
            "delta_utility": -0.02,
            "summary": "Realigned commission structure for tier-1 distributors in East territory; recovered 85% of target volume."
        },
        {
            "record_id": "DEC-HIST-006",
            "situation_id": "sales_decline",
            "situation_title": "Northeast Slump In Demand",
            "date": "2024-09-14",
            "applied_strategy": "Aggressive 25% Blanket Price Slash",
            "expected_utility": 0.65,
            "estimated_cost": 65000,
            "realized_outcome": 0.45,
            "outcome_status": "FAILURE",
            "delta_utility": -0.20,
            "summary": "Policy violation (exceeded 15% discount limit); eroded gross margin without lasting volume recovery."
        },
        {
            "record_id": "DEC-HIST-007",
            "situation_id": "inventory_shortage",
            "situation_title": "Central Hub OmniSensor Stockout Threat",
            "date": "2024-03-19",
            "applied_strategy": "Inter-Warehouse Emergency Stock Rebalance",
            "expected_utility": 0.89,
            "estimated_cost": 12500,
            "realized_outcome": 0.91,
            "outcome_status": "SUCCESS",
            "delta_utility": 0.02,
            "summary": "Reallocated 320 units from Coastal distribution warehouse to Central hub; 0 order SLA breaches."
        },
        {
            "record_id": "DEC-HIST-008",
            "situation_id": "inventory_shortage",
            "situation_title": "OptiFlow Valve Supply Crunch",
            "date": "2024-07-02",
            "applied_strategy": "Selective Order Rationing & Backorder Guarantees",
            "expected_utility": 0.72,
            "estimated_cost": 8000,
            "realized_outcome": 0.74,
            "outcome_status": "SUCCESS",
            "delta_utility": 0.02,
            "summary": "Prioritized high-ACV enterprise orders while offering partial discount credits for SMB delayed shipments."
        },
        {
            "record_id": "DEC-HIST-009",
            "situation_id": "inventory_shortage",
            "situation_title": "CyberShield Stock Depletion",
            "date": "2024-10-11",
            "applied_strategy": "Emergency Spot Market Component Purchase",
            "expected_utility": 0.83,
            "estimated_cost": 34000,
            "realized_outcome": 0.80,
            "outcome_status": "SUCCESS",
            "delta_utility": -0.03,
            "summary": "Purchased 200 units on secondary spot market at 35% premium; protected key enterprise renewal."
        },
        {
            "record_id": "DEC-HIST-010",
            "situation_id": "customer_churn_spike",
            "situation_title": "Enterprise Tier Service Degradation",
            "date": "2024-04-28",
            "applied_strategy": "Executive Sponsor Outreach + 8% Retention Credit",
            "expected_utility": 0.86,
            "estimated_cost": 38000,
            "realized_outcome": 0.88,
            "outcome_status": "SUCCESS",
            "delta_utility": 0.02,
            "summary": "Deployed technical account managers to top 15 accounts; churn dropped from 34% back to baseline 7%."
        },
        {
            "record_id": "DEC-HIST-011",
            "situation_id": "customer_churn_spike",
            "situation_title": "Mid-Market Dissatisfaction Spike",
            "date": "2024-06-30",
            "applied_strategy": "Extended Warranty & Premium SLA Upgrade",
            "expected_utility": 0.78,
            "estimated_cost": 21000,
            "realized_outcome": 0.76,
            "outcome_status": "SUCCESS",
            "delta_utility": -0.02,
            "summary": "Offered free 12-month tier-2 support upgrades; retained 88% of renewal contract pool."
        },
        {
            "record_id": "DEC-HIST-012",
            "situation_id": "customer_churn_spike",
            "situation_title": "SMB Cancellation Cluster",
            "date": "2024-08-19",
            "applied_strategy": "Automated Survey & Inaction",
            "expected_utility": 0.45,
            "estimated_cost": 1500,
            "realized_outcome": 0.31,
            "outcome_status": "FAILURE",
            "delta_utility": -0.14,
            "summary": "Automated email apology failed to address delivery dissatisfaction; 22 SMB clients churned."
        },
        {
            "record_id": "DEC-HIST-013",
            "situation_id": "supplier_delay",
            "situation_title": "Raw Silicon Wafer Quota Delays",
            "date": "2024-09-05",
            "applied_strategy": "Dual-Source Split (Pacifica Electronics)",
            "expected_utility": 0.85,
            "estimated_cost": 31000,
            "realized_outcome": 0.89,
            "outcome_status": "SUCCESS",
            "delta_utility": 0.04,
            "summary": "Rerouted wafer orders to Pacifica Electronics; bridged supply gap with minimal yield differential."
        },
        {
            "record_id": "DEC-HIST-014",
            "situation_id": "sales_decline",
            "situation_title": "Southern Region Seasonal Dip",
            "date": "2024-10-25",
            "applied_strategy": "Regional Roadshow & Partner Enablement",
            "expected_utility": 0.80,
            "estimated_cost": 26000,
            "realized_outcome": 0.82,
            "outcome_status": "SUCCESS",
            "delta_utility": 0.02,
            "summary": "Co-op advertising spend with southern regional integrators revived project pipeline."
        },
        {
            "record_id": "DEC-HIST-015",
            "situation_id": "inventory_shortage",
            "situation_title": "Holiday Buffer Depletion PROD-05",
            "date": "2024-11-15",
            "applied_strategy": "Inter-Warehouse Emergency Stock Rebalance",
            "expected_utility": 0.87,
            "estimated_cost": 14000,
            "realized_outcome": 0.86,
            "outcome_status": "SUCCESS",
            "delta_utility": -0.01,
            "summary": "Relocated safety stock smoothly; fulfilled all Q4 government compliance deliveries."
        },
        {
            "record_id": "DEC-HIST-016",
            "situation_id": "customer_churn_spike",
            "situation_title": "Contract Renegotiation Discontent",
            "date": "2024-11-28",
            "applied_strategy": "Executive Sponsor Outreach + Multi-Year Cap",
            "expected_utility": 0.84,
            "estimated_cost": 29000,
            "realized_outcome": 0.87,
            "outcome_status": "SUCCESS",
            "delta_utility": 0.03,
            "summary": "Structured multi-year price locks for key enterprise accounts in exchange for renewal commitment."
        },
        {
            "record_id": "DEC-HIST-017",
            "situation_id": "supplier_delay",
            "situation_title": "Semiconductor Packaging Line Slowdown",
            "date": "2024-12-10",
            "applied_strategy": "Expedited Air Freight + Priority Buffer",
            "expected_utility": 0.82,
            "estimated_cost": 24000,
            "realized_outcome": 0.81,
            "outcome_status": "SUCCESS",
            "delta_utility": -0.01,
            "summary": "Activated priority logistics lane; secured arrival before year-end production freeze."
        },
        {
            "record_id": "DEC-HIST-018",
            "situation_id": "sales_decline",
            "situation_title": "Central Region Enterprise Freeze",
            "date": "2024-12-20",
            "applied_strategy": "Targeted Solution Bundling",
            "expected_utility": 0.77,
            "estimated_cost": 19000,
            "realized_outcome": 0.75,
            "outcome_status": "SUCCESS",
            "delta_utility": -0.02,
            "summary": "Bundled hardware with cyber-security gateways to boost deal value and shorten sales cycle."
        }
    ]

    for record in histories:
        filename = f"{record['record_id']}.json"
        with open(HISTORY_DIR / filename, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    print("Historical decision records written.")

if __name__ == "__main__":
    generate_enterprise_data()
    generate_policy_documents()
    generate_history_records()
    print("All enterprise synthetic data generation completed successfully.")
