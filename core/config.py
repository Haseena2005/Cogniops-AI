"""
CogniOps AI - Core Configuration Module.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
POLICIES_DIR = DATA_DIR / "policies"
HISTORY_DIR = DATA_DIR / "history"
DB_PATH = BASE_DIR / "core" / "cogniops.db"
REPORT_FIGURES_DIR = BASE_DIR / "report_figures"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
POLICIES_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Random Seed for Deterministic Simulation
RANDOM_SEED = 42

# LLM Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Business Situation Definitions
SITUATIONS = {
    "supplier_delay": {
        "id": "supplier_delay",
        "title": "Supplier Delay Crisis (Apex Global Logistics)",
        "entity_type": "Supplier",
        "entity_id": "SUP-01",
        "description": "Primary Tier-1 microchip and sensor supplier Apex Global Logistics is experiencing severe shipment delays of 22+ days, causing critical order fulfillment risks across high-margin product lines.",
        "severity": "CRITICAL",
        "category": "Supply Chain Bottleneck",
    },
    "sales_decline": {
        "id": "sales_decline",
        "title": "Regional Sales Contraction (North America East)",
        "entity_type": "Region",
        "entity_id": "Region-East",
        "description": "Regional monthly revenues dropped 24.3% over the last 90 days in Region-East driven by competitive undercutting and delayed promotional campaigns.",
        "severity": "HIGH",
        "category": "Market Penetration",
    },
    "inventory_shortage": {
        "id": "inventory_shortage",
        "title": "Critical Stockout Threat (OmniSensor X1)",
        "entity_type": "Product",
        "entity_id": "PROD-01",
        "description": "Warehouse current stock for flagship product PROD-01 dropped to 140 units, breaching safety stock threshold (reorder point: 450 units) amid unexpected enterprise order demand.",
        "severity": "CRITICAL",
        "category": "Inventory Imbalance",
    },
    "customer_churn_spike": {
        "id": "customer_churn_spike",
        "title": "Enterprise Tier Customer Churn Escalation",
        "entity_type": "Segment",
        "entity_id": "Enterprise",
        "description": "Enterprise customer retention fell 18.5% due to SLA breaches on delayed support tickets and missed delivery target commitments.",
        "severity": "HIGH",
        "category": "Customer Retention",
    }
}
