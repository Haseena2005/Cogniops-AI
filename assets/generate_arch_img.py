"""
Generates clean 9-block architecture diagram for CogniOps AI.
Saves to assets/architecture.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_architecture_image():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    # Blocks definition
    blocks = [
        # (id, name, desc, x, y, width, height, color)
        ("Data Ingestion", "1. Enterprise Ingestion", "Sales, Inventory, Suppliers, Customers CSVs", 1, 6.2, 3.8, 1.4, "#1E3A8A"),
        ("Data Agent", "2. Data Quality Agent", "Validation, Cleaning & Schema Audit (98.8%)", 5.2, 6.2, 3.8, 1.4, "#0284C7"),
        ("Analytics Agent", "3. Predictive Analytics", "Random Forest Risk Classifier & ROC/PR Curves", 9.4, 6.2, 3.8, 1.4, "#0D9488"),
        
        ("KG Agent", "4. Knowledge Graph Agent", "Multi-relational Supply Chain Topology & Traversal", 1, 3.8, 3.8, 1.4, "#4F46E5"),
        ("RAG Agent", "5. RAG Policy Agent", "Sentence Transformers / TF-IDF Policy Retrieval", 5.2, 3.8, 3.8, 1.4, "#7C3AED"),
        ("Planner Agent", "6. Planner & Guardrails", "Candidate Strategy Generation & Policy Pruning", 9.4, 3.8, 3.8, 1.4, "#C026D3"),
        
        ("Scenario Agent", "7. Scenario Simulation", "1,000 Monte Carlo Runs across 3 Macro Worlds", 1, 1.2, 3.8, 1.4, "#DB2777"),
        ("Explain Agent", "8. Explainability Agent", "TreeSHAP Attributions & Multimodal Citations", 5.2, 1.2, 3.8, 1.4, "#E11D48"),
        ("Coordinator Agent", "9. Coordinator & Feedback", "Human-in-the-Loop Review & SQLite Decision Store", 9.4, 1.2, 3.8, 1.4, "#EA580C"),
    ]

    for bid, title, desc, x, y, w, h, col in blocks:
        # Draw box
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor='#1A1C24', edgecolor=col, linewidth=2.5
        )
        ax.add_patch(rect)
        # Add badge bar
        badge = patches.FancyBboxPatch(
            (x, y + h - 0.35), w, 0.35, boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=col, edgecolor=col
        )
        ax.add_patch(badge)
        ax.text(x + w/2, y + h - 0.20, title, color='white', weight='bold', fontsize=11, ha='center', va='center')
        ax.text(x + w/2, y + (h-0.35)/2, desc, color='#E2E8F0', fontsize=8.5, ha='center', va='center', wrap=True)

    # Connecting directional arrows
    arrowprops = dict(arrowstyle="->", color="#94A3B8", lw=2, mutation_scale=15)
    ax.annotate("", xy=(5.2, 6.9), xytext=(4.8, 6.9), arrowprops=arrowprops)
    ax.annotate("", xy=(9.4, 6.9), xytext=(9.0, 6.9), arrowprops=arrowprops)
    
    ax.annotate("", xy=(2.9, 5.2), xytext=(2.9, 6.2), arrowprops=arrowprops)
    ax.annotate("", xy=(7.1, 5.2), xytext=(7.1, 6.2), arrowprops=arrowprops)
    ax.annotate("", xy=(11.3, 5.2), xytext=(11.3, 6.2), arrowprops=arrowprops)

    ax.annotate("", xy=(2.9, 2.6), xytext=(2.9, 3.8), arrowprops=arrowprops)
    ax.annotate("", xy=(7.1, 2.6), xytext=(7.1, 3.8), arrowprops=arrowprops)
    ax.annotate("", xy=(11.3, 2.6), xytext=(11.3, 3.8), arrowprops=arrowprops)

    ax.set_xlim(0.5, 13.8)
    ax.set_ylim(0.5, 8.2)
    ax.axis('off')
    plt.title("CogniOps AI: 9-Block Autonomous Enterprise Architecture", color='white', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("assets/architecture.png", dpi=200, bbox_inches='tight', facecolor='#0E1117')
    plt.close()
    print("Generated assets/architecture.png successfully.")

if __name__ == "__main__":
    generate_architecture_image()
