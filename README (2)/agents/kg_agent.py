"""
CogniOps AI - Knowledge Graph Agent.
Constructs multi-relational enterprise ontology (Supplier, Product, Order, Customer, Region, Warehouse)
and performs multi-hop risk propagation queries to compute revenue-at-risk.
"""
import networkx as nx
import pandas as pd
from typing import Dict, Any, List, Tuple

class KnowledgeGraphAgent:
    def __init__(self):
        self.name = "Knowledge Graph Agent"
        self.graph = nx.DiGraph()

    def build_graph(
        self,
        suppliers_df: pd.DataFrame,
        products_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        inventory_df: pd.DataFrame
    ) -> nx.DiGraph:
        """
        Builds a directed enterprise Knowledge Graph with rich node attributes and relations:
        - (Supplier) -[supplies]-> (Product)
        - (Product) -[stocked_in]-> (Warehouse)
        - (Customer) -[placed]-> (Order)
        - (Order) -[contains]-> (Product)
        - (Order) -[ships_from]-> (Warehouse)
        - (Customer) -[located_in]-> (Region)
        """
        G = nx.DiGraph()

        # Add Suppliers
        for _, r in suppliers_df.iterrows():
            G.add_node(
                r["supplier_id"],
                label=r["name"],
                node_type="Supplier",
                country=r["country"],
                reliability=r["reliability_score"],
                delay_days=r["delay_days"]
            )

        # Add Products
        for _, r in products_df.iterrows():
            G.add_node(
                r["product_id"],
                label=r["name"],
                node_type="Product",
                category=r["category"],
                unit_price=r["unit_price"]
            )
            # Edge: Supplier -> Product
            G.add_edge(r["supplier_id"], r["product_id"], relation="supplies")

        # Add Warehouses from Inventory
        for _, r in inventory_df.iterrows():
            wh_id = r["warehouse_id"]
            if not G.has_node(wh_id):
                G.add_node(wh_id, label=wh_id, node_type="Warehouse")
            # Edge: Product -> Warehouse
            G.add_edge(r["product_id"], wh_id, relation="stocked_in", stock=r["current_stock"])

        # Add Customers
        for _, r in customers_df.iterrows():
            G.add_node(
                r["customer_id"],
                label=r["customer_name"],
                node_type="Customer",
                segment=r["segment"],
                acv=r["annual_contract_value"],
                churn=r["churn_flag"]
            )
            reg_id = r["assigned_region"]
            if not G.has_node(reg_id):
                G.add_node(reg_id, label=reg_id, node_type="Region")
            G.add_edge(r["customer_id"], reg_id, relation="located_in")

        # Add Orders
        for _, r in orders_df.iterrows():
            ord_id = r["order_id"]
            G.add_node(
                ord_id,
                label=ord_id,
                node_type="Order",
                order_value=r["order_value"],
                status=r["status"]
            )
            G.add_edge(r["customer_id"], ord_id, relation="placed")
            G.add_edge(ord_id, r["product_id"], relation="contains", qty=r["quantity"])
            G.add_edge(ord_id, r["warehouse_id"], relation="ships_from")

        self.graph = G
        return G

    def query_impact_path(self, situation_id: str) -> Dict[str, Any]:
        """
        Executes a multi-hop traversal to identify entities exposed to the situation crisis:
        Supplier -> Products -> Orders -> Customers -> Total Revenue at Risk.
        """
        if self.graph.number_of_nodes() == 0:
            return {"affected_nodes": [], "affected_edges": [], "revenue_at_risk": 0.0, "affected_orders": []}

        affected_nodes = set()
        affected_edges = []
        affected_orders_data = []
        total_rev_at_risk = 0.0
        insights = []

        if situation_id == "supplier_delay":
            root_node = "SUP-01"
            affected_nodes.add(root_node)

            # Hop 1: Products supplied by SUP-01
            prods = [target for _, target, data in self.graph.out_edges(root_node, data=True) if data.get("relation") == "supplies"]
            for p in prods:
                affected_nodes.add(p)
                affected_edges.append((root_node, p))

                # Hop 2: Orders containing Product
                # In graph: Order -> Product (in_edges to Product with relation contains)
                for ord_node, _, data in self.graph.in_edges(p, data=True):
                    if data.get("relation") == "contains":
                        affected_nodes.add(ord_node)
                        affected_edges.append((ord_node, p))
                        ord_val = self.graph.nodes[ord_node].get("order_value", 0.0)
                        ord_status = self.graph.nodes[ord_node].get("status", "Fulfilled")
                        
                        # Hop 3: Customer who placed the order
                        for cust_node, _, cdata in self.graph.in_edges(ord_node, data=True):
                            if cdata.get("relation") == "placed":
                                affected_nodes.add(cust_node)
                                affected_edges.append((cust_node, ord_node))
                                cust_name = self.graph.nodes[cust_node].get("label", cust_node)
                                cust_seg = self.graph.nodes[cust_node].get("segment", "Enterprise")
                                
                                if ord_status == "Delayed":
                                    total_rev_at_risk += ord_val
                                    affected_orders_data.append({
                                        "order_id": ord_node,
                                        "customer_id": cust_node,
                                        "customer_name": cust_name,
                                        "segment": cust_seg,
                                        "product_id": p,
                                        "product_name": self.graph.nodes[p].get("label", p),
                                        "order_value": ord_val,
                                        "status": ord_status
                                    })
            
            insights.append(f"Supplier SUP-01 delay propagates through {len(prods)} products to {len(affected_orders_data)} pending client orders.")
            insights.append(f"Immediate order fulfillment exposure amounts to ${total_rev_at_risk:,.2f} at critical risk.")

        elif situation_id == "inventory_shortage":
            root_node = "PROD-01"
            affected_nodes.add(root_node)
            for ord_node, _, data in self.graph.in_edges(root_node, data=True):
                if data.get("relation") == "contains":
                    affected_nodes.add(ord_node)
                    affected_edges.append((ord_node, root_node))
                    ord_val = self.graph.nodes[ord_node].get("order_value", 0.0)
                    total_rev_at_risk += ord_val
                    for cust_node, _, _ in self.graph.in_edges(ord_node, data=True):
                        affected_nodes.add(cust_node)
                        affected_edges.append((cust_node, ord_node))
                        affected_orders_data.append({
                            "order_id": ord_node,
                            "customer_id": cust_node,
                            "customer_name": self.graph.nodes[cust_node].get("label", cust_node),
                            "segment": self.graph.nodes[cust_node].get("segment", "Enterprise"),
                            "product_id": root_node,
                            "product_name": self.graph.nodes[root_node].get("label", root_node),
                            "order_value": ord_val,
                            "status": self.graph.nodes[ord_node].get("status", "Pending")
                        })
            insights.append(f"PROD-01 warehouse shortage jeopardizes {len(affected_orders_data)} downstream customer deliveries.")

        elif situation_id == "sales_decline":
            reg_node = "Region-East"
            affected_nodes.add(reg_node)
            for cust_node, _, data in self.graph.in_edges(reg_node, data=True):
                if data.get("relation") == "located_in":
                    affected_nodes.add(cust_node)
                    affected_edges.append((cust_node, reg_node))
                    acv = self.graph.nodes[cust_node].get("acv", 0.0)
                    total_rev_at_risk += (acv * 0.06) # quarterly impact fraction of annual ACV
            # Strict cap: cannot exceed total quarterly revenue for the region ($1,093,810)
            total_rev_at_risk = min(total_rev_at_risk, 1093810.0)
            insights.append(f"Regional contraction affects accounts across Region-East with ${total_rev_at_risk:,.2f} projected quarterly deficit.")

        elif situation_id == "customer_churn_spike":
            for n, d in self.graph.nodes(data=True):
                if d.get("node_type") == "Customer" and d.get("segment") == "Enterprise" and d.get("churn") == 1:
                    affected_nodes.add(n)
                    acv = d.get("acv", 0.0)
                    # Quarterly ARR exposure at risk (ACV / 4)
                    quarterly_val = acv * 0.25
                    total_rev_at_risk += quarterly_val
                    affected_orders_data.append({
                        "order_id": "N/A (Account)",
                        "customer_id": n,
                        "customer_name": d.get("label", n),
                        "segment": "Enterprise",
                        "product_id": "Enterprise Tier",
                        "product_name": "Full Suite",
                        "order_value": quarterly_val,
                        "status": "Churned"
                    })
            insights.append(f"High-ACV Enterprise segment churn escalates ${total_rev_at_risk:,.2f} in recurrent quarterly contract attrition.")

        # Safety upper bound: revenue at risk can never exceed total annualized enterprise revenue
        max_period_revenue = 60374610.0
        total_rev_at_risk = min(total_rev_at_risk, max_period_revenue)

        return {
            "affected_nodes": list(affected_nodes),
            "affected_edges": affected_edges,
            "revenue_at_risk": round(total_rev_at_risk, 2),
            "max_period_revenue": max_period_revenue,
            "affected_orders": affected_orders_data[:15], # top 15 records
            "kg_insights": insights
        }
