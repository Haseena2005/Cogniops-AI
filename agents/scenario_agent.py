"""
CogniOps AI - Scenario Simulation Agent.
Executes Monte Carlo what-if stochastic simulation (1,000 runs) across
Optimistic, Expected, and Pessimistic enterprise scenarios.
Calculates Expected Utility U = sum(p_k * u_k), Downside Risk, and Pareto Strategy Ranking.
"""
import numpy as np
from typing import List, Dict, Any
from core.models import StrategyCandidate, ScenarioOutcome
from core.config import RANDOM_SEED

class ScenarioAgent:
    def __init__(self):
        self.name = "Scenario Agent"
        self.num_simulations = 1000

    def simulate_strategies(
        self,
        strategies: List[StrategyCandidate],
        situation_id: str,
        seed: int = RANDOM_SEED,
        perturbation_pct: float = 0.0,
        delay_multiplier: float = 1.0,
        demand_multiplier: float = 1.0,
        budget_cap: float = float("inf")
    ) -> List[ScenarioOutcome]:
        """
        Runs Monte Carlo what-if simulation for each surviving candidate strategy.
        Models payoff probability distribution across 3 macroeconomic scenarios:
        - Optimistic (prob = 0.25)
        - Expected (prob = 0.55)
        - Pessimistic (prob = 0.20)
        Supports what-if sandbox multipliers and stochastic input perturbations (+/- 10%).
        """
        np.random.seed(seed)
        outcomes = []

        scenario_weights = {"optimistic": 0.25, "expected": 0.55, "pessimistic": 0.20}

        # Baseline payoff parameters per situation
        payoff_profiles = {
            "supplier_delay": {
                "STRAT-01": {"mu_opt": 0.94, "mu_exp": 0.88, "mu_pess": 0.72, "sigma": 0.05, "roi_multiplier": 3.8},
                "STRAT-02": {"mu_opt": 0.89, "mu_exp": 0.82, "mu_pess": 0.65, "sigma": 0.07, "roi_multiplier": 3.2},
                "STRAT-04": {"mu_opt": 0.55, "mu_exp": 0.38, "mu_pess": 0.18, "sigma": 0.12, "roi_multiplier": 0.0}
            },
            "sales_decline": {
                "STRAT-01": {"mu_opt": 0.92, "mu_exp": 0.83, "mu_pess": 0.68, "sigma": 0.06, "roi_multiplier": 3.5},
                "STRAT-03": {"mu_opt": 0.85, "mu_exp": 0.77, "mu_pess": 0.62, "sigma": 0.07, "roi_multiplier": 3.0},
                "STRAT-04": {"mu_opt": 0.52, "mu_exp": 0.35, "mu_pess": 0.16, "sigma": 0.11, "roi_multiplier": 0.0}
            },
            "inventory_shortage": {
                "STRAT-01": {"mu_opt": 0.95, "mu_exp": 0.89, "mu_pess": 0.74, "sigma": 0.04, "roi_multiplier": 4.2},
                "STRAT-02": {"mu_opt": 0.84, "mu_exp": 0.76, "mu_pess": 0.58, "sigma": 0.08, "roi_multiplier": 2.9},
                "STRAT-04": {"mu_opt": 0.48, "mu_exp": 0.30, "mu_pess": 0.12, "sigma": 0.14, "roi_multiplier": 0.0}
            },
            "customer_churn_spike": {
                "STRAT-01": {"mu_opt": 0.93, "mu_exp": 0.87, "mu_pess": 0.70, "sigma": 0.05, "roi_multiplier": 4.0},
                "STRAT-02": {"mu_opt": 0.86, "mu_exp": 0.79, "mu_pess": 0.60, "sigma": 0.07, "roi_multiplier": 3.1},
                "STRAT-04": {"mu_opt": 0.50, "mu_exp": 0.32, "mu_pess": 0.15, "sigma": 0.13, "roi_multiplier": 0.0}
            }
        }

        default_profile = {"mu_opt": 0.80, "mu_exp": 0.70, "mu_pess": 0.50, "sigma": 0.08, "roi_multiplier": 2.5}
        sit_profiles = payoff_profiles.get(situation_id, {})

        for strat in strategies:
            p = sit_profiles.get(strat.id, default_profile).copy()

            # Apply perturbations if requested (robustness check)
            if perturbation_pct != 0.0:
                p["mu_opt"] = np.clip(p["mu_opt"] * (1.0 + np.random.uniform(-perturbation_pct, perturbation_pct)), 0.0, 1.0)
                p["mu_exp"] = np.clip(p["mu_exp"] * (1.0 + np.random.uniform(-perturbation_pct, perturbation_pct)), 0.0, 1.0)
                p["mu_pess"] = np.clip(p["mu_pess"] * (1.0 + np.random.uniform(-perturbation_pct, perturbation_pct)), 0.0, 1.0)

            # Apply what-if sandbox controls
            effective_cost = strat.estimated_cost
            if strat.action_type == "Expedited Logistics":
                p["mu_exp"] = np.clip(p["mu_exp"] * (1.0 - (delay_multiplier - 1.0) * 0.15), 0.1, 1.0)
            if "Marketing" in strat.action_type or "Promo" in strat.title:
                p["mu_exp"] = np.clip(p["mu_exp"] * demand_multiplier, 0.1, 1.0)

            # Budget penalty if cost exceeds budget cap
            if effective_cost > budget_cap:
                penalty = min(0.40, (effective_cost - budget_cap) / (budget_cap + 1e-5) * 0.20)
                p["mu_exp"] = max(0.1, p["mu_exp"] - penalty)
                p["mu_opt"] = max(0.1, p["mu_opt"] - penalty)

            # Monte Carlo sampling: 1,000 synthetic trajectory iterations
            opt_samples = np.random.normal(p["mu_opt"], p["sigma"], self.num_simulations)
            exp_samples = np.random.normal(p["mu_exp"], p["sigma"], self.num_simulations)
            pess_samples = np.random.normal(p["mu_pess"], p["sigma"], self.num_simulations)

            # Clip utility metrics to [0, 1]
            opt_mean = float(np.clip(np.mean(opt_samples), 0.0, 1.0))
            exp_mean = float(np.clip(np.mean(exp_samples), 0.0, 1.0))
            pess_mean = float(np.clip(np.mean(pess_samples), 0.0, 1.0))

            # Composite Monte Carlo distribution across weighted scenarios
            weighted_distribution = (
                scenario_weights["optimistic"] * opt_samples +
                scenario_weights["expected"] * exp_samples +
                scenario_weights["pessimistic"] * pess_samples
            )

            expected_utility = float(np.clip(np.mean(weighted_distribution), 0.0, 1.0))
            risk_std = float(np.std(weighted_distribution))
            worst_case = float(np.percentile(pess_samples, 5)) # 5th percentile downside VaR

            # Financial ROI calculation
            gross_benefit = expected_utility * strat.estimated_cost * p["roi_multiplier"]
            net_roi = round(gross_benefit - strat.estimated_cost, 2) if strat.estimated_cost > 0 else 0.0

            outcomes.append(ScenarioOutcome(
                strategy_id=strat.id,
                strategy_title=strat.title,
                optimistic_value=round(opt_mean, 3),
                expected_value=round(exp_mean, 3),
                pessimistic_value=round(pess_mean, 3),
                expected_utility=round(expected_utility, 3),
                risk_std_dev=round(risk_std, 3),
                worst_case_downside=round(worst_case, 3),
                cost=strat.estimated_cost,
                net_roi=net_roi
            ))

        # Rank strategies strictly by expected utility descending
        outcomes.sort(key=lambda x: x.expected_utility, reverse=True)
        for idx, out in enumerate(outcomes):
            out.rank = idx + 1

        return outcomes
