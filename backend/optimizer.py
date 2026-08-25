from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from math import ceil

import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class NewsvendorRecommendation:
    optimal_quantity: int
    expected_waste: float
    expected_shortage: float
    expected_cost_inr: float
    critical_ratio: float
    shortage_probability: float
    service_level_target: float
    service_level_satisfied: bool

    def to_dict(self) -> dict:
        return asdict(self)


class NewsvendorOptimizer:
    """Single-period Newsvendor under asymmetric waste and shortage costs.

    The optimizer chooses the preparation quantity *Q* that minimises

        E[cost] = c_w · E[waste(Q)] + c_s · E[shortage(Q)]

    where demand ~ N(μ, σ²).  The closed-form optimal *Q* is

        Q* = μ + σ · Φ⁻¹(critical_ratio)

    The ``critical_ratio`` is the larger of:

    * **cost-optimal ratio** ``c_s / (c_s + c_w)`` — the textbook
      Newsvendor solution; and
    * **service-level target** — a hard floor on fill-rate probability,
      typically 0.95.

    When the service-level target binds, the optimizer prepares *more*
    than cost-optimal to avoid politically unacceptable shortages.
    """

    def __init__(
        self,
        waste_cost: float,
        shortage_cost: float,
        service_level_target: float = 0.95,
    ) -> None:
        self.waste_cost = waste_cost
        self.shortage_cost = shortage_cost
        self.service_level_target = min(max(float(service_level_target), 0.50), 0.999)

    @property
    def cost_critical_ratio(self) -> float:
        """Pure cost-optimal critical ratio: c_s / (c_s + c_w).

        This is the fractile that minimises expected cost when there is
        no separate service-level constraint.
        """
        return self.shortage_cost / (self.shortage_cost + self.waste_cost)

    @property
    def critical_ratio(self) -> float:
        """Effective critical ratio: max(cost_ratio, service_level_target).

        When the service-level target exceeds the cost ratio, the
        optimizer over-prepares relative to cost-optimal to meet the
        fill-rate floor.
        """
        return max(self.cost_critical_ratio, self.service_level_target)

    def evaluate_quantity(self, mu: float, sigma: float, quantity: float) -> NewsvendorRecommendation:
        """Evaluate a given preparation quantity against the demand distribution.

        Computes expected waste, expected shortage, expected cost, and
        whether the implied shortage probability satisfies the
        service-level target.
        """
        safe_sigma = max(float(sigma), 1e-6)
        standardized = (quantity - mu) / safe_sigma
        phi = norm.pdf(standardized)
        cdf = norm.cdf(standardized)
        shortage_probability = max(1.0 - cdf, 0.0)

        expected_waste = safe_sigma * phi + (quantity - mu) * cdf
        expected_shortage = safe_sigma * phi + (mu - quantity) * (1 - cdf)
        expected_cost = (
            self.waste_cost * expected_waste + self.shortage_cost * expected_shortage
        )

        return NewsvendorRecommendation(
            optimal_quantity=int(ceil(max(quantity, 0.0))),
            expected_waste=float(expected_waste),
            expected_shortage=float(expected_shortage),
            expected_cost_inr=float(expected_cost),
            critical_ratio=float(self.critical_ratio),
            shortage_probability=float(shortage_probability),
            service_level_target=float(self.service_level_target),
            service_level_satisfied=bool(shortage_probability <= (1.0 - self.service_level_target)),
        )

    def recommend(self, mu: float, sigma: float) -> NewsvendorRecommendation:
        """Compute the optimal preparation quantity Q* = μ + σ · Φ⁻¹(CR).

        Logs which constraint binds — the cost-optimal ratio or the
        service-level target — since the binding constraint determines
        whether Q* is cost-optimal or inflated for political safety.
        """
        safe_sigma = max(float(sigma), 1e-6)
        cost_cr = self.cost_critical_ratio
        effective_cr = self.critical_ratio
        if effective_cr > cost_cr:
            logger.debug(
                "Service-level target binds (%.3f > cost ratio %.3f); "
                "preparing above cost-optimal.",
                effective_cr, cost_cr,
            )
        else:
            logger.debug(
                "Cost ratio binds (%.3f >= service level %.3f); "
                "preparing at cost-optimal.",
                cost_cr, self.service_level_target,
            )
        z_score = norm.ppf(effective_cr)
        quantity = mu + safe_sigma * z_score
        return self.evaluate_quantity(mu, sigma, quantity)

    def realized_metrics(self, quantity: float, actual_demand: float) -> dict[str, float]:
        """Compute realized waste, shortage, and cost after observing actual demand."""
        waste = max(float(quantity) - float(actual_demand), 0.0)
        shortage = max(float(actual_demand) - float(quantity), 0.0)
        cost = waste * self.waste_cost + shortage * self.shortage_cost
        return {"waste": waste, "shortage": shortage, "cost_inr": cost}

    def ingredient_plan(
        self, forecasts: pd.DataFrame, recipes: pd.DataFrame
    ) -> list[dict[str, float | str]]:
        if forecasts.empty or recipes.empty:
            return []

        merged = forecasts.merge(recipes, on="menu_type", how="left")
        merged["planned_quantity"] = merged["optimal_quantity"] * merged["qty_per_100_meals"] / 100.0
        grouped = (
            merged.groupby(["ingredient_name", "unit"], as_index=False)["planned_quantity"].sum()
            .rename(columns={"planned_quantity": "total_quantity"})
            .sort_values(["ingredient_name", "unit"])
        )
        grouped["total_quantity"] = grouped["total_quantity"].round(2)
        return grouped.to_dict("records")
