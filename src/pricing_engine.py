# pricing_engine.py

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PricingResult:
    optimal_price: float
    expected_demand: float
    expected_revenue: float
    static_price: float
    static_revenue: float
    uplift_percent: float


class DynamicPricingEngine:
    """
    Rule-based pricing optimization layer on top of demand prediction.

    Improvements in this version:
    - searches around a realistic reference-price band instead of a very wide range
    - adds context-aware elasticity
    - adds a small penalty for extreme price movement
    - still remains fully compatible with main.py and app.py
    """

    def __init__(
        self,
        min_price: float = 5.0,
        max_price: float = 100.0,
        price_step: float = 1.0,
        elasticity: float = 0.35,
        floor_demand_ratio: float = 0.05,
        min_discount_multiplier: float = 0.75,
        max_markup_multiplier: float = 1.35,
        price_change_penalty: float = 0.015,
    ):
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.price_step = float(price_step)
        self.elasticity = float(elasticity)
        self.floor_demand_ratio = float(floor_demand_ratio)
        self.min_discount_multiplier = float(min_discount_multiplier)
        self.max_markup_multiplier = float(max_markup_multiplier)
        self.price_change_penalty = float(price_change_penalty)

    def _get_reference_price(self, row: pd.Series) -> float:
        """
        Use the observed average fare as a realistic anchor price.
        If unavailable, fall back to a mid-range base price.
        """
        for col in ("avg_fare_amount", "avg_total_amount", "fare_amount", "total_amount"):
            if col in row and pd.notna(row[col]) and float(row[col]) > 0:
                return float(row[col])
        return (self.min_price + self.max_price) / 2.0

    def _get_price_bounds(self, reference_price: float) -> Tuple[float, float]:
        """
        Build a realistic price search band around the reference price.
        This prevents the optimizer from always selecting the absolute upper bound.
        """
        if reference_price <= 0:
            reference_price = (self.min_price + self.max_price) / 2.0

        lower = max(self.min_price, reference_price * self.min_discount_multiplier)
        upper = min(self.max_price, reference_price * self.max_markup_multiplier)

        if upper <= lower:
            lower = self.min_price
            upper = self.max_price

        return float(lower), float(upper)

    def _context_elasticity(self, row: Optional[pd.Series], base_demand: float) -> float:
        """
        Adjust price sensitivity using simple context signals.

        Peak-hour demand is usually less price-sensitive.
        Weekends and low-demand cases can be slightly more sensitive.
        """
        e = float(self.elasticity)

        if row is not None:
            if "is_peak_hour" in row and pd.notna(row["is_peak_hour"]) and int(row["is_peak_hour"]) == 1:
                e *= 0.85

            if "is_weekend" in row and pd.notna(row["is_weekend"]) and int(row["is_weekend"]) == 1:
                e *= 0.95

            if "zone_trip_volume" in row and pd.notna(row["zone_trip_volume"]):
                zone_vol = float(row["zone_trip_volume"])
                if zone_vol > 700:
                    e *= 0.90
                elif zone_vol < 200:
                    e *= 1.05

        if base_demand >= 150:
            e *= 0.90
        elif base_demand <= 20:
            e *= 1.08

        return max(e, 0.05)

    def _adjust_demand(
        self,
        base_demand: float,
        reference_price: float,
        candidate_price: float,
        row: Optional[pd.Series] = None,
    ) -> float:
        """
        Smooth price sensitivity model with a demand floor.

        adjusted_demand = base_demand * exp(-elasticity * pct_change)
        """
        if reference_price <= 0:
            reference_price = 1.0

        elasticity = self._context_elasticity(row=row, base_demand=base_demand)

        pct_change = (candidate_price - reference_price) / reference_price
        adjusted = base_demand * np.exp(-elasticity * pct_change)

        floor = self.floor_demand_ratio * base_demand
        return float(max(adjusted, floor))

    def _objective(
        self,
        base_demand: float,
        reference_price: float,
        candidate_price: float,
        row: Optional[pd.Series] = None,
    ) -> Tuple[float, float]:
        """
        Revenue objective with a small penalty for extreme price movement.
        This makes the recommendation more realistic and avoids boundary saturation.
        """
        adjusted_demand = self._adjust_demand(
            base_demand=base_demand,
            reference_price=reference_price,
            candidate_price=candidate_price,
            row=row,
        )

        revenue = candidate_price * adjusted_demand
        pct_change = 0.0 if reference_price <= 0 else (candidate_price - reference_price) / reference_price

        # Gentle penalty for extreme changes in price level
        penalty = self.price_change_penalty * base_demand * reference_price * (pct_change ** 2)
        score = revenue - penalty

        return float(score), float(adjusted_demand)

    def optimize_price(
        self,
        base_demand: float,
        reference_price: Optional[float] = None,
        row: Optional[pd.Series] = None,
    ) -> PricingResult:
        """
        Search over a realistic price grid and choose the price that maximizes the objective.
        """
        if reference_price is None:
            reference_price = 1.0

        lower, upper = self._get_price_bounds(reference_price)
        price_grid = np.arange(lower, upper + self.price_step, self.price_step)

        best_price = None
        best_demand = None
        best_score = -np.inf
        best_revenue = -np.inf

        for price in price_grid:
            score, adjusted_demand = self._objective(
                base_demand=base_demand,
                reference_price=reference_price,
                candidate_price=price,
                row=row,
            )
            revenue = price * adjusted_demand

            if score > best_score:
                best_score = score
                best_revenue = revenue
                best_price = price
                best_demand = adjusted_demand

        static_price = float(reference_price)
        static_demand = self._adjust_demand(
            base_demand=base_demand,
            reference_price=reference_price,
            candidate_price=static_price,
            row=row,
        )
        static_revenue = static_price * static_demand

        uplift_percent = 0.0
        if static_revenue > 0:
            uplift_percent = ((best_revenue - static_revenue) / static_revenue) * 100.0

        return PricingResult(
            optimal_price=float(best_price),
            expected_demand=float(best_demand),
            expected_revenue=float(best_revenue),
            static_price=float(static_price),
            static_revenue=float(static_revenue),
            uplift_percent=float(uplift_percent),
        )

    def revenue_curve(
        self,
        base_demand: float,
        reference_price: float,
        row: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Return a dataframe showing price, adjusted demand, revenue, and objective score
        across the search band.
        """
        lower, upper = self._get_price_bounds(reference_price)

        rows = []
        for price in np.arange(lower, upper + self.price_step, self.price_step):
            score, adjusted_demand = self._objective(
                base_demand=base_demand,
                reference_price=reference_price,
                candidate_price=price,
                row=row,
            )
            rows.append({
                "price": float(price),
                "adjusted_demand": float(adjusted_demand),
                "revenue": float(price * adjusted_demand),
                "objective_score": float(score),
            })

        return pd.DataFrame(rows)

    def sensitivity_curve(
        self,
        base_demand: float,
        reference_price: float,
        elasticity_values=None,
        row: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compare the revenue optimum under different elasticity assumptions.
        Useful for robustness analysis in report and viva.
        """
        if elasticity_values is None:
            elasticity_values = [0.15, 0.25, 0.35, 0.50, 0.70]

        rows = []
        original_elasticity = self.elasticity

        for e in elasticity_values:
            self.elasticity = float(e)
            result = self.optimize_price(
                base_demand=base_demand,
                reference_price=reference_price,
                row=row,
            )
            rows.append({
                "elasticity": float(e),
                "optimal_price": float(result.optimal_price),
                "expected_demand": float(result.expected_demand),
                "expected_revenue": float(result.expected_revenue),
                "static_price": float(result.static_price),
                "static_revenue": float(result.static_revenue),
                "uplift_percent": float(result.uplift_percent),
            })

        self.elasticity = original_elasticity
        return pd.DataFrame(rows)

    def explain_price_choice(self, result: PricingResult) -> str:
        return (
            f"Optimal price = {result.optimal_price:.2f}, "
            f"expected demand = {result.expected_demand:.2f}, "
            f"expected revenue = {result.expected_revenue:.2f}, "
            f"uplift over static pricing = {result.uplift_percent:.2f}%"
        )


def optimize_for_row(
    row: pd.Series,
    predicted_demand: float,
    min_price: float = 5.0,
    max_price: float = 100.0,
    price_step: float = 1.0,
    elasticity: float = 0.35
) -> Tuple[PricingResult, pd.DataFrame]:
    """
    Convenience function:
    - Takes one row from demand_panel
    - Uses its observed fare as the reference price
    - Optimizes revenue using the predicted demand
    """
    engine = DynamicPricingEngine(
        min_price=min_price,
        max_price=max_price,
        price_step=price_step,
        elasticity=elasticity,
    )

    ref_price = engine._get_reference_price(row)
    result = engine.optimize_price(
        base_demand=float(predicted_demand),
        reference_price=ref_price,
        row=row,
    )
    curve = engine.revenue_curve(
        base_demand=float(predicted_demand),
        reference_price=ref_price,
        row=row,
    )

    return result, curve


def save_curve(curve_df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    curve_df.to_csv(path, index=False)


def save_result(result: PricingResult, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.__dict__, f, indent=4)


if __name__ == "__main__":
    demo_row = pd.Series({"avg_fare_amount": 16.5, "avg_total_amount": 19.2})
    demo_predicted_demand = 120.0

    result, curve = optimize_for_row(
        row=demo_row,
        predicted_demand=demo_predicted_demand,
        min_price=5,
        max_price=60,
        price_step=1,
        elasticity=0.35
    )

    print("=== PRICING ENGINE DEMO ===")
    print(f"Reference price : {result.static_price:.2f}")
    print(f"Static revenue  : {result.static_revenue:.2f}")
    print(f"Optimal price   : {result.optimal_price:.2f}")
    print(f"Best demand     : {result.expected_demand:.2f}")
    print(f"Best revenue    : {result.expected_revenue:.2f}")
    print(f"Uplift (%)      : {result.uplift_percent:.2f}%")

    os.makedirs("outputs/pricing", exist_ok=True)
    save_curve(curve, "outputs/pricing/revenue_curve.csv")
    save_result(result, "outputs/pricing/best_pricing_result.json")

    print("\nSaved:")
    print("- outputs/pricing/revenue_curve.csv")
    print("- outputs/pricing/best_pricing_result.json")