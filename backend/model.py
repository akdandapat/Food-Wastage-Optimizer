from __future__ import annotations

from backend.forecasting import KitchenForecastSystem

# Compatibility alias for older imports. The production implementation now lives
# in backend.forecasting.KitchenForecastSystem.
DemandForecaster = KitchenForecastSystem

__all__ = ["KitchenForecastSystem", "DemandForecaster"]
