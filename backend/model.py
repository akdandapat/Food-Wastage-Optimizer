"""Backward-compatible model alias module.

This module re-exports :class:`~backend.forecasting.KitchenForecastSystem`
under the legacy name ``DemandForecaster`` so that older import paths
(``from backend.model import DemandForecaster``) continue to resolve
without modification.

The canonical implementation lives in
:mod:`backend.forecasting.KitchenForecastSystem`.  All new code should
import directly from ``backend.forecasting``.

Example::

    # Preferred (new code)
    from backend.forecasting import KitchenForecastSystem

    # Legacy (still works)
    from backend.model import DemandForecaster
"""

from __future__ import annotations

from typing import List, Type

from backend.forecasting import KitchenForecastSystem

# Compatibility alias for older imports. The production implementation now lives
# in backend.forecasting.KitchenForecastSystem.
DemandForecaster: Type[KitchenForecastSystem] = KitchenForecastSystem

__all__: List[str] = ["KitchenForecastSystem", "DemandForecaster"]
