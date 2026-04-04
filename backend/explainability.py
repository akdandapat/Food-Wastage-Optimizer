"""
Model explainability: SHAP-style drivers for tree models, linear coefficients for
Ridge, permutation importance as a lightweight ablation proxy, and optional TFT
attention summaries when the deep model is selected.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# SHAP is optional on constrained environments; import lazily inside functions.


def explain_tabular_instance(
    artifact: dict[str, Any],
    row_features: pd.DataFrame,
    model_name: str,
    background_matrix: np.ndarray | None = None,
    max_features: int = 12,
) -> dict[str, Any]:
    """
    Produce local feature attributions for one preprocessed feature row.

    ``row_features`` must contain the same columns as training features.
    """
    preprocessor = artifact["preprocessor"]
    model = artifact["model"]
    columns = artifact["feature_columns"]
    x = preprocessor.transform(row_features[columns])
    names = list(preprocessor.get_feature_names_out(columns))
    short_names = [n.split("__", 1)[-1] for n in names]

    local_scores: dict[str, float] = {}
    method = "unavailable"

    if model_name == "ridge":
        method = "linear_coefficients"
        coef = np.asarray(model.coef_).ravel()
        x_dense = np.asarray(x).ravel()
        local_scores = {
            short_names[i]: float(x_dense[i] * coef[i]) for i in range(min(len(coef), len(short_names)))
        }

    elif model_name in {"random_forest", "xgboost", "lightgbm"}:
        try:
            import shap  # type: ignore[import-untyped]

            if model_name == "random_forest":
                explainer = shap.TreeExplainer(model)
                vec = explainer.shap_values(x)
                if isinstance(vec, list):
                    vec = vec[0]
                vec = np.asarray(vec).ravel()
            else:
                explainer = shap.TreeExplainer(model)
                vec = explainer.shap_values(x)
                vec = np.asarray(vec)
                if vec.ndim > 1:
                    vec = vec.ravel()
            method = "shap_tree"
            local_scores = {short_names[i]: float(vec[i]) for i in range(min(len(vec), len(short_names)))}
        except Exception as exc:
            method = f"shap_failed:{exc}"

    sorted_local = sorted(local_scores.items(), key=lambda item: abs(item[1]), reverse=True)[
        :max_features
    ]
    return {
        "method": method,
        "local_top_features": [{"feature": k, "value": v} for k, v in sorted_local],
    }


def permutation_ablation_summary(
    model: Any,
    x_matrix: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_features: int = 10,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Global ablation via sklearn permutation importance (held-out sample).
    """
    if len(y) < 30:
        return []
    result = permutation_importance(
        model,
        x_matrix,
        y,
        n_repeats=8,
        random_state=random_state,
        n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)[::-1][:max_features]
    return [
        {
            "feature": feature_names[i],
            "importance_mean": float(result.importances_mean[i]),
            "importance_std": float(result.importances_std[i]),
        }
        for i in order
    ]


def tft_attention_summary(tft_artifact: dict[str, Any] | None) -> dict[str, Any]:
    """
    Placeholder hook for TFT interpretability — full interpretation requires
    a forward pass with ``pytorch_forecasting`` interpret_output.

    When the bundled model exposes attention weights they can be logged here.
    """
    if tft_artifact is None or not tft_artifact.get("model"):
        return {"available": False, "note": "No TFT artifact loaded."}
    return {
        "available": False,
        "note": "Use TFT interpret_output in training worker for full attention maps; "
        "dashboard uses global permutation and tree SHAP for tabular champions.",
    }


def build_why_summary(
    local_top: list[dict[str, Any]],
    global_drivers: list[dict[str, Any]],
) -> str:
    """Short natural-language summary for API consumers."""
    if not local_top:
        return "Insufficient local attribution data for this forecast."
    top = local_top[0]
    parts = [
        f"Largest local driver is {top['feature']} (signed contribution {top['value']:.2f})."
    ]
    if global_drivers:
        g = global_drivers[0].get("driver") or global_drivers[0].get("feature")
        parts.append(f"Globally, {g} remains among the strongest demand signals.")
    return " ".join(parts)
