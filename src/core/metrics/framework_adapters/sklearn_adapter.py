from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SklearnParamBreakdown:
    total_params: int
    estimator_type: str          # "linear" | "svm" | "tree_ensemble" | "single_tree" | "ensemble_meta" | "unknown"
    is_fitted: bool
    is_sparse: bool
    n_sub_estimators: int


class SklearnParamCountError(Exception):
    pass


class SklearnAdapter:
    def matches(self, model: Any) -> bool:
        return hasattr(model, "get_params") and (hasattr(model, "fit") or hasattr(model, "predict"))

    def framework_name(self) -> str:
        return "scikit-learn"

    def count_parameters(self, model: Any) -> tuple[int, int]:
        breakdown = self._breakdown(self._resolve_final_estimator(model))
        return breakdown.total_params, breakdown.total_params

    def model_size_bytes(self, model: Any) -> int:
        import pickle
        try:
            return len(pickle.dumps(model))
        except Exception as e:
            raise SklearnParamCountError(f"Failed to estimate model size via pickle: {e}") from e

    def full_breakdown(self, model: Any) -> SklearnParamBreakdown:
        return self._breakdown(self._resolve_final_estimator(model))

    # ---- internals ----

    def _resolve_final_estimator(self, model: Any) -> Any:
        """Разворачивает Pipeline / GridSearchCV / RandomizedSearchCV до конечного estimator'а."""
        current = model
        seen = set()

        while id(current) not in seen:
            seen.add(id(current))

            # GridSearchCV / RandomizedSearchCV / HalvingSearchCV
            if hasattr(current, "best_estimator_"):
                current = current.best_estimator_
                continue

            if hasattr(current, "steps"):
                current = current.steps[-1][1]
                continue

            # CalibratedClassifierCV
            if hasattr(current, "calibrated_classifiers_"):
                calibrated = current.calibrated_classifiers_
                if calibrated:
                    current = calibrated[0].estimator
                    continue

            break

        return current

    def _breakdown(self, model: Any) -> SklearnParamBreakdown:
        is_fitted = self._is_fitted(model)
        if not is_fitted:
            return SklearnParamBreakdown(
                total_params=0, estimator_type="unknown",
                is_fitted=False, is_sparse=False, n_sub_estimators=0,
            )

        total = 0
        is_sparse = False
        n_sub = 0
        estimator_type = "unknown"

        # Linear models / Logistic Regression / Ridge / Lasso
        if hasattr(model, "coef_"):
            coef = model.coef_
            is_sparse = self._is_sparse(coef)
            total += coef.nnz if is_sparse else coef.size
            if hasattr(model, "intercept_"):
                total += getattr(model.intercept_, "size", 1)
            estimator_type = "linear"

        # SVM
        elif hasattr(model, "support_vectors_"):
            if hasattr(model, "dual_coef_"):
                total += model.dual_coef_.size
            total += model.support_vectors_.size
            if hasattr(model, "intercept_"):
                total += getattr(model.intercept_, "size", 1)
            estimator_type = "svm"

        # Voting / Stacking meta-ensembles — считаем params каждого под-estimator'а
        elif hasattr(model, "estimators_") and hasattr(model, "named_estimators_"):
            for name, sub_model in model.named_estimators_.items():
                sub_breakdown = self._breakdown(sub_model)
                total += sub_breakdown.total_params
                n_sub += 1
            estimator_type = "ensemble_meta"

        # RandomForest / GradientBoosting / ExtraTrees
        elif hasattr(model, "estimators_"):
            for estimator in model.estimators_:
                # GradientBoosting: estimators_ может быть 2D массивом (multi-output)
                sub_list = estimator.ravel() if hasattr(estimator, "ravel") else [estimator]
                for sub in sub_list:
                    if hasattr(sub, "tree_"):
                        total += sub.tree_.node_count
                        n_sub += 1
            estimator_type = "tree_ensemble"

        # Single DecisionTree
        elif hasattr(model, "tree_"):
            total += model.tree_.node_count
            estimator_type = "single_tree"

        # KMeans / clustering centroids
        elif hasattr(model, "cluster_centers_"):
            total += model.cluster_centers_.size
            estimator_type = "clustering"

        # Naive Bayes
        elif hasattr(model, "theta_") or hasattr(model, "feature_log_prob_"):
            for attr in ("theta_", "var_", "feature_log_prob_", "class_log_prior_"):
                if hasattr(model, attr):
                    total += getattr(model, attr).size
            estimator_type = "naive_bayes"

        else:
            raise SklearnParamCountError(
                f"Cannot determine parameter count for estimator type: {type(model).__name__}"
            )

        return SklearnParamBreakdown(
            total_params=total,
            estimator_type=estimator_type,
            is_fitted=True,
            is_sparse=is_sparse,
            n_sub_estimators=n_sub,
        )

    def _is_fitted(self, model: Any) -> bool:
        return any(attr.endswith("_") and not attr.startswith("__") for attr in vars(model) if not attr.endswith("__"))

    def _is_sparse(self, arr: Any) -> bool:
        try:
            from scipy import sparse
            return sparse.issparse(arr)
        except ImportError:
            return False