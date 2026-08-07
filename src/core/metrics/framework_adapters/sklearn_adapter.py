from typing import Any

class SklearnAdapter:
    def matches(self, model: Any) -> bool:
        return hasattr(model, "get_params") and hasattr(model, "fit")

    def count_parameters(self, model: Any) -> tuple[int, int]:
        total = 0
        if hasattr(model, "coef_"):
            total += model.coef_.size
            if hasattr(model, "intercept_"):
                total += model.intercept_.size
        elif hasattr(model, "estimators_"):
            total += sum(e.tree_.node_count for e in model.estimators_ if hasattr(e, "tree_"))
        elif hasattr(model, "tree_"):
            total += model.tree_.node_count
        return total, total  

    def model_size_bytes(self, model: Any) -> int:
        import sys
        return sys.getsizeof(model) 

    def framework_name(self) -> str:
        return "scikit-learn"