import logging
import numpy as np

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.model_type = config["model"]["type"]
        self.thresholds = config["evaluation"].get("promotion_thresholds", {})

    def evaluate(self, model, X_test, y_test) -> dict:
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
        y_pred = model.predict(X_test)
        metrics = {}

        if self.model_type == "classification":
            metrics["test_accuracy"] = round(accuracy_score(y_test, y_pred), 4)
            metrics["test_f1_weighted"] = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
        else:
            metrics["test_rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
            metrics["test_r2"] = round(r2_score(y_test, y_pred), 4)

        metrics["promote"] = self._should_promote(metrics)
        logger.info(f"Eval metrics: {metrics}")
        return metrics

    def _should_promote(self, metrics):
        for k, v in self.thresholds.items():
            if metrics.get(f"test_{k}", 0) < v:
                return False
        return True