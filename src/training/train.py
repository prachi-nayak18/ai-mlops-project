import logging
import pickle
from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

logger = logging.getLogger(__name__)

CLF = {"random_forest": RandomForestClassifier, "logistic": LogisticRegression}
REG = {"random_forest": RandomForestRegressor, "linear": LinearRegression}

try:
    from xgboost import XGBClassifier, XGBRegressor
    CLF["xgboost"] = XGBClassifier
    REG["xgboost"] = XGBRegressor
except ImportError:
    pass


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config["model"]
        self.mlflow_config = config["mlflow"]
        self.save_path = Path(self.model_config["save_path"])
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.mlflow_config["tracking_uri"])
        mlflow.set_experiment(self.mlflow_config["experiment_name"])

    def train(self, X_train, y_train, X_val, y_val):
        registry = CLF if self.model_config["type"] == "classification" else REG
        model = registry[self.model_config["algorithm"]](**self.model_config.get("hyperparameters", {}))

        with mlflow.start_run(run_name=self.mlflow_config.get("run_name", "run")) as run:
            mlflow.log_params(self.model_config.get("hyperparameters", {}))
            model.fit(X_train, y_train)
            metrics = self._metrics(model, X_val, y_val)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            logger.info(f"Metrics: {metrics}")
            with open(self.save_path, "wb") as f:
                pickle.dump(model, f)
            return model, run.info.run_id

    def _metrics(self, model, X, y):
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
        y_pred = model.predict(X)
        if self.model_config["type"] == "classification":
            return {
                "val_accuracy": round(accuracy_score(y, y_pred), 4),
                "val_f1": round(f1_score(y, y_pred, average="weighted", zero_division=0), 4),
            }
        return {
            "val_rmse": round(np.sqrt(mean_squared_error(y, y_pred)), 4),
            "val_r2": round(r2_score(y, y_pred), 4),
        }