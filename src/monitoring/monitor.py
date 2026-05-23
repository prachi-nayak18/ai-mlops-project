import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelMonitor:

    def __init__(self, log_dir: str = "logs/predictions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "predictions.jsonl"
        self.reference_stats: Dict = {}

    def log_prediction(self, features: Dict, prediction: Any, proba: List = None):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "features": features,
            "prediction": prediction,
            "probabilities": proba,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def set_reference(self, df: pd.DataFrame):
        for col in df.select_dtypes(include=np.number).columns:
            self.reference_stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
            }
        logger.info(f"Reference stats set for {len(self.reference_stats)} features.")

    def detect_drift(self, df: pd.DataFrame) -> Dict[str, bool]:
        drift_report = {}
        if not self.reference_stats:
            logger.warning("No reference stats. Run set_reference() first.")
            return drift_report

        for col in df.select_dtypes(include=np.number).columns:
            if col not in self.reference_stats:
                continue
            ref = self.reference_stats[col]
            current_mean = df[col].mean()
            z_score = abs(current_mean - ref["mean"]) / (ref["std"] + 1e-8)
            drift_detected = z_score > 3.0
            drift_report[col] = drift_detected
            if drift_detected:
                logger.warning(f"⚠️  Drift in '{col}' | z={z_score:.2f}")

        logger.info(f"Drift: {sum(drift_report.values())}/{len(drift_report)} features.")
        return drift_report