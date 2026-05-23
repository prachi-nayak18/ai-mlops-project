import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.ingest import DataIngester
from src.ingestion.validate import DataValidator
from src.training.preprocess import Preprocessor
from src.training.train import ModelTrainer
from src.training.evaluate import ModelEvaluator
from src.monitoring.monitor import ModelMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 50)
    logger.info(" MLOps Pipeline Starting")
    logger.info("=" * 50)

    # Step 1 — Ingest
    logger.info(" Step 1: Data Ingestion")
    ingester = DataIngester(config)
    df = ingester.ingest()

    # Step 2 — Validate
    logger.info(" Step 2: Data Validation")
    validator = DataValidator(config)
    report = validator.validate(df)
    if not report.passed:
        raise RuntimeError(f"Validation failed:\n{report.summary()}")

    # Step 3 — Preprocess
    logger.info("  Step 3: Preprocessing")
    preprocessor = Preprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.fit_transform(df)
    preprocessor.save("models/artifacts/preprocessor.pkl")

    # Step 4 — Train
    logger.info("  Step 4: Model Training")
    trainer = ModelTrainer(config)
    model, run_id = trainer.train(X_train, y_train, X_val, y_val)

    # Step 5 — Evaluate
    logger.info(" Step 5: Evaluation")
    evaluator = ModelEvaluator(config)
    metrics = evaluator.evaluate(model, X_test, y_test)
    logger.info(f"Metrics: {metrics}")

    # Step 6 — Monitor setup
    logger.info("Step 6: Monitor Reference Set")
    import pandas as pd
    monitor = ModelMonitor()
    monitor.set_reference(pd.DataFrame(X_train))

    logger.info("=" * 50)
    logger.info(f"Pipeline Complete | Run ID: {run_id}")
    logger.info(f"Promote: {metrics.get('promote', False)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)