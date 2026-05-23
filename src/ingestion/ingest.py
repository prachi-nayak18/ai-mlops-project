import logging
import yaml
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class DataIngester:
    def __init__(self, config: dict):
        self.config = config
        self.data_config = config["data"]
        self.raw_path = Path(self.data_config["raw_path"])
        self.raw_path.mkdir(parents=True, exist_ok=True)

    def ingest(self) -> pd.DataFrame:
        source_type = self.data_config.get("source_type", "csv")
        source_path = self.data_config.get("source_path")
        logger.info(f"Ingesting data | source={source_type} | path={source_path}")

        if source_type == "csv":
            df = pd.read_csv(source_path)
        elif source_type == "parquet":
            df = pd.read_parquet(source_path)
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

        logger.info(f"Ingested shape: {df.shape}")
        output = self.raw_path / "ingested_data.parquet"
        df.to_parquet(output, index=False)
        logger.info(f"Saved to: {output}")
        return df


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)