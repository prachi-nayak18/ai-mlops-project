import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg):
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg):
        self.warnings.append(msg)

    def summary(self):
        lines = [f"Validation {'PASSED' if self.passed else 'FAILED'}"]
        for e in self.errors:
            lines.append(f"  [ERROR] {e}")
        for w in self.warnings:
            lines.append(f"  [WARN] {w}")
        return "\n".join(lines)


class DataValidator:
    def __init__(self, config: dict):
        self.config = config
        self.target_col = config["data"].get("target_column", "target")

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport()
        if df.empty:
            report.add_error("DataFrame is empty!")
        if self.target_col not in df.columns:
            report.add_error(f"Target '{self.target_col}' not found!")
        for col, pct in df.isnull().mean().items():
            if pct > 0.5:
                report.add_warning(f"'{col}' has {pct:.1%} missing")
        if df.duplicated().sum() > 0:
            report.add_warning(f"{df.duplicated().sum()} duplicates found")
        logger.info(report.summary())
        return report