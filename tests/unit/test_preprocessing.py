import numpy as np
import pandas as pd
import pytest
import yaml

from src.training.preprocess import Preprocessor


@pytest.fixture
def config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(20, 60, 200),
        "income": np.random.randint(30000, 100000, 200),
        "education": np.random.choice(["graduate", "undergraduate"], 200),
        "target": np.random.choice([0, 1], 200),
    })


def test_output_shape(config, sample_df):
    prep = Preprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test = prep.fit_transform(sample_df)
    assert X_train.shape[0] > 0
    assert X_val.shape[0] > 0
    assert X_test.shape[0] > 0


def test_no_nulls_after_preprocessing(config, sample_df):
    prep = Preprocessor(config)
    X_train, X_val, X_test, *_ = prep.fit_transform(sample_df)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()


def test_train_val_test_split_ratio(config, sample_df):
    prep = Preprocessor(config)
    X_train, X_val, X_test, *_ = prep.fit_transform(sample_df)
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(sample_df)


def test_preprocessor_save_load(config, sample_df, tmp_path):
    prep = Preprocessor(config)
    prep.fit_transform(sample_df)
    save_path = str(tmp_path / "preprocessor.pkl")
    prep.save(save_path)
    loaded = Preprocessor.load(save_path)
    assert loaded.pipeline is not None