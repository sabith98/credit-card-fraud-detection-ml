import logging
import pandas as pd
from zenml import step

from src.model_dev import LogisticRegressionModel
from sklearn.base import RegressorMixin
from sklearn.linear_model import LogisticRegression
from .config import ModelNameConfig


@step()
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
# ) -> RegressorMixin:
) -> LogisticRegression:
    """
    Trains the model on the ingested data

    Args:
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame
    """
    try:
        model = None
        if config.model_name == "LogisticRegression":
            model = LogisticRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model: {}".format(e))