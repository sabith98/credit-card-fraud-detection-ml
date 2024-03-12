import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

from sklearn.base import RegressorMixin
from sklearn.linear_model import LogisticRegression
from src.evaluation import Accuracy_score

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "accuracy_score"]
]:
    """
    Evaluates the model on the ingested data

    Args:
        model: Trained model
        X_test:  The features to predict from (testing set).
        y_test: Test labels
    """
    try:
        prediction = model.predict(X_test)

        accuracy_score_class = Accuracy_score()
        accuracy_score = accuracy_score_class.calculate_score(y_test, prediction)
        mlflow.log_metric("accuracy_score", accuracy_score)

        return (accuracy_score,)
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e