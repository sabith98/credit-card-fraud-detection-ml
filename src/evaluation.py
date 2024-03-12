import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score


class Evaluation(ABC):
    """Abstract base class for evaluating the models."""

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass

class Accuracy_score(Evaluation):
    """
    Evaluation strategy that uses Accuracy Score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating Accuracy Score')
            acc_score = accuracy_score(y_true, y_pred)
            logging.info("Accuracy Score: {}".format(acc_score))
            return acc_score
        except Exception as e:
            logging.error("Error in calculating Accuracy Score: {}".format(e))
            raise e