import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataProcessingStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # separating the data for analysis
            legit = data[data.Class == 0]
            fraud = data[data.Class == 1]

            legit_sample = legit.sample(n=492)
            new_dataset = pd.concat([legit_sample, fraud], axis=0)

            return new_dataset

        except Exception as e:
            logging.error(e)
            raise e
        
class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing dataset into train and test sets
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test
        """
        try:
            pass
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e
        
class DataCleaning:
    """
    class for cleaning data which process the data and divides it into train and test
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """handle data"""
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error handling data: {}".format(e))
            raise e
