import logging
import pandas as pd
from zenml import step

from typing import Tuple
from typing_extensions import Annotated

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataProcessingStrategy


@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
        """
    Cleans the data and divides it into train and test

    Args:
        df: raw data
    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels
    """
        try:
            process_strategy = DataProcessingStrategy()
            data_cleaning = DataCleaning(df, process_strategy)
            processed_data = data_cleaning.handle_data()

            divide_strategy = DataDivideStrategy()
            data_cleaning = DataCleaning(processed_data, divide_strategy)
            X_train, X_test, y_train, y_test = data_cleaning.handle_data()
            logging.info("Data cleaning completed")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error occurred during data cleaning: {}".format(e))
            raise e