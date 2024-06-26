import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataProcessor:
    """
    Handles data processing tasks such as loading, cleaning, transforming, and saving data with advanced logging.
    """

    def __init__(self, logger=None):
        """
        Initializes the DataProcessor with an optional logger.
        """
        self.logger = logger or logging.getLogger(__name__)

    def load_data(self, file_path: str, read_func=pd.read_csv, parse_dates=None, **kwargs):
        """
        Load data from a file using a specified pandas reading function, defaulting to read_csv.
        """
        try:
            data = read_func(file_path, parse_dates=parse_dates, **kwargs)
            self.logger.info(f"Data loaded successfully from {file_path}.")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path} due to: {e}", exc_info=True)
            return None

    def clean_data(self, df, drop_na=True, fill_na=None, replacements=None):
        """
        Clean data by handling missing values, replacing values, and providing detailed logging.
        """
        try:
            if drop_na:
                df = df.dropna()
                self.logger.info("Dropped rows with NA values.")
            if fill_na:
                df = df.fillna(fill_na)
                self.logger.info(f"Filled NA values with specified values: {fill_na}.")
            if replacements:
                df.replace(replacements, inplace=True)
                self.logger.info(f"Replaced values as per the specified replacements: {replacements}.")
            return df
        except Exception as e:
            self.logger.error("Error during data cleaning process.", exc_info=True)
            return df

    def transform_data(self, df, transformations):
        """
        Apply transformations to the DataFrame, with logging.
        """
        try:
            for column, func in transformations.items():
                if column in df.columns:
                    original_data = df[column].copy()
                    df[column] = df[column].apply(func)
                    self.logger.debug(f"Transformed column {column}.")
                else:
                    self.logger.warning(f"Column {column} not found in DataFrame.")
            return df
        except Exception as e:
            self.logger.error("Error during data transformation process.", exc_info=True)
            return df

    def save_data(self, df, file_path, index=False):
        """
        Save a DataFrame to a file, with logging.
        """
        try:
            df.to_csv(file_path, index=index)
            self.logger.info(f"Data saved to {file_path}.")
        except Exception as e:
            self.logger.error(f"Failed to save data to {file_path}: {e}", exc_info=True)

    def scale_features(self, df, columns, scaler=StandardScaler()):
        """
        Scale specified features using a given scaler (StandardScaler by default).
        """
        try:
            df[columns] = scaler.fit_transform(df[columns])
            self.logger.info(f"Features {columns} scaled using {scaler.__class__.__name__}.")
            return df
        except Exception as e:
            self.logger.error(f"Failed to scale features {columns}: {e}", exc_info=True)
            return df

    def detect_outliers(self, df, column, method='z-score', threshold=3):
        """
        Detect outliers in a specified column using a given method ('z-score' or 'iqr').
        """
        try:
            if method == 'z-score':
                mean = df[column].mean()
                std = df[column].std()
                outliers = df[np.abs(df[column] - mean) > threshold * std]
            elif method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[column] < (Q1 - threshold * IQR)) | (df[column] > (Q3 + threshold * IQR))]
            else:
                self.logger.error(f"Unknown outlier detection method: {method}")
                return None
            self.logger.info(f"Outliers detected using {method} method with threshold {threshold}.")
            return outliers
        except Exception as e:
            self.logger.error(f"Failed to detect outliers in column {column}: {e}", exc_info=True)
            return None

    def remove_outliers(self, df, column, method='z-score', threshold=3):
        """
        Remove outliers from a specified column using a given method ('z-score' or 'iqr').
        """
        try:
            if method == 'z-score':
                mean = df[column].mean()
                std = df[column].std()
                df = df[np.abs(df[column] - mean) <= threshold * std]
            elif method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[column] >= (Q1 - threshold * IQR)) & (df[column] <= (Q3 + threshold * IQR))]
            else:
                self.logger.error(f"Unknown outlier removal method: {method}")
                return df
            self.logger.info(f"Outliers removed using {method} method with threshold {threshold}.")
            return df
        except Exception as e:
            self.logger.error(f"Failed to remove outliers from column {column}: {e}", exc_info=True)
            return df

# Example of setting up logging and using DataProcessor
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DataProcessorLogger")
    processor = DataProcessor(logger)
    df = processor.load_data("example_data.csv", read_func=pd.read_csv, parse_dates=['date'])
    if df is not None:
        cleaned_df = processor.clean_data(df, fill_na={'column': 0}, replacements={np.nan: 'Unknown'})
        transformed_df = processor.transform_data(cleaned_df, {"column": lambda x: x * 2})
        scaled_df = processor.scale_features(transformed_df, columns=['column'])
        outliers = processor.detect_outliers(scaled_df, column='column')
        cleaned_df = processor.remove_outliers(scaled_df, column='column')
        processor.save_data(cleaned_df, "cleaned_data.csv")
