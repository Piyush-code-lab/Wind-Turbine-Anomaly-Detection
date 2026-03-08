import os
import logging
import pandas as pd
from datetime import datetime
import numpy as np

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def preprocess_data(df):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('is_anomaly')
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        for col in numeric_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower, upper)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        return df
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main():
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        data = pd.read_csv('./data/raw/data.csv')
        logger.debug('Data loaded properly')
        # Transform the data
        processed_data = preprocess_data(data)
        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        processed_data.to_csv(os.path.join(data_path, "processed_data.csv"), index=False)
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()