import pandas as pd
import os
from sklearn.model_selection import TimeSeriesSplit
import logging
import yaml
from sklearn.preprocessing import RobustScaler


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url: str) -> pd.DataFrame:
    """
    Loads CSVs with optimized dtypes for memory efficiency.
    """
    try:
       # Define dtypes to save memory (525k rows * 5 years is manageable, but good practice)
        dtypes = {
            'gearbox_oil_temp': 'float32',
            'gearbox_bearing_temp': 'float32',
            'vibration_x': 'float32',
            'vibration_y': 'float32',
            'vibration_z': 'float32',
            'oil_pressure': 'float32',
            'particle_count': 'int16',
            'is_anomaly': 'int8'
        }
        print("📂 Loading Labeled Data...")
        df = pd.read_csv(data_url, parse_dates=['timestamp'], dtype=dtypes)
        df = df.sort_values('timestamp').reset_index(drop=True)
        print(f"✅ Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"📅 Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def save_data(data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        data.to_csv(os.path.join(raw_data_path, "data.csv"), index=False)
        
        logger.debug('data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        
        data_path = 'https://raw.githubusercontent.com/Piyush-code-lab/Wind-Turbine-Anomaly-Detection/refs/heads/main/Dataset/turbine_5yr_complex_data.csv'
        df = load_data(data_url=data_path)
        save_data(df, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()