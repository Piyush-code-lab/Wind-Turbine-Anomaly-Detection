import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def create_features(df):
    """Creating New Features"""
    try:
        df = df.copy()
        sensor_cols = ['gearbox_oil_temp', 'gearbox_bearing_temp', 'vibration_x', 'vibration_y', 'vibration_z', 'oil_pressure', 'particle_count']
        
        # 1. Lag Features (Memory of the system)
        # Gearboxes have thermal inertia. What happened 1 hour ago matters.
        for col in sensor_cols:
            df[f'{col}_lag_1'] = df[col].shift(1)       # 10 mins ago
            df[f'{col}_lag_6'] = df[col].shift(6)       # 1 hour ago
            df[f'{col}_lag_144'] = df[col].shift(144)   # 24 hours ago (Daily seasonality)
        
        # 2. Rolling Statistics (Trend detection)
        # Sudden change in rolling mean indicates drift
        for col in sensor_cols:
            df[f'{col}_roll_mean_6'] = df[col].rolling(window=6).mean()
            df[f'{col}_roll_std_6'] = df[col].rolling(window=6).std()
            df[f'{col}_roll_max_24'] = df[col].rolling(window=144).max()
            
        # 3. Interaction Features (Domain Knowledge)
        # High Temp + High Vibration = Critical Failure Risk
        df['temp_vib_interaction'] = df['gearbox_bearing_temp'] * df['vibration_x']
        
        # Pressure to Temp Ratio (Lubrication efficiency)
        df['pressure_temp_ratio'] = df['oil_pressure'] / (df['gearbox_oil_temp'] + 1e-5)
        
        # 4. Rate of Change (Derivative)
        for col in sensor_cols:
            df[f'{col}_diff'] = df[col].diff()
            
        # Fill NaNs created by lag/rolling
        df = df.fillna(0) 
        
        return df
    except Exception as e:
        logger.error('ERROR during Feature Engineering: %s', e)
        raise


       

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        data = load_data('./data/interim/processed_data.csv')
        df = create_features(data)
        save_data(df, os.path.join("./data", "processed", "data.csv"))
        
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()