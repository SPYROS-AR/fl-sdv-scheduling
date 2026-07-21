import os
import pandas as pd


def load_and_clean_dataset(filepath="dataset/data/network_dataset.csv")-> pd.DataFrame:

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    print(f"Loading and cleaning dataset from {filepath}...")
    df = pd.read_csv(filepath)

    # Convert boolean columns to integers
    bool_cols = ['handover_active', 'mec_available']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    def calculate_critical_load(row):
        load = 0.20 # Base system load
        

        load += (row['car_speed_kmh'] / 150.0) * 0.30 
        

        if row['scenario'] == 'urban_dense':
            load += 0.25
        elif row['scenario'] == 'urban_sparse':
            load += 0.15
            
        if row['handover_active'] == 1:
            load += 0.10
            
        return round(min(load, 0.95), 3)

    df['u_crit_t'] = df.apply(calculate_critical_load, axis=1)

    if 'scenario' in df.columns:
        df = pd.get_dummies(df, columns=['scenario'], dtype=int)

    df = df.dropna()
    
    df.to_csv("dataset/data/cleaned_network_dataset.csv", index=False)

    print(f"Data cleaned successfully! Shape: {df.shape}")
    return df