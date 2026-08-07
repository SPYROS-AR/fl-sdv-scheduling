import os
import pandas as pd
import numpy as np

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
        # Base load from vehicle speed and safety tasks
        base_load = 0.15 + (row.get('car_speed_kmh', 0) / 200.0) * 0.25
        
        time_h = row.get('time_of_day_h', 12)
        is_rush_hour = (7 <= time_h <= 9) or (16 <= time_h <= 19)
        if is_rush_hour:
            base_load += 0.15
            
        base_load += min(0.20, (row.get('num_connected_ues', 0) / 100.0) * 0.20)
        
        if row.get('handover_active', 0) == 1:
            base_load += 0.10
            
        return round(min(base_load, 0.95), 3)

    def calculate_effective_throughput(row):
        bw_mhz = row.get('mec_channel_bw_mhz', 20.0)
        rssi_dbm = row.get('mec_rssi_dbm', -70.0)
        ues = max(1, row.get('num_connected_ues', 1))
        load_factor = 1.0 - (row.get('mec_load_pct', 0) / 100.0)
        
        snr_db = rssi_dbm + 95.0
        snr_linear = 10 ** (max(0, snr_db) / 10.0)
        
        spectral_efficiency = np.log2(1 + snr_linear)
        throughput_mbps = (bw_mhz * spectral_efficiency / ues) * load_factor
        
        return round(max(0.5, throughput_mbps), 2)  # Floor at 0.5 Mbps

    # Apply the advanced metrics
    df['u_crit_t'] = df.apply(calculate_critical_load, axis=1)
    df['effective_throughput_mbps'] = df.apply(calculate_effective_throughput, axis=1)
    
    if 'scenario' in df.columns:
        df = pd.get_dummies(df, columns=['scenario'], dtype=int)
        
    df = df.dropna()
    
    df.to_csv("dataset/data/cleaned_network_dataset.csv", index=False)
    print(f"Data cleaned successfully! Shape: {df.shape}")
    
    return df