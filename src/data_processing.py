import os
import numpy as np
import pandas as pd

# Extracted Magic Numbers
BASE_LOAD_DEFAULT = 0.15
MAX_SPEED_KMH = 200.0
SPEED_PENALTY_WEIGHT = 0.25
PEAK_HOURS = [(7, 9), (16, 19)]
PEAK_PENALTY = 0.15
UE_SCALE = 100.0
UE_PENALTY_MAX = 0.20
HANDOVER_PENALTY = 0.10
MAX_LOAD_CAP = 0.95
NOISE_FLOOR_DBM = 95.0
MIN_THROUGHPUT_MBPS = 0.5

def _is_peak_hour(time_h: int) -> bool:
    return any(start <= time_h <= end for start, end in PEAK_HOURS)

def _calculate_critical_load(row) -> float:
    base_load = BASE_LOAD_DEFAULT + (row.get('car_speed_kmh', 0) / MAX_SPEED_KMH) * SPEED_PENALTY_WEIGHT
    
    if _is_peak_hour(row.get('time_of_day_h', 12)):
        base_load += PEAK_PENALTY
        
    base_load += min(UE_PENALTY_MAX, (row.get('num_connected_ues', 0) / UE_SCALE) * UE_PENALTY_MAX)
    
    if row.get('handover_active', 0) == 1:
        base_load += HANDOVER_PENALTY
        
    return round(min(base_load, MAX_LOAD_CAP), 3)

def _calculate_effective_throughput(row) -> float:
    bw_mhz = row.get('mec_channel_bw_mhz', 20.0)
    rssi_dbm = row.get('mec_rssi_dbm', -70.0)
    ues = max(1, row.get('num_connected_ues', 1))
    load_factor = 1.0 - (row.get('mec_load_pct', 0) / 100.0)
    
    snr_linear = 10 ** (max(0, rssi_dbm + NOISE_FLOOR_DBM) / 10.0)
    spectral_efficiency = np.log2(1 + snr_linear)
    throughput_mbps = (bw_mhz * spectral_efficiency / ues) * load_factor
    
    return round(max(MIN_THROUGHPUT_MBPS, throughput_mbps), 2)

def load_and_clean_dataset(filepath="dataset/data/network_dataset.csv") -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    print(f"Loading and cleaning dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    for col in ['handover_active', 'mec_available']:
        if col in df.columns:
            df[col] = df[col].astype(int)
            
    df['u_crit_t'] = df.apply(_calculate_critical_load, axis=1)
    df['effective_throughput_mbps'] = df.apply(_calculate_effective_throughput, axis=1)
    
    if 'scenario' in df.columns:
        df = pd.get_dummies(df, columns=['scenario'], dtype=int)
        
    df = df.dropna()
    df.to_csv("dataset/data/cleaned_network_dataset.csv", index=False)
    
    print(f"Data cleaned successfully! Shape: {df.shape}")
    return df