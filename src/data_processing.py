import os

import numpy as np
import pandas as pd

DEFAULT_BASE_LOAD = 0.15
MAX_VEHICLE_SPEED_KMH = 200.0
SPEED_LOAD_FACTOR = 0.25
PEAK_HOURS = [(7, 9), (16, 19)]
PEAK_LOAD_PENALTY = 0.15
UE_NORMALIZATION = 100.0
MAX_UE_PENALTY = 0.20
HANDOVER_LOAD_PENALTY = 0.10
MAX_PERMISSIBLE_LOAD = 0.95
DEFAULT_NOISE_FLOOR_DBM = 95.0
THROUGHPUT_FLOOR_MBPS = 0.5


def _is_peak_window(hour: int) -> bool:
    return any(start <= hour <= end for start, end in PEAK_HOURS)


def _compute_critical_workload(row: pd.Series) -> float:
    speed = row.get("car_speed_kmh", 0.0)
    hour = row.get("time_of_day_h", 12)
    ues = row.get("num_connected_ues", 0)
    is_handover = row.get("handover_active", 0) == 1

    load = DEFAULT_BASE_LOAD + (speed / MAX_VEHICLE_SPEED_KMH) * SPEED_LOAD_FACTOR
    if _is_peak_window(hour):
        load += PEAK_LOAD_PENALTY
    load += min(MAX_UE_PENALTY, (ues / UE_NORMALIZATION) * MAX_UE_PENALTY)
    if is_handover:
        load += HANDOVER_LOAD_PENALTY
    return round(min(load, MAX_PERMISSIBLE_LOAD), 3)


def _compute_effective_throughput(row: pd.Series) -> float:
    bw = row.get("mec_channel_bw_mhz", 20.0)
    rssi = row.get("mec_rssi_dbm", -70.0)
    ues = max(1, row.get("num_connected_ues", 1))
    load_ratio = 1.0 - (row.get("mec_load_pct", 0.0) / 100.0)
    snr_linear = 10 ** (max(0.0, rssi + DEFAULT_NOISE_FLOOR_DBM) / 10.0)
    spectral_eff = np.log2(1.0 + snr_linear)
    throughput = (bw * spectral_eff / ues) * load_ratio
    return round(max(THROUGHPUT_FLOOR_MBPS, throughput), 2)


def load_and_clean_dataset(
    filepath: str = "dataset/data/network_dataset.csv",
) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    for flag in ["handover_active", "mec_available"]:
        if flag in df.columns:
            df[flag] = df[flag].astype(int)

    df["u_crit_t"] = df.apply(_compute_critical_workload, axis=1)
    df["effective_throughput_mbps"] = df.apply(_compute_effective_throughput, axis=1)
    if "scenario" in df.columns:
        df = pd.get_dummies(df, columns=["scenario"], dtype=int)

    cleaned_df = df.dropna()
    output_dir = os.path.dirname("dataset/data/cleaned_network_dataset.csv")
    os.makedirs(output_dir, exist_ok=True)
    cleaned_df.to_csv("dataset/data/cleaned_network_dataset.csv", index=False)
    return cleaned_df
