import torch

# --- Simulation & Networking Constants ---
NUM_SAMPLES = 100000
NUM_VEHICLES = 10
TOTAL_BANDWIDTH = 500.0   # Cell bandwidth capacity in Mbps
MAX_ALLOWED_DRIFT = 20.0  # Max L2 distance threshold for model weights
MODEL_SIZE_MB = 50.0      # Broadcast payload size

# --- Utility Policy Hyperparameters ---
ALPHA = 0.005
BETA = 1.0
GAMMA = 0.006
DELTA = 27.0

# --- Dataset & Environment Config ---
DATASET = "uoft-cs/cifar10"
DIRICHLET_ALPHA = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Client & Server Hyperparameters ---
LEARNING_RATE = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 32
TEST_SPLIT = 0.2
SEED = 42

# --- Dynamic Policy Thresholds ---
TAU_CONTINUE = 0.1
GAIN_THRESHOLD = 0.01
BW_THRESHOLD = 20.0
EMA_WEIGHT = 0.3