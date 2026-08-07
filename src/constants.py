import torch


NUM_SAMPLES = 100000 # Total number of samples 
NUM_VEHICLES = 10 # Total number of vehicles
TOTAL_BANDWIDTH = 500.0  # Mbps
MAX_ALLOWED_DRIFT = 20.0  # Maximum allowed drift

ALPHA = 0.002 # Weight for network cost
BETA = 1.0 # Weight for compute interference cost
GAMMA = 0.05 # Weight for model drift cost
DELTA =  20 # Weight for gain cost

# Dataset
DATASET = "uoft-cs/cifar10"
DIRICHLET_ALPHA = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CNN

# Conv2d: in_channels, out_channels, kernel_size
LAYER_1 = [3, 8, 3]  
LAYER_2 = [8, 16, 3]