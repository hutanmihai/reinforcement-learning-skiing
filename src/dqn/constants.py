from pathlib import Path
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_ACTIONS = 3
INPUT_SHAPE = (1, 80, 80)  # PyTorch uses (channels, height, width) format

# TODO: Fine tuning
LEARNING_RATE = 1e-2
MEMORY_CAPACITY = 32
NUM_EPISODES = 10
BATCH_SIZE = 16
UPDATE_FREQUENCY = 20

# These might be good
GAMMA = 0.99
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99

MODELS_PATH = Path("models")
POLICY_NET_PATH = MODELS_PATH / "policy_net.pth"
TARGET_NET_PATH = MODELS_PATH / "target_net.pth"
MODEL_PATH = MODELS_PATH / "model.pth"
