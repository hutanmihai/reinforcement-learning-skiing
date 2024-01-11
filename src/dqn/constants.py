from pathlib import Path
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHANNELS = 4  # 4 if using stacked frames, 1 if not
WINDOW_SIZE = 80  # We are using squared preprocessed images for the network, this is the size of the image in pixels.

NUM_ACTIONS = 3  # Number of actions the agent can take
INPUT_SHAPE = (CHANNELS, WINDOW_SIZE, WINDOW_SIZE)  # PyTorch uses (channels, height, width) format

# TODO: Fine tuning
LEARNING_RATE = 0.001
MIN_MEMORY_CAPACITY = 100_000  # This should be at least BATCH_SIZE
MEMORY_CAPACITY = 500_000
NUM_EPISODES = 200
BATCH_SIZE = 32  # How many samples to take from the replay memory when learning
UPDATE_FREQUENCY = 10  # How often to update the target network, measured in episodes

# These might be good
GAMMA = 0.99
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# Models saving and loading paths
MODELS_PATH = Path("models")
POLICY_NET_PATH = MODELS_PATH / "policy_net.pth"
TARGET_NET_PATH = MODELS_PATH / "target_net.pth"
MODEL_PATH = MODELS_PATH / "model.pth"

# Performance plots saving paths
PERFORMANCE_PATH = Path("performance")
DQN_PERFORMANCE_PATH = PERFORMANCE_PATH / "dqn_performance.html"
