from pathlib import Path
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHANNELS = 4  # 4 if using stacked frames, 1 if not
WINDOW_SIZE = 80  # We are using squared preprocessed images for the network, this is the size of the image in pixels.

NUM_ACTIONS = 3  # Number of actions the agent can take
INPUT_SHAPE = (CHANNELS, WINDOW_SIZE, WINDOW_SIZE)  # PyTorch uses (channels, height, width) format

LEARNING_RATE = 0.001

MIN_MEMORY_CAPACITY = 50_000  # This should be at least BATCH_SIZE
MEMORY_CAPACITY = 500_000

NUM_EPISODES = 4000
BATCH_SIZE = 32
UPDATE_FREQUENCY = 7  # How often to update the target network, measured in episodes

GAMMA = 0.99
EPSILON_MAX = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.9975

# Models saving and loading paths
MODELS_PATH = Path("models")
POLICY_NET_PATH_SKELETON = str(MODELS_PATH) + "/policy_net_"
TARGET_NET_PATH_SKELETON = str(MODELS_PATH) + "/target_net_"

# Performance plots saving paths
PERFORMANCE_PATH = Path("performance")
PERFORMANCE_PATH_SKELETON = str(PERFORMANCE_PATH) + "/performance_"
