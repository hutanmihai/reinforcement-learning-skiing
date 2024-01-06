# reinforcement-learning-skiing

## Required libraries

```bash
conda create -n rl-skiing python=3.11.5
conda activate rl-skiing
pip install numpy matplotlib jupyter opencv-python gymnasium
pip install "gymnasium[accept-rom-license, atari]"
```

- For windows (GPU):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

- For windows (CPU) / macos (CPU) / linux (GPU):

```bash
pip install torch torchvision
```

- For linux (CPU):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## How to run the project

### 1. Set the PYTHONPATH

- Windows - Powershell:

```bash
$env:PYTHONPATH='.' 
```

- Windows - CMD:

```bash
set PYTHONPATH=.
```

- Linux / MacOS:

```bash
export PYTHONPATH=.
```

### 2. Run the project

- Run DQN:

```bash
python src/dqn/main.py
```

- Run PPO:

```bash
python src/ppo/main.py
```