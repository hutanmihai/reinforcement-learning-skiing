# reinforcement-learning-skiing

## Required libraries

```bash
conda create -n rl-skiing python=3.11.5
conda activate rl-skiing
pip3 install numpy plotly==5.18.0 jupyter opencv-python gymnasium
pip3 install "gymnasium[accept-rom-license, atari]"
```

- For windows (GPU):

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cu121
```

- For windows (CPU) / macos (CPU) / linux (GPU):

```bash
pip3 install torch
```

- For linux (CPU):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
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