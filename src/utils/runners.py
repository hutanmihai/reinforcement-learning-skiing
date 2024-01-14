from gymnasium import Env

from src.constants import MIN_MEMORY_CAPACITY, BATCH_SIZE
from src.utils.preprocessing import preprocess


def reset(env: Env):
    state, _info = env.reset()
    state = preprocess(state)
    return state


def step(env: Env, action: int):
    next_state, reward, terminated, truncated, info = env.step(action)
    next_state = preprocess(next_state)
    done = terminated or truncated
    return next_state, reward, done, info


def init_memory(env: Env, agent):
    while len(agent.replay_memory) < max(MIN_MEMORY_CAPACITY, BATCH_SIZE):
        state = reset(env)
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = step(env, action)
            agent.replay_memory.store(state, action, reward, done, next_state)
            state = next_state


def init_new_episode(env: Env, agent):
    state = reset(env)
    agent.set_loss(0.0)
    rewards = 0.0

    next_state, reward, done, info = step(env, 0)
    agent.replay_memory.store(state, 0, reward, done, next_state)
    rewards += reward
    next_state, reward, done, info = step(env, 0)
    agent.replay_memory.store(state, 0, reward, done, next_state)
    rewards += reward
    next_state, reward, done, info = step(env, 0)
    agent.replay_memory.store(state, 0, reward, done, next_state)
    rewards += reward
    state = next_state

    return state, rewards, done
