import timeit

import numpy as np
import torch
from gymnasium import make, Env
from src.dqn.agent import Agent
from src.dqn.constants import (
    MODEL_PATH,
    NUM_EPISODES,
    MODELS_PATH,
    BATCH_SIZE,
    MIN_MEMORY_CAPACITY,
)
from src.utils.helpers import check_if_dirs_exist
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


def init_memory(env: Env, agent: Agent):
    while len(agent.replay_memory) < max(MIN_MEMORY_CAPACITY, BATCH_SIZE):
        state = reset(env)
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = step(env, action)
            agent.replay_memory.store(state, action, reward, done, next_state)
            state = next_state


def train(env: Env, agent: Agent):
    init_memory(env, agent)
    print(f"Memory initialized with {BATCH_SIZE} samples! The training shall begin! Let's rock!")

    reward_history = []
    best_score = -np.inf

    for episode in range(NUM_EPISODES):
        start_time = timeit.default_timer()
        state = reset(env)
        done = False
        episode_reward = 0
        step_counter = 0
        agent.set_loss(0.0)

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = step(env, action)
            agent.replay_memory.store(state, action, reward, done, next_state)
            agent.please_learn()

            state = next_state
            episode_reward += reward
            step_counter += 1

        # We update the target net every episode, one episode has around 4k steps
        agent.update_target_net()

        agent.decay_epsilon()
        reward_history.append(episode_reward)

        current_avg_score = np.mean(reward_history[-20:])  # moving average over last 20 episodes

        print(
            f"Episode {episode + 1} | Reward: {episode_reward} | Avg Reward: {current_avg_score} | Epsilon: {agent.epsilon}"
        )
        print(f"Avg Loss: {agent.total_loss / max(step_counter, 1)} | Steps: {step_counter}")
        print(f"Episode {episode + 1} took {timeit.default_timer() - start_time} seconds.")
        print("-" * 100)

        if current_avg_score > best_score:
            best_score = current_avg_score
            check_if_dirs_exist([MODELS_PATH])
            torch.save(agent.policy_net.state_dict(), MODEL_PATH)

        agent.save()


if __name__ == "__main__":
    env: Env = make("SkiingNoFrameskip-v4")
    agent = Agent(action_space=env.action_space)
    train(env, agent)
