import timeit

import numpy as np
import torch
from gymnasium import make, Env
from src.dqn.agent import Agent
from src.dqn.constants import MEMORY_CAPACITY, MODEL_PATH, NUM_EPISODES, MODELS_PATH, UPDATE_FREQUENCY, BATCH_SIZE
from src.utils.helpers import show_image, check_if_dirs_exist
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
    # we use BATCH_SIZE because we need at least BATCH_SIZE samples to start learning,
    # no need to wait for MEMORY_CAPACITY to fill all the way
    for _ in range(BATCH_SIZE):
        state = reset(env)
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = step(env, action)
            agent.replay_memory.store(state, action, reward, done, next_state)
            state = next_state


def train(
    env,
    agent: Agent,
):
    init_memory(env, agent)
    print(f"Memory initialized with {BATCH_SIZE} samples! The training shall begin! Let's rock!")

    reward_history = []
    best_score = -np.inf

    for episode in range(NUM_EPISODES):
        start_time = timeit.default_timer()
        state = reset(env)
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = step(env, action)
            agent.replay_memory.store(state, action, reward, done, next_state)
            agent.please_learn()

            state = next_state
            episode_reward += reward

        agent.decay_epsilon()
        reward_history.append(episode_reward)

        current_avg_score = np.mean(reward_history[-20:])  # moving average over last 20 episodes
        avg_loss_this_episode = agent.total_loss / max(agent.learns_this_episode, 1)

        print(
            f"Episode {episode + 1} | Reward: {episode_reward} | Avg. Reward: {current_avg_score} | Epsilon: {agent.epsilon} | Avg. Loss: {avg_loss_this_episode}"
        )
        print(f"Episode {episode + 1} took {timeit.default_timer() - start_time} seconds.")
        print("-" * 100)

        agent.set_loss(0.0)
        agent.set_learns_this_episode(0)

        if current_avg_score > best_score:
            best_score = current_avg_score
            check_if_dirs_exist([MODELS_PATH])
            torch.save(agent.policy_net.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    env: Env = make("SkiingNoFrameskip-v4")
    agent = Agent(action_space=env.action_space)
    train(env, agent)
    agent.save()
