import timeit

import numpy as np
from gymnasium import make, Env
from src.ddqn.agent import Agent
from src.ddqn.constants import (
    NUM_EPISODES,
    MODELS_PATH,
    PERFORMANCE_PATH,
    UPDATE_FREQUENCY,
)
from src.utils.helpers import check_if_dirs_exist, save_results_plot_html
from src.utils.runners import reset, step, init_memory


def train(env: Env, agent: Agent):
    init_memory(env, agent)
    print(f"Memory initialized with {len(agent.replay_memory)} samples! The training shall begin! Let's rock!")

    reward_history = []
    best_score = -np.inf
    best_avg_score = -np.inf

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
            agent.learn()

            state = next_state
            episode_reward += reward
            step_counter += 1

        if episode % UPDATE_FREQUENCY == 0:
            agent.update_target_net()

        agent.decay_epsilon()
        reward_history.append(episode_reward)

        current_avg_score = np.mean(reward_history[-100:])  # moving average over last 100 episodes

        print(
            f"Episode {episode + 1} | Reward: {episode_reward} | Avg Reward: {current_avg_score} | Epsilon: {agent.epsilon}"
        )
        print(f"Avg Loss: {agent.total_loss / max(step_counter, 1)} | Steps: {step_counter}")
        print(f"Episode {episode + 1} took {timeit.default_timer() - start_time} seconds.")
        print("-" * 100)

        if reward_history[-1] > best_score:
            best_score = reward_history[-1]
            agent.save(name_suffix="solo")

        if current_avg_score > best_avg_score:
            best_avg_score = current_avg_score
            agent.save(name_suffix="avg")

    save_results_plot_html(reward_history, agent.algorithm)


if __name__ == "__main__":
    check_if_dirs_exist([MODELS_PATH, PERFORMANCE_PATH])
    env: Env = make("ALE/Skiing-v5")
    agent = Agent(action_space=env.action_space, algorithm="ddqn")
    train(env, agent)
