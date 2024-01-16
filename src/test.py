from argparse import ArgumentParser

import torch
from gymnasium import Env, make

from src.agent import Agent
from src.main import step
from src.utils.runners import init_new_episode


def run_test(env: Env, agent: Agent, episodes_to_run: int = 100):
    agent.set_epsilon(0.1)
    agent.policy_net.eval()
    rewards_history = []
    for episode in range(episodes_to_run):
        total_reward = 0

        state, rewards, done = init_new_episode(env, agent)
        total_reward += rewards

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = step(env, action)
            agent.replay_memory.store(state, action, reward, done, next_state)
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

        print(f"Episode {episode + 1} finished with reward {total_reward}!")

    print(f"Average reward over {episodes_to_run} episodes: {sum(rewards_history) / episodes_to_run}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dqn", action="store_true", help="Flag to run in DQN mode")
    parser.add_argument("--ddqn", action="store_true", help="Flag to run in DDQN mode")
    args = parser.parse_args()

    if args.dqn:
        algorithm = "dqn"
    elif args.ddqn:
        algorithm = "ddqn"
    else:
        raise ValueError("Please specify either --dqn or --ddqn flag!")

    # To take the best results in one episode models
    policy_net_path = f"models/policy_net_{algorithm}_solo.pth"

    # To take the best average results models in 100 episodes
    # policy_net_path = f"models/policy_net_{algorithm}_avg.pth"

    env: Env = make("ALE/Skiing-v5")
    agent = Agent(action_space=env.action_space, algorithm=algorithm)
    agent.policy_net.load_state_dict(torch.load(policy_net_path, map_location=torch.device("cpu")))
    run_test(env, agent)
