from argparse import ArgumentParser

from gymnasium import Env, make

from src.agent import Agent
from src.main import reset, step


def run_test(env: Env, agent: Agent, episodes_to_run: int = 10):
    agent.set_epsilon(0)
    agent.policy_net.eval()
    for episode in range(episodes_to_run):
        state = reset(env)
        total_reward = 0

        next_state, reward, done, info = step(env, 0)
        agent.replay_memory.store(state, 0, reward, done, next_state)
        next_state, reward, done, info = step(env, 0)
        agent.replay_memory.store(state, 0, reward, done, next_state)
        next_state, reward, done, info = step(env, 0)
        agent.replay_memory.store(state, 0, reward, done, next_state)

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = step(env, action)
            agent.replay_memory.store(state, action, reward, done, next_state)
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1} finished with reward {total_reward}!")


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
    if algorithm == "ddqn":
        policy_net_path = f"models/policy_net_{algorithm}_solo.pth"
        target_net_path = f"models/target_net_{algorithm}_solo.pth"
    else:
        policy_net_path = f"models/policy_net_{algorithm}_solo.pth"
        target_net_path = None

    # To take the best average results models in 100 episodes
    # if algorithm == "ddqn":
    #     policy_net_path = f"models/policy_net_{algorithm}_avg.pth"
    #     target_net_path = f"models/target_net_{algorithm}_avg.pth"
    # else:
    #     policy_net_path = f"models/policy_net_{algorithm}_avg.pth"
    #     target_net_path = None

    env: Env = make("ALE/Skiing-v5", render_mode="human")
    agent = Agent(action_space=env.action_space, algorithm=algorithm)
    if algorithm == "ddqn":
        agent.load(policy_net_path, target_net_path)
    else:
        agent.load(policy_net_path)
    run_test(env, agent)
