from gymnasium import Env, make

from src.dqn.agent import Agent
from src.dqn.main import reset, step


def run(env: Env, agent: Agent, episodes_to_run: int = 10):
    agent.set_epsilon(0.1)
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


if __name__ == '__main__':
    env: Env = make("ALE/Skiing-v5", render_mode="human")
    agent = Agent(action_space=env.action_space)
    agent.load()
    run(env, agent)
