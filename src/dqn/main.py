import numpy as np
import gymnasium

env = gymnasium.make("ALE/Skiing-v5", render_mode="human")
env.metadata["render_fps"] = 60


def run_episode(env, policy, render=False, max_steps=10000):
    """Run a single episode with the given policy"""
    obs = env.reset()
    obs = obs[0]
    for _ in range(max_steps):
        if render:
            env.render()
            # time.sleep(0.5)
        action = policy(obs)
        next_state, reward, terminated, truncated, info = env.step(action)
        obs = next_state
    env.render()


def random_policy(obs):
    """A random policy for the Skiing environment"""
    return np.random.randint(0, 3)


run_episode(env, random_policy, render=True, max_steps=10000)
