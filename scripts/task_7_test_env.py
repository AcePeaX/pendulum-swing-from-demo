import numpy as np

from task_7_env import Task7PendulumEnv


def main():
    env = Task7PendulumEnv(max_steps=100, should_balance=False, gui=True)
    obs, info = env.reset()
    print("Initial observation norm:", np.linalg.norm(obs))
    episode_return = 0.0

    while True:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        episode_return += reward
        if done or truncated:
            print(
                f"Episode finished. total_reward={episode_return} steps={env.step_count} failure={info['failure']} success={info['success']}"
            )
            obs, info = env.reset()
            episode_return = 0.0


if __name__ == "__main__":
    main()
