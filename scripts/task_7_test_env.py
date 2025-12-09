import numpy as np

from task_7_env import Task7PendulumEnv


def main():
    env = Task7PendulumEnv(max_steps=200, should_balance=True, gui=True)
    obs, info = env.reset()
    print("Initial observation norm:", np.linalg.norm(obs))

    while True:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            print(
                f"Episode finished. reward={reward} steps={env.step_count} failure={info['failure']} success={info['success']}"
            )
            obs, info = env.reset()


if __name__ == "__main__":
    main()
