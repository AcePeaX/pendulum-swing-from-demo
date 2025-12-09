# RoboticsRLDemoProject

## Set-up environment and install dependecies

```bash
# Set-up the environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Task 7 environment

`scripts/task_7_env.py` provides a Gymnasium (`gymnasium.Env`) wrapper around the IIWA + pendulum simulation from `task_7_env_part1.py`. Key characteristics:

- **Action space**: discrete {−1, 0, 1}. Actions set the wrist Y target to `−Y_MAX`, keep the current value, or set it to `+Y_MAX`.
- **Observation**: 14‑D vector `[pendulum_angle, pendulum_velocity, pendulum_position (xyz), pendulum_linear_velocity (xyz), end_effector_position (xyz), end_effector_velocity (xyz)]`.
- **Rewards**:
  - When `should_balance=True`, every non-terminal step yields +1 as long as the pendulum remains within ±20° of upright. Terminal failures give 0.
  - When `should_balance=False`, reward is +1 only on steps where `|pendulum_angle| < 20°`, encouraging swing-up; terminal failures give 0.
- **Termination**: fails if `|y_target| > Y_LIMIT` or (when balancing) `|pendulum_angle| > 20°`; succeeds after `max_steps` steps.

Use `scripts/task_7_test_env.py` to run a GUI demo with random actions and verify installation.
