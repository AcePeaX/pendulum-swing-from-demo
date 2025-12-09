import math
import os
import time
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask


def wrap_joint_positions(values, lower_limits, upper_limits):
    wrapped = np.array(values, dtype=float)
    two_pi = 2 * np.pi
    for idx in range(len(wrapped)):
        lower = lower_limits[idx]
        upper = upper_limits[idx]
        if not np.isfinite(lower) or not np.isfinite(upper):
            continue
        if lower >= upper:
            continue
        range_val = upper - lower
        if range_val >= two_pi - 1e-6:
            while wrapped[idx] < lower:
                wrapped[idx] += two_pi
            while wrapped[idx] > upper:
                wrapped[idx] -= two_pi
        else:
            if wrapped[idx] < lower:
                wrapped[idx] = lower
            elif wrapped[idx] > upper:
                wrapped[idx] = upper
    return wrapped


Y_MAX = 0.45
Y_LIMIT = 0.5
ANGLE_FAILURE_LIMIT = math.radians(20.0)
SWING_REWARD_LIMIT = math.radians(20.0)
POSITION_PENALTY_GAIN = 3.0
STATE_SIZE = 14  # pend angle/vel + (pend pos, pend lin vel, ee pos, ee vel)


@dataclass
class Task7Observation:
    vector: np.ndarray
    pendulum_angle: float
    pendulum_velocity: float
    end_effector_position: np.ndarray


class Task7PendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        max_steps=600,
        should_balance=True,
        gui=False,
        sim_substeps=6,
        penalize_position=True,
        render_mode=None,
        initial_pendulum_angle=None,
        continuous_actions=False,
    ):
        super().__init__()
        self.gui = gui
        self.should_balance = should_balance
        self.max_steps = max_steps
        self.sim_substeps = sim_substeps
        self.penalize_position = penalize_position
        self.render_mode = render_mode
        self.initial_pendulum_angle = (
            float(initial_pendulum_angle) if initial_pendulum_angle is not None else None
        )
        self.continuous_actions = bool(continuous_actions)
        self.dt = 1.0 / 240.0
        self.ik_integration_gain = 9.0
        self.action_speed_scale = 0.0
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._build_world()
        if self.continuous_actions:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_SIZE,), dtype=np.float32
        )
        self.current_y_target = 0.0
        self.step_count = 0
        self.reset()

    def _build_world(self):
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        robot_urdf_path = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
        self.robot_id = p.loadURDF(robot_urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
        self.camera_distance = 2.032
        self.camera_yaw = 89.2
        self.camera_pitch = -21.4
        self.camera_target = [0.0, 0.0, 0.5]
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=self.camera_distance,
                cameraYaw=self.camera_yaw,
                cameraPitch=self.camera_pitch,
                cameraTargetPosition=self.camera_target,
            )

        self.target_x = 0.5
        self.target_z = 0.6
        self.default_joint_pose = np.array(
            [0.5499, 0.3302, -0.7612, -1.5311, 0.4585, 1.4043, -0.0314], dtype=float
        )

        n_joints = p.getNumJoints(self.robot_id)
        self.joint_name_to_index = {}
        self.link_name_to_index = {}
        for i in range(n_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode()
            link_name = info[12].decode()
            if info[2] == p.JOINT_REVOLUTE:
                self.joint_name_to_index[joint_name] = i
            self.link_name_to_index[link_name] = i

        pendulum_urdf_path = os.path.abspath("pendulum.urdf")
        self.pendulum_id = p.loadURDF(
            pendulum_urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=False,
        )
        self.pendulum_parent_link = self.joint_name_to_index.get(
            "lbr_iiwa_joint_7", n_joints - 1
        )
        anchor_state = p.getLinkState(
            self.robot_id, self.pendulum_parent_link, computeForwardKinematics=True
        )
        anchor_pos = anchor_state[4]
        p.resetBasePositionAndOrientation(self.pendulum_id, anchor_pos, [0, 0, 0, 1])
        pend_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.pendulum_parent_link,
            childBodyUniqueId=self.pendulum_id,
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(pend_constraint, maxForce=1e6)
        p.changeDynamics(self.pendulum_id, -1, linearDamping=0.11, angularDamping=0.7)
        p.setCollisionFilterPair(self.pendulum_id, self.plane_id, -1, -1, enableCollision=0)
        for link_idx in range(-1, n_joints):
            p.setCollisionFilterPair(self.robot_id, self.pendulum_id, link_idx, -1, enableCollision=0)

        self.pend_axis_angle = 0.0
        self.ee_reference_position = np.zeros(3)
        self.pendulum_down_axis_angle = 0.0

        self.full_model = pin.buildModelFromUrdf(robot_urdf_path)
        self.full_data = self.full_model.createData()
        self.reduced_model = self.full_model
        self.reduced_data = self.reduced_model.createData()
        self.configuration = pink.Configuration(
            self.reduced_model, self.reduced_data, pin.neutral(self.reduced_model)
        )
        self.model_joint_names = list(self.reduced_model.names)[1:]
        self.joint_ids = [self.joint_name_to_index[name] for name in self.model_joint_names]
        self.full_joint_names = list(self.full_model.names)[1:]
        self.full_arm_indices = [self.full_joint_names.index(name) for name in self.model_joint_names]
        self.lower_limits = self.reduced_model.lowerPositionLimit
        self.upper_limits = self.reduced_model.upperPositionLimit

        for j in self.joint_ids:
            p.setJointMotorControl2(self.robot_id, j, controlMode=p.VELOCITY_CONTROL, force=0)

        for link_idx in range(-1, n_joints):
            p.changeDynamics(self.robot_id, link_idx, jointDamping=0.05)

        Kp_base = np.array([120, 80, 60, 45, 30, 20, 12], dtype=float)
        Kd_base = np.array([18, 14, 12, 10, 8, 6, 4], dtype=float)
        controller_gain_scale = 0.6
        self.Kp = controller_gain_scale * Kp_base[: len(self.joint_ids)]
        self.Kd = controller_gain_scale * Kd_base[: len(self.joint_ids)]
        self.max_tau = np.array([180, 150, 120, 90, 60, 40, 25], dtype=float)[: len(self.joint_ids)]

        self.ee_task = FrameTask("lbr_iiwa_link_7", position_cost=1.0, orientation_cost=0.0)
        self.ee_task.gain = 25.0
        self.posture_task = PostureTask(cost=1e-4)
        self.tasks = [self.ee_task, self.posture_task]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_y_target = 0.0
        self.action_speed_scale = 0.0
        self._reset_environment_state()
        obs = self._get_observation()
        return obs.vector.copy(), {"state": obs}

    def _reset_environment_state(self):
        for idx, joint in enumerate(self.joint_ids):
            p.resetJointState(
                self.robot_id,
                joint,
                targetValue=float(self.default_joint_pose[idx]),
                targetVelocity=0.0,
            )
        anchor = p.getLinkState(
            self.robot_id, self.pendulum_parent_link, computeForwardKinematics=True
        )
        anchor_pos = anchor[4]
        if self.initial_pendulum_angle is not None:
            self.pend_axis_angle = float(self.initial_pendulum_angle)
        elif self.should_balance:
            self.pend_axis_angle = math.pi + np.random.uniform(-0.03, 0.01)
        else:
            self.pend_axis_angle = self.pendulum_down_axis_angle + np.random.uniform(-0.03, 0.03)
        orientation = p.getQuaternionFromEuler([self.pend_axis_angle, 0.0, 0.0])
        p.resetBasePositionAndOrientation(self.pendulum_id, anchor_pos, orientation)
        p.resetBaseVelocity(self.pendulum_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        ee_state = p.getLinkState(
            self.robot_id, self.link_name_to_index["lbr_iiwa_link_7"], computeLinkVelocity=False
        )
        self.ee_reference_position = np.array(ee_state[0], dtype=float)

    def step(self, action):
        if self.continuous_actions:
            action_arr = np.asarray(action, dtype=float).reshape(-1)
            action_scalar = action_arr[0] if action_arr.size else 0.0
            action_value = float(np.clip(action_scalar, -1.0, 1.0))
            self.current_y_target = action_value * Y_MAX
        else:
            action_idx = int(action)
            action_value = [-1, 0, 1][action_idx]
            if action_value == -1:
                self.current_y_target = -Y_MAX
            elif action_value == 1:
                self.current_y_target = Y_MAX
            # action_value == 0 keeps the previous target
        self.action_speed_scale = float(np.clip(abs(action_value), 0.0, 1.0))

        for _ in range(self.sim_substeps):
            self._apply_control_step()

        obs = self._get_observation()
        self.step_count += 1

        failure = False
        success = False
        if abs(self.current_y_target) > Y_LIMIT:
            failure = True
        if self.should_balance and abs(obs.pendulum_angle) > ANGLE_FAILURE_LIMIT:
            failure = True
        if self.step_count >= self.max_steps:
            success = True

        terminated = failure or success
        if failure:
            reward = 0.0
        elif self.should_balance:
            reward = 1.0
        elif abs(obs.pendulum_angle) < SWING_REWARD_LIMIT:
            reward = 1.0
        else:
            reward = 0.0

        if self.penalize_position:
            distance = float(np.linalg.norm(obs.end_effector_position))
            position_penalty = min(0.7, POSITION_PENALTY_GAIN * distance)
            reward -= position_penalty

        info = {
            "failure": failure,
            "success": success,
            "state": obs,
            "y_target": self.current_y_target,
        }
        return obs.vector.copy(), reward, terminated, False, info

    def _apply_control_step(self):
        q = np.array([p.getJointState(self.robot_id, j)[0] for j in self.joint_ids], dtype=float)
        q = wrap_joint_positions(q, self.lower_limits, self.upper_limits)
        qd = np.array([p.getJointState(self.robot_id, j)[1] for j in self.joint_ids], dtype=float)

        q_model = np.array(
            [p.getJointState(self.robot_id, self.joint_name_to_index[name])[0] for name in self.model_joint_names],
            dtype=float,
        )
        q_model = wrap_joint_positions(q_model, self.lower_limits, self.upper_limits)
        q_full = np.array(
            [p.getJointState(self.robot_id, self.joint_name_to_index[name])[0] for name in self.full_joint_names],
            dtype=float,
        )
        q_full = wrap_joint_positions(
            q_full, self.full_model.lowerPositionLimit, self.full_model.upperPositionLimit
        )

        self.configuration = pink.Configuration(self.reduced_model, self.reduced_data, q_model.copy())
        self.posture_task.set_target(self.configuration.q.copy())

        target_pos = np.array([self.target_x, self.current_y_target, self.target_z], dtype=float)
        T_target = pin.SE3(np.eye(3), target_pos)
        self.ee_task.set_target(T_target)

        dq = pink.solve_ik(self.configuration, self.tasks, self.dt, solver="quadprog")
        integration_factor = self.dt * self.ik_integration_gain * self.action_speed_scale
        if integration_factor > 0.0:
            self.configuration.integrate_inplace(dq, integration_factor)
        q_des = wrap_joint_positions(self.configuration.q.copy(), self.lower_limits, self.upper_limits)
        v_des = np.zeros_like(q_des)

        tau_full = pin.rnea(
            self.full_model, self.full_data, q_full, np.zeros(self.full_model.nv), np.zeros(self.full_model.nv)
        )
        tau_g = np.array([tau_full[idx] for idx in self.full_arm_indices], dtype=float)
        tau = tau_g + self.Kp * (q_des - q) + self.Kd * (v_des - qd)
        tau = np.clip(tau, -self.max_tau, self.max_tau)

        for idx, j in enumerate(self.joint_ids):
            p.setJointMotorControl2(self.robot_id, j, controlMode=p.TORQUE_CONTROL, force=float(tau[idx]))

        p.stepSimulation()
        self._enforce_single_axis_rotation()
        if self.gui:
            time.sleep(self.dt)

    def _enforce_single_axis_rotation(self):
        pend_pos, _ = p.getBasePositionAndOrientation(self.pendulum_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.pendulum_id)
        self.pend_axis_angle += ang_vel[0] * self.dt
        constrained_quat = p.getQuaternionFromEuler([self.pend_axis_angle, 0.0, 0.0])
        p.resetBasePositionAndOrientation(self.pendulum_id, pend_pos, constrained_quat)
        constrained_ang_vel = [ang_vel[0], 0.0, 0.0]
        p.resetBaseVelocity(self.pendulum_id, lin_vel, constrained_ang_vel)

    def _get_observation(self) -> Task7Observation:
        pendulum_pos, _ = p.getBasePositionAndOrientation(self.pendulum_id)
        pendulum_lin_vel, pendulum_ang_vel = p.getBaseVelocity(self.pendulum_id)
        pendulum_angle = self.pend_axis_angle - math.pi
        pendulum_angle = (pendulum_angle + math.pi) % (2 * math.pi) - math.pi
        pendulum_velocity = pendulum_ang_vel[0]

        ee_state = p.getLinkState(
            self.robot_id, self.link_name_to_index["lbr_iiwa_link_7"], computeLinkVelocity=1
        )
        ee_position = np.array(ee_state[0], dtype=float) - self.ee_reference_position
        ee_velocity = np.array(ee_state[6], dtype=float)

        obs_vector = np.concatenate(
            [
                [pendulum_angle, pendulum_velocity],
                np.array(pendulum_pos, dtype=float),
                np.array(pendulum_lin_vel, dtype=float),
                ee_position,
                ee_velocity,
            ]
        ).astype(np.float32)

        return Task7Observation(
            vector=obs_vector,
            pendulum_angle=pendulum_angle,
            pendulum_velocity=pendulum_velocity,
            end_effector_position=ee_position.copy(),
        )

    def render(self, mode=None):
        mode = mode or self.render_mode or "human"
        if mode == "rgb_array":
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.camera_target,
                distance=self.camera_distance,
                yaw=self.camera_yaw,
                pitch=self.camera_pitch,
                roll=0.0,
                upAxisIndex=2,
            )
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=60.0,
                aspect=16.0 / 9.0,
                nearVal=0.05,
                farVal=5.0,
            )
            width = 960
            height = 540
            renderer = p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
            _, _, rgba, _, _ = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=renderer,
            )
            return np.reshape(rgba, (height, width, 4))
        return None

    def close(self):
        if p.isConnected():
            p.disconnect()

    def set_initial_pendulum_angle(self, angle=None):
        """Set or clear a deterministic initial pendulum angle (radians) for resets."""
        if angle is None:
            self.initial_pendulum_angle = None
        else:
            self.initial_pendulum_angle = float(angle)


if __name__ == "__main__":
    env = Task7PendulumEnv(gui=True, should_balance=True)
    obs, info = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        if done:
            obs, info = env.reset()
