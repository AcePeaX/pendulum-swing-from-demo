import pybullet as p
import pybullet_data
import numpy as np
import time
import os


import pink
from pink.tasks import FrameTask, PostureTask
import pinocchio as pin


def wrap_joint_positions(values, lower_limits, upper_limits):
    """Project revolute joint angles inside their URDF limits."""
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

# Connect to physics
physicsClient = p.connect(p.GUI)
assert physicsClient >= 0, "PyBullet connection failed"

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(
    cameraDistance=2.432,
    cameraYaw=89.2,
    cameraPitch=-21.4,
    cameraTargetPosition=[0.0, 0.0, 0.3]
)


# Gravity
p.setGravity(0, 0, -9.81)

# Load ground
plane = p.loadURDF("plane.urdf")

robot_urdf_path = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")

# Load robot
robot_id = p.loadURDF(
    os.path.abspath(robot_urdf_path),
    basePosition=[0, 0, 0],
    useFixedBase=True
)

target_x = 0.5
target_z = 0.6
target_y_default = 0.0
default_joint_pose = np.array([0.5499, 0.3302, -0.7612, -1.5311, 0.4585, 1.4043, -0.0314], dtype=float)
y_slider = p.addUserDebugParameter("Target y", -0.5, 0.5, target_y_default)
respawn_slider = p.addUserDebugParameter("Reset environment", 0, 1, 0)
respawn_prev = 0.0


# Target desired
target_radius = 0.07

target_visual = p.createVisualShape(
    shapeType=p.GEOM_SPHERE,
    radius=target_radius,
    rgbaColor=[1, 0, 0, 1]
)

target_sphere = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=target_visual,
    basePosition=[0, 0, 0]
)

n_joints = p.getNumJoints(robot_id)
joint_name_to_index = {}
link_name_to_index = {}

for i in range(n_joints):
    info = p.getJointInfo(robot_id, i)
    name = info[1].decode()
    link_name = info[12].decode()
    if info[2] == p.JOINT_REVOLUTE:
        joint_name_to_index[name] = i
    link_name_to_index[link_name] = i

# Load standalone pendulum and attach via spherical constraint
pendulum_urdf_path = os.path.abspath("pendulum.urdf")
pendulum_id = p.loadURDF(
    pendulum_urdf_path,
    basePosition=[0, 0, 0],
    baseOrientation=[0, 0, 0, 1],
    useFixedBase=False
)
pendulum_parent_link = joint_name_to_index.get("lbr_iiwa_joint_7", n_joints - 1)

anchor_state = p.getLinkState(robot_id, pendulum_parent_link, computeForwardKinematics=True)
anchor_pos = anchor_state[4]
p.resetBasePositionAndOrientation(pendulum_id, anchor_pos, [0, 0, 0, 1])

pend_constraint = p.createConstraint(
    parentBodyUniqueId=robot_id,
    parentLinkIndex=pendulum_parent_link,
    childBodyUniqueId=pendulum_id,
    childLinkIndex=-1,
    jointType=p.JOINT_POINT2POINT,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0],
)
p.changeConstraint(pend_constraint, maxForce=1e6)
p.changeDynamics(pendulum_id, -1, linearDamping=0.11, angularDamping=0.7)
p.setCollisionFilterPair(robot_id, pendulum_id, -1, -1, enableCollision=0)
for link_idx in range(n_joints):
    p.setCollisionFilterPair(robot_id, pendulum_id, link_idx, -1, enableCollision=0)
p.setCollisionFilterPair(pendulum_id, plane, -1, -1, enableCollision=0)

pend_axis_angle = 0.0
ee_reference_position = np.array([0.50046, 0.00443, 0.58009], dtype=float)

Kp_base = np.array([120, 80, 60, 45, 30, 20, 12], dtype=float)
Kd_base = np.array([18, 14, 12, 10, 8, 6, 4], dtype=float)
controller_gain_scale = 0.6
Kp = controller_gain_scale * Kp_base
Kd = controller_gain_scale * Kd_base
max_tau = np.array([180, 150, 120, 90, 60, 40, 25], dtype=float) * 1.5
joint_damping = 0.05

full_model = pin.buildModelFromUrdf(robot_urdf_path)
full_data = full_model.createData()
full_joint_names = list(full_model.names)[1:]
reduced_model = full_model
data = reduced_model.createData()
q0 = pin.neutral(reduced_model)
configuration = pink.Configuration(reduced_model, data, q0)
model_joint_names = list(reduced_model.names)[1:]
lower_limits = reduced_model.lowerPositionLimit
upper_limits = reduced_model.upperPositionLimit
joint_ids = [joint_name_to_index[name] for name in model_joint_names]
full_arm_indices = [full_joint_names.index(name) for name in model_joint_names]
n = len(joint_ids)
print("Number of controlled joints:", n)


def enforce_single_axis_rotation(dt):
    global pend_axis_angle
    pend_pos, _ = p.getBasePositionAndOrientation(pendulum_id)
    lin_vel, ang_vel = p.getBaseVelocity(pendulum_id)
    pend_axis_angle += ang_vel[0] * dt
    constrained_quat = p.getQuaternionFromEuler([pend_axis_angle, 0.0, 0.0])
    p.resetBasePositionAndOrientation(pendulum_id, pend_pos, constrained_quat)
    constrained_ang_vel = [ang_vel[0], 0.0, 0.0]
    p.resetBaseVelocity(pendulum_id, lin_vel, constrained_ang_vel)


def reset_environment_state():
    global pend_axis_angle, ee_reference_position
    for idx, joint in enumerate(joint_ids):
        p.resetJointState(
            robot_id,
            joint,
            targetValue=float(default_joint_pose[idx]),
            targetVelocity=0.0,
        )
    anchor = p.getLinkState(robot_id, pendulum_parent_link, computeForwardKinematics=True)
    anchor_pos = anchor[4]
    pend_axis_angle = np.pi + np.random.uniform(-0.03, 0.01)
    orientation = p.getQuaternionFromEuler([pend_axis_angle, 0.0, 0.0])
    p.resetBasePositionAndOrientation(pendulum_id, anchor_pos, orientation)
    p.resetBaseVelocity(pendulum_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    ee_state = p.getLinkState(robot_id, link_name_to_index["lbr_iiwa_link_7"], computeLinkVelocity=False)
    ee_reference_position = np.array(ee_state[0], dtype=float)
    print("\n\nRESET\n")


reset_environment_state()

for j in joint_ids:
    p.setJointMotorControl2(
        robot_id,
        j,
        controlMode=p.VELOCITY_CONTROL,
        force=0
    )

for link_idx in range(-1, n_joints):
    p.changeDynamics(
        robot_id,
        link_idx,
        jointDamping=joint_damping
    )

# Align gains to controlled joints
Kp = Kp[:n]
Kd = Kd[:n]
max_tau = max_tau[:n]

ee_link_name = "pendulum_anchor_link"

ee_task = FrameTask(
    "lbr_iiwa_link_7",
    position_cost=1.0,
    orientation_cost=0.0,
)
ee_task.gain = 25.0
print("Task gain:",ee_task.gain)

# Posture regularization
posture_task = PostureTask(cost=1e-4)

tasks = [ee_task, posture_task]

print("Kp:", Kp)
print("Kd:", Kd)

dt = 1/240
debug_counter = 0
start_time = time.time()

# Simulation loop
while True:
    q_raw = np.array(
        [p.getJointState(robot_id, j)[0] for j in joint_ids],
        dtype=float
    )
    q = wrap_joint_positions(
        q_raw,
        lower_limits,
        upper_limits
    )
    qd = np.array(
        [p.getJointState(robot_id, j)[1] for j in joint_ids],
        dtype=float
    )

    q_model = np.array(
        [p.getJointState(robot_id, joint_name_to_index[name])[0] for name in model_joint_names],
        dtype=float
    )
    q_model = wrap_joint_positions(q_model, lower_limits, upper_limits)
    q_full = np.array(
        [p.getJointState(robot_id, joint_name_to_index[name])[0] for name in full_joint_names],
        dtype=float
    )
    q_full = wrap_joint_positions(q_full, full_model.lowerPositionLimit, full_model.upperPositionLimit)

    configuration = pink.Configuration(reduced_model, data, q_model.copy())
    posture_task.set_target(configuration.q.copy())

    # Control the visual target
    x_des = target_x
    y_des = p.readUserDebugParameter(y_slider)
    z_des = target_z
    respawn_val = p.readUserDebugParameter(respawn_slider)
    if respawn_val > 0.5 and respawn_prev <= 0.5:
        reset_environment_state()
    respawn_prev = respawn_val

    pendulum_pos, pendulum_quat = p.getBasePositionAndOrientation(pendulum_id)
    pendulum_lin_vel, pendulum_ang_vel = p.getBaseVelocity(pendulum_id)
    pendulum_angle = pend_axis_angle - np.pi
    pendulum_angle = (pendulum_angle + np.pi) % (2 * np.pi) - np.pi
    pendulum_velocity = pendulum_ang_vel[0]

    ee_state = p.getLinkState(robot_id, link_name_to_index["lbr_iiwa_link_7"], computeLinkVelocity=1)
    ee_position = np.array(ee_state[0], dtype=float) - ee_reference_position
    ee_velocity = np.array(ee_state[6], dtype=float)

    env_state = {
        "pendulum_angle": pendulum_angle,
        "pendulum_velocity": pendulum_velocity,
        "pendulum_position": np.array(pendulum_pos, dtype=float),
        "pendulum_linear_velocity": np.array(pendulum_lin_vel, dtype=float),
        "end_effector_position": ee_position,
        "end_effector_velocity": ee_velocity,
    }

    print(
        "state | pend_theta=%.3f pend_theta_dot=%.3f ee_pos=%s ee_vel=%s" % (
            env_state["pendulum_angle"],
            env_state["pendulum_velocity"],
            np.round(env_state["end_effector_position"], 5),
            np.round(env_state["end_effector_velocity"], 5),
        )
    )

    target_pos = [x_des, y_des, z_des]

    # --- Move the visual target sphere ---
    p.resetBasePositionAndOrientation(
        target_sphere,
        target_pos,
        [0, 0, 0, 1]
    )

    T_target = pin.SE3(
        np.eye(3),
        np.array(target_pos, dtype=np.float64)
    )

    ee_task.set_target(T_target)

    dq = pink.solve_ik(
        configuration,
        tasks,
        dt,
        solver="quadprog"
    )

    configuration.integrate_inplace(dq, dt*9)
    q_wrapped = wrap_joint_positions(configuration.q, lower_limits, upper_limits)
    configuration.q = q_wrapped
    q_des = configuration.q.copy()

    v_des = np.zeros_like(q_des)

    tau_full = pin.rnea(
        full_model,
        full_data,
        q_full,
        np.zeros(full_model.nv),
        np.zeros(full_model.nv)
    )
    tau_g = np.array([tau_full[idx] for idx in full_arm_indices], dtype=float)

    tau = tau_g + Kp * (q_des - q) + Kd * (v_des - qd)

    tau = np.clip(tau, -max_tau, max_tau)

    for idx, j in enumerate(joint_ids):
        p.setJointMotorControl2(
            robot_id,
            j,
            controlMode=p.TORQUE_CONTROL,
            force=float(tau[idx])
        )
    p.stepSimulation()
    enforce_single_axis_rotation(dt)
    time.sleep(dt)
