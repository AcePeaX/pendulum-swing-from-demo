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
y_slider = p.addUserDebugParameter("Target y", -0.5, 0.5, target_y_default)
reset_pendulum_slider = p.addUserDebugParameter("Reset pendulum (toggle)", -1, 1, 0)
save_state_slider = p.addUserDebugParameter("Save robot state", 0, 1, 0)
load_state_slider = p.addUserDebugParameter("Load saved state", 0, 1, 0)
reset_env_slider = p.addUserDebugParameter("Reset from saved pose", 0, 1, 0)
camera_print_slider = p.addUserDebugParameter("Print camera pose", 0, 1, 0)
reset_slider_prev = 0.0
save_slider_prev = 0.0
load_slider_prev = 0.0
reset_env_prev = 0.0
camera_print_prev = 0.0
default_joint_pose = np.array([0.5499, 0.3302, -0.7612, -1.5311, 0.4585, 1.4043, -0.0314], dtype=float)
saved_state = {
    "joint_positions": default_joint_pose.copy(),
    "joint_velocities": np.zeros_like(default_joint_pose),
    "pend_axis_angle": -0.0003,
    "pend_linear_velocity": [0.0, 0.0, 0.0],
    "pend_angular_velocity": [0.0, 0.0, 0.0],
    "pendulum_position": [0.0, 0.0, 0.0],
    "pendulum_orientation": [0.0, 0.0, 0.0, 1.0],
}


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

for i in range(n_joints):
    info = p.getJointInfo(robot_id, i)
    name = info[1].decode()
    if info[2] == p.JOINT_REVOLUTE:
        joint_name_to_index[name] = i

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
p.setJointMotorControl2(
    pendulum_id,
    0,
    controlMode=p.VELOCITY_CONTROL,
    force=0,
    targetVelocity=0
)
p.setCollisionFilterPair(robot_id, pendulum_id, -1, -1, enableCollision=0)
for link_idx in range(n_joints):
    p.setCollisionFilterPair(robot_id, pendulum_id, link_idx, -1, enableCollision=0)
p.setCollisionFilterPair(pendulum_id, plane, -1, -1, enableCollision=0)

pend_axis_angle = 0.0


def reset_pendulum_state(inverted=False):
    global pend_axis_angle
    anchor = p.getLinkState(robot_id, pendulum_parent_link, computeForwardKinematics=True)
    anchor_pos = anchor[4]
    base_angle = np.pi if inverted else 0.0
    if inverted:
        eps = np.random.uniform(-0.005, 0.005)
        pend_axis_angle = base_angle + eps
    else:
        pend_axis_angle = base_angle
    orientation = p.getQuaternionFromEuler([pend_axis_angle, 0.0, 0.0])
    p.resetBasePositionAndOrientation(pendulum_id, anchor_pos, orientation)
    p.resetBaseVelocity(pendulum_id, [0, 0, 0], [0, 0, 0])


def enforce_single_axis_rotation(dt):
    global pend_axis_angle
    pend_pos, _ = p.getBasePositionAndOrientation(pendulum_id)
    lin_vel, ang_vel = p.getBaseVelocity(pendulum_id)
    pend_axis_angle += ang_vel[0] * dt
    constrained_quat = p.getQuaternionFromEuler([pend_axis_angle, 0.0, 0.0])
    p.resetBasePositionAndOrientation(pendulum_id, pend_pos, constrained_quat)
    constrained_ang_vel = [ang_vel[0], 0.0, 0.0]
    p.resetBaseVelocity(pendulum_id, lin_vel, constrained_ang_vel)


def print_camera_pose():
    info = p.getDebugVisualizerCamera()
    dist = info[10]
    yaw = info[8]
    pitch = info[9]
    target = info[11]
    print(
        "Camera pose -> dist=%.3f yaw=%.2f pitch=%.2f target=%s" %
        (dist, yaw, pitch, target)
    )


def reset_environment_from_saved_state():
    global pend_axis_angle, saved_state
    if saved_state is None:
        print("No saved state available for environment reset.")
        return
    for idx, j in enumerate(joint_ids):
        target_q = float(saved_state["joint_positions"][idx])
        p.resetJointState(
            robot_id,
            j,
            targetValue=target_q,
            targetVelocity=0.0
        )
    anchor = p.getLinkState(robot_id, pendulum_parent_link, computeForwardKinematics=True)
    anchor_pos = anchor[4]
    pend_axis_angle = np.pi + np.random.uniform(-0.03, 0.01)
    orientation = p.getQuaternionFromEuler([pend_axis_angle, 0.0, 0.0])
    p.resetBasePositionAndOrientation(pendulum_id, anchor_pos, orientation)
    p.resetBaseVelocity(pendulum_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    print(
        "Environment reset: joints=",
        np.round(saved_state["joint_positions"], 4),
        "pend_angle=%.4f" % pend_axis_angle
    )


def capture_current_state():
    joint_positions = [p.getJointState(robot_id, j)[0] for j in joint_ids]
    joint_velocities = [p.getJointState(robot_id, j)[1] for j in joint_ids]
    pend_pos, pend_quat = p.getBasePositionAndOrientation(pendulum_id)
    pend_lin, pend_ang = p.getBaseVelocity(pendulum_id)
    return {
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "pend_axis_angle": pend_axis_angle,
        "pend_linear_velocity": pend_lin,
        "pend_angular_velocity": pend_ang,
        "pendulum_position": pend_pos,
        "pendulum_orientation": pend_quat,
    }


def apply_saved_state(state):
    global pend_axis_angle
    if state is None:
        return
    for idx, j in enumerate(joint_ids):
        q_val = state["joint_positions"][idx]
        qd_val = state["joint_velocities"][idx]
        p.resetJointState(
            robot_id,
            j,
            targetValue=float(q_val),
            targetVelocity=float(qd_val)
        )
    anchor = p.getLinkState(robot_id, pendulum_parent_link, computeForwardKinematics=True)
    anchor_pos = anchor[4]
    pend_axis_angle = state.get("pend_axis_angle", 0.0)
    orientation = p.getQuaternionFromEuler([pend_axis_angle, 0.0, 0.0])
    p.resetBasePositionAndOrientation(pendulum_id, anchor_pos, orientation)
    pend_lin = state.get("pend_linear_velocity", [0.0, 0.0, 0.0])
    pend_ang = state.get("pend_angular_velocity", [0.0, 0.0, 0.0])
    constrained_ang = [pend_ang[0], 0.0, 0.0]
    p.resetBaseVelocity(pendulum_id, pend_lin, constrained_ang)


reset_pendulum_state()


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
    reset_val = p.readUserDebugParameter(reset_pendulum_slider)
    save_val = p.readUserDebugParameter(save_state_slider)
    load_val = p.readUserDebugParameter(load_state_slider)
    reset_env_val = p.readUserDebugParameter(reset_env_slider)
    camera_val = p.readUserDebugParameter(camera_print_slider)
    if reset_val > 0.5 and reset_slider_prev <= 0.5:
        reset_pendulum_state(inverted=False)
    elif reset_val < -0.5 and reset_slider_prev >= -0.5:
        reset_pendulum_state(inverted=True)
    reset_slider_prev = reset_val
    if save_val > 0.5 and save_slider_prev <= 0.5:
        saved_state = capture_current_state()
        print(
            "State saved: joints=",
            np.round(saved_state["joint_positions"], 4),
            "pend_angle=%.4f" % saved_state["pend_axis_angle"]
        )
    save_slider_prev = save_val
    if load_val > 0.5 and load_slider_prev <= 0.5 and saved_state is not None:
        apply_saved_state(saved_state)
        print("State loaded.")
    load_slider_prev = load_val
    if reset_env_val > 0.5 and reset_env_prev <= 0.5:
        reset_environment_from_saved_state()
    reset_env_prev = reset_env_val
    if camera_val > 0.5 and camera_print_prev <= 0.5:
        print_camera_pose()
    camera_print_prev = camera_val

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

    configuration.integrate_inplace(dq, dt*5)
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
