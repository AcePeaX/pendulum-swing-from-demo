import pybullet as p
import pybullet_data
import numpy as np
import time
import os


import pink
from pink.tasks import FrameTask, PostureTask
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

# Connect to physics
physicsClient = p.connect(p.GUI)
assert physicsClient >= 0, "PyBullet connection failed"

p.setAdditionalSearchPath(pybullet_data.getDataPath())


# Gravity
p.setGravity(0, 0, -9.81)

# Load ground
plane = p.loadURDF("plane.urdf")

robot_urdf_path = "kuka_iiwa_pendulum.urdf"

# Load robot
robot_id = p.loadURDF(
    os.path.abspath(robot_urdf_path),
    basePosition=[0, 0, 0],
    useFixedBase=True
)

x_slider = p.addUserDebugParameter("Target x", -2, 2, 0.2)
y_slider = p.addUserDebugParameter("Target y", -2, 2, 0.2)
z_slider = p.addUserDebugParameter("Target z",  0.02, 2.0, 0.2)


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



Kp = np.array([120, 30, 25, 20, 15, 10], dtype=float)
Kd = np.array([12, 10, 8, 6, 4, 3], dtype=float) * 1.5
max_tau = np.array([120, 100, 80, 60, 30, 20], dtype=float)

model = pin.buildModelFromUrdf(robot_urdf_path)
data = model.createData()
q0 = pin.neutral(model)
configuration = pink.Configuration(model, data, q0)
model_joint_names = list(model.names)[1:]
arm_joint_indices = [idx for idx, name in enumerate(model_joint_names) if "pendulum" not in name]
arm_joint_names = [model_joint_names[idx] for idx in arm_joint_indices]
joint_ids = [joint_name_to_index[name] for name in arm_joint_names]
n = len(joint_ids)
print("Number of controlled joints:", n)

for j in joint_ids:
    p.setJointMotorControl2(
        robot_id,
        j,
        controlMode=p.VELOCITY_CONTROL,
        force=0
    )

ee_link = joint_ids[-1]

# End-effector task
ee_link_name = p.getJointInfo(robot_id, ee_link)[12].decode("utf-8")

ee_task = FrameTask(
    ee_link_name,
    position_cost=1.0,
    orientation_cost=0.0,
)
ee_task.gain = 40.0
print("Task gain:",ee_task.gain)

# Posture regularization
posture_task = PostureTask(cost=1e-4)

tasks = [ee_task, posture_task]

print("Kp:", Kp)
print("Kd:", Kd)

dt = 1/240

# Simulation loop
while True:
    q = np.array(
        [p.getJointState(robot_id, j)[0] for j in joint_ids],
        dtype=float
    )
    qd = np.array(
        [p.getJointState(robot_id, j)[1] for j in joint_ids],
        dtype=float
    )

    q_model = np.array(
        [p.getJointState(robot_id, joint_name_to_index[name])[0] for name in model_joint_names],
        dtype=float
    )

    configuration = pink.Configuration(model, data, q_model.copy())
    posture_task.set_target(configuration.q.copy())

    # Control the visual target
    x_des = p.readUserDebugParameter(x_slider)
    y_des = p.readUserDebugParameter(y_slider)
    z_des = p.readUserDebugParameter(z_slider)

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
    q_des = configuration.q[arm_joint_indices].copy()

    v_des = np.zeros_like(q_des)

    q_idyn = [p.getJointState(robot_id, joint_name_to_index[name])[0] for name in model_joint_names]
    qd_idyn = [p.getJointState(robot_id, joint_name_to_index[name])[1] for name in model_joint_names]
    qdd_idyn = [0.0] * len(model_joint_names)

    tau_full = p.calculateInverseDynamics(
        robot_id,
        q_idyn,
        qd_idyn,
        qdd_idyn
    )
    tau_g = np.array([tau_full[idx] for idx in arm_joint_indices], dtype=float)

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
    time.sleep(dt)
