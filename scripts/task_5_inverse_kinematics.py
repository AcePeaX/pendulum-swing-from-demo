import pybullet as p
import pybullet_data
import numpy as np
import time


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


# Load robot
robot_id = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True
)
robot_urdf_path = pybullet_data.getDataPath() + "/kuka_iiwa/model.urdf"

x_slider = p.addUserDebugParameter("Target x", -2, 2, 0.5)
y_slider = p.addUserDebugParameter("Target y", -2, 2, 0.5)
z_slider = p.addUserDebugParameter("Target z",  0.02, 2.0, 0.6)


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
joint_ids = []

for i in range(n_joints):
    info = p.getJointInfo(robot_id, i)
    if info[2] == p.JOINT_REVOLUTE:
        joint_ids.append(i)

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




Kp = np.array([120, 30, 25, 20, 15, 10, 6], dtype=float)
Kd = np.array([12, 10, 8, 6, 4, 3, 2], dtype=float) * 1.5
max_tau = np.array([120, 100, 80, 60, 30, 20, 10], dtype=float)

model = pin.buildModelFromUrdf(robot_urdf_path)

data = model.createData()

q0 = pin.neutral(model)
configuration = pink.Configuration(model, data, q0)

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
    q = []
    qd = []
    for j in joint_ids:
        state = p.getJointState(robot_id, j)
        q.append(state[0])
        qd.append(state[1])

    q = np.array(q)
    qd = np.array(qd)

    posture_task.set_target(configuration.q.copy())

    configuration = pink.Configuration(model, data, q.copy())

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
    q_des = configuration.q[:n].copy()

    v_des = np.zeros_like(q_des)

    tau_g = p.calculateInverseDynamics(
        robot_id,
        q.tolist(),
        [0.0] * n,
        [0.0] * n
    )
    tau_g = np.array(tau_g)

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