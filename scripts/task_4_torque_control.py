import pybullet as p
import pybullet_data
import numpy as np
import time

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

q_des = []
for j in joint_ids:
    state = p.getJointState(robot_id, j)
    q_des.append(state[0])
q_des = np.array(q_des)
q_des[1] = 3.14/2
q_des[3] = 3.14/2
q_des[4] = -3.14/2
q_des[5] = 3.14/2
#q_des = np.array([0.5, 0.3, 0.0, -1.0, 0.0, 1.0, 0.0])
v_des = np.zeros_like(q_des)


Kp = np.array([120, 30, 25, 20, 15, 10, 6], dtype=float)
Kd = np.array([12, 10, 8, 6, 4, 3, 2], dtype=float)
max_tau = np.array([120, 100, 80, 60, 30, 20, 10], dtype=float)


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