import os
import pybullet_data

original_urdf_path = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
robot_root_path = os.path.dirname(os.path.dirname(original_urdf_path))


with open(original_urdf_path, "r") as file:
    original_content = file.read()

pendulum_length = 0.35
pendulum_radius = 0.015
pendulum_mass = 0.5

pendulum_xml = f"""
  <link name="pendulum_anchor_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.02" radius="0.01"/>
      </geometry>
      <material name="Gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.01"/>
      <inertia ixx="1e-5" ixy="0" ixz="0"
               iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
  </link>

  <joint name="pendulum_anchor_joint" type="fixed">
    <parent link="lbr_iiwa_link_7"/>
    <child link="pendulum_anchor_link"/>
    <origin rpy="0 0 0" xyz="0 0 0"/>
  </joint>

  <joint name="pendulum_yaw_joint" type="revolute">
    <parent link="pendulum_anchor_link"/>
    <child link="pendulum_yaw_link"/>
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1000.0" upper="1000.0" effort="0" velocity="100"/>
    <dynamics damping="0.05" friction="0.01"/>
  </joint>

  <link name="pendulum_yaw_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.01"/>
      <inertia ixx="1e-5" ixy="0" ixz="0"
               iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
  </link>

  <joint name="pendulum_pitch_joint" type="revolute">
    <parent link="pendulum_yaw_link"/>
    <child link="pendulum_link"/>
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1000.0" upper="1000.0" effort="0" velocity="100"/>
    <dynamics damping="0.05" friction="0.01"/>
  </joint>

  <link name="pendulum_link">
    <inertial>
      <origin xyz="0 0 {-pendulum_length * 0.5:.3f}" rpy="0 0 0"/>
      <mass value="{pendulum_mass}"/>
      <inertia ixx="0.002" ixy="0" ixz="0"
               iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>

    <visual>
      <origin xyz="0 0 {-pendulum_length * 0.5:.3f}" rpy="0 0 0"/>
      <geometry>
        <cylinder length="{pendulum_length:.3f}" radius="{pendulum_radius:.3f}"/>
      </geometry>
      <material name="Gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>

    <visual>
      <origin xyz="0 0 {-pendulum_length:.3f}" rpy="0 0 0"/>
      <geometry>
        <sphere radius="{pendulum_radius * 1.5:.3f}"/>
      </geometry>
      <material name="Red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
  </link>

"""


problematic_prefix = "package://example-robot-data/robots/ur_description"
fixed_content = original_content.replace(problematic_prefix, robot_root_path)

final_content = fixed_content.replace('</robot>', pendulum_xml + '\n</robot>')

new_filename = "kuka_iiwa_pendulum.urdf"
new_filepath = os.path.abspath(new_filename)

with open(new_filepath, 'w') as file:
    file.write(final_content)
