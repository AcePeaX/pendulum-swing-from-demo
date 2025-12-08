import os
from robot_descriptions import ur3_description
import re

original_urdf_path = ur3_description.URDF_PATH
robot_root_path = os.path.dirname(os.path.dirname(original_urdf_path))


with open(original_urdf_path, 'r') as file:
    original_content = file.read()

pendulum_xml = """
  <joint name="pendulum_joint" type="revolute">
    <parent link="wrist_3_link"/>
    <child link="pendulum_link"/>
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-10.0" upper="10.0" effort="0" velocity="100"/>
    <dynamics damping="0.01" friction="0.01"/>
  </joint>

  <link name="pendulum_link">
    <inertial>
      <origin xyz="0 0 0.30" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.002" ixy="0" ixz="0"
               iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.30" radius="0.015"/>
      </geometry>
      <material name="Gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>

    <visual>
      <origin xyz="0 0 0.30" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.03"/>
      </geometry>
      <material name="Red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.30" radius="0.015"/>
      </geometry>
    </collision>
  </link>
"""


problematic_prefix = "package://example-robot-data/robots/ur_description"
fixed_content = original_content.replace(problematic_prefix, robot_root_path)

final_content = fixed_content.replace('</robot>', pendulum_xml + '\n</robot>')

new_filename = "kuka_iiwa_pendulum.urdf"
new_filepath = os.path.abspath(new_filename)

with open(new_filepath, 'w') as file:
    file.write(final_content)
