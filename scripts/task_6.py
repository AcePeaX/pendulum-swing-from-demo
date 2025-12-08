import os
import textwrap


def main():
    pendulum_length = 0.6
    pendulum_radius = 0.025
    pendulum_mass = 0.5

    pendulum_urdf = textwrap.dedent(
        f"""\
        <?xml version="1.0" ?>
        <robot name="decoupled_pendulum">
          <link name="base">
            <visual>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <geometry>
                <cylinder length="0.05" radius="0.02"/>
              </geometry>
              <material name="DarkRed">
                <color rgba="0.6 0.1 0.1 1"/>
              </material>
            </visual>
            <inertial>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <mass value="0.01"/>
              <inertia ixx="1e-4" ixy="0" ixz="0"
                       iyy="1e-4" iyz="0" izz="1e-4"/>
            </inertial>
          </link>

          <joint name="base_to_rod" type="fixed">
            <parent link="base"/>
            <child link="rod"/>
            <origin xyz="0 0 0" rpy="0 0 0"/>
          </joint>

          <link name="rod">
            <inertial>
              <origin xyz="0 0 {-pendulum_length * 0.5:.3f}" rpy="0 0 0"/>
              <mass value="{pendulum_mass}"/>
              <inertia ixx="0.003" ixy="0" ixz="0"
                       iyy="0.003" iyz="0" izz="0.003"/>
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
                <sphere radius="{pendulum_radius * 1.8:.3f}"/>
              </geometry>
              <material name="Yellow">
                <color rgba="1 0.9 0 1"/>
              </material>
            </visual>
          </link>
        </robot>
        """
    )

    filepath = os.path.abspath("pendulum.urdf")
    with open(filepath, "w") as f:
        f.write(pendulum_urdf)
    print(f"Wrote pendulum URDF to {filepath}")


if __name__ == "__main__":
    main()
