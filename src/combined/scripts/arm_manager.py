#!/usr/bin/python3
"""
Module for moving the robot's arm.
"""
from typing import List

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String

# Defining constants
JOINT_NAMES = [
    "arm_shoulder_pan_joint",
    "arm_shoulder_lift_joint",
    "arm_elbow_flex_joint",
    "arm_wrist_flex_joint",
]


class ArmManager:
    """
    ArmMover class for moving the robot's arm.
    """

    def __init__(self):
        # Initialize publishers and subscribers
        self.arm_movement_pub = rospy.Publisher(
            "/turtlebot_arm/arm_controller/command", JointTrajectory, queue_size=1
        )

        self.arm_user_command_sub = rospy.Subscriber(
            "/arm_command", String, self.arm_command_callback
        )

        # Define the trajectories
        # "extension_direction_camera-position"
        self.commands = {
            "retract_above_down": self.define_trajectory([0, -1.3, 2.2, 1]),
            "extend_above_down": self.define_trajectory([0, -0.45, 0, 1.15]),
            "extend_front_down": self.define_trajectory([0, 1.2, 0, 0.3]),
            "wave1": self.define_wave_trajectory(
                [[-0.5, -0.45, 0, 1.15], [-0.25, -0.45, 0, 1.15], [0, -0.45, 0, 1.15]]
            ),
            "wave2": self.define_wave_trajectory(
                [[0.5, -0.45, 0, 1.15], [0.25, -0.45, 0, 1.15], [0, -0.45, 0, 1.15]]
            ),
        }

        # Define the initial arm pose
        self.arm_movement_pub.publish(self.commands["extend_above_down"])
        self.arm_pose = "extend_ring"

    def move_arm(self, command: str) -> None:
        """
        Move the arm to a given position.

        Args:
            command (str): The command to move the arm.
        """
        if command in self.commands:
            self.arm_movement_pub.publish(self.commands[command])
            self.arm_pose = command
        else:
            rospy.logerr(f"Command {command} not recognized.")

    def define_trajectory(self, positions: List[float]) -> JointTrajectory:
        """
        Define a trajectory given joint positions.
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = JOINT_NAMES
        trajectory.points = [
            JointTrajectoryPoint(positions=positions, time_from_start=rospy.Duration(1))
        ]
        return trajectory

    def define_wave_trajectory(self, positions: List[List[float]]) -> JointTrajectory:
        """
        Define a wave trajectory given joint positions.
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = JOINT_NAMES
        trajectory.points = [
            JointTrajectoryPoint(positions=pos, time_from_start=rospy.Duration((i + 1)))
            for i, pos in enumerate(positions)
        ]
        return trajectory

    def arm_command_callback(self, data: String) -> None:
        """
        Callback for the arm command subscriber.
        """
        command = data.data
        self.move_arm(command)


if __name__ == "__main__":
    rospy.init_node("arm_mover", anonymous=True)
    am = ArmManager()

    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        r.sleep()
