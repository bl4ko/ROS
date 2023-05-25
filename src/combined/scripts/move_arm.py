#!/usr/bin/python3
"""
Module for moving the robot's arm.
"""

import time
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String


# TODO: refractor: too many instance attributes pylint: disable=fixme
# pylint: disable=too-many-instance-attributes
class ArmMover:
    """
    Class for moving the robot's arm.
    """

    def __init__(self):
        # currently run from inside another node (brain.py)
        # rospy.init_node('arm_mover', anonymous=True)

        self.arm_movement_pub = rospy.Publisher(
            "/turtlebot_arm/arm_controller/command", JointTrajectory, queue_size=1
        )
        self.arm_user_command_sub = rospy.Subscriber(
            "/arm_command", String, self.new_user_command
        )

        # Just for controlling wheter to set the new arm position
        self.user_command = None
        self.send_command = False

        # Pre-defined positions for the arm
        self.retract = JointTrajectory()
        self.retract.joint_names = [
            "arm_shoulder_pan_joint",
            "arm_shoulder_lift_joint",
            "arm_elbow_flex_joint",
            "arm_wrist_flex_joint",
        ]
        self.retract.points = [
            JointTrajectoryPoint(
                positions=[0, -1.3, 2.2, 1], time_from_start=rospy.Duration(1)
            )
        ]

        self.extend = JointTrajectory()
        self.extend.joint_names = [
            "arm_shoulder_pan_joint",
            "arm_shoulder_lift_joint",
            "arm_elbow_flex_joint",
            "arm_wrist_flex_joint",
        ]
        # self.extend.points = [JointTrajectoryPoint(positions=[0,0.3,1,0],
        # self.extend.points = [JointTrajectoryPoint(positions=[0.35,0.7,0.75,0],
        self.extend.points = [
            JointTrajectoryPoint(
                positions=[0.35, 1.1, 0.5, 0], time_from_start=rospy.Duration(1)
            )
        ]

        self.extend_ring = JointTrajectory()
        self.extend_ring.joint_names = [
            "arm_shoulder_pan_joint",
            "arm_shoulder_lift_joint",
            "arm_elbow_flex_joint",
            "arm_wrist_flex_joint",
        ]
        # self.extend_ring.points = [JointTrajectoryPoint(positions=[0,-1.3,2.7,-0.5],
        #                                             time_from_start = rospy.Duration(1))]
        self.extend_ring.points = [
            JointTrajectoryPoint(
                positions=[0, -0.45, 0, 1.15], time_from_start=rospy.Duration(1)
            )
        ]

        self.extend_ring_close = JointTrajectory()
        self.extend_ring_close.joint_names = [
            "arm_shoulder_pan_joint",
            "arm_shoulder_lift_joint",
            "arm_elbow_flex_joint",
            "arm_wrist_flex_joint",
        ]
        # self.extend_ring.points = [JointTrajectoryPoint(positions=[0,-1.3,2.7,-0.5],
        #                                             time_from_start = rospy.Duration(1))]
        self.extend_ring_close.points = [
            JointTrajectoryPoint(
                positions=[0, 1.2, 0, 0.3], time_from_start=rospy.Duration(1)
            )
        ]
        self.wave1 = JointTrajectory()
        self.wave1.joint_names = [
            "arm_shoulder_pan_joint",
            "arm_shoulder_lift_joint",
            "arm_elbow_flex_joint",
            "arm_wrist_flex_joint",
        ]
        self.wave1.points = [
            JointTrajectoryPoint(
                positions=[-0.5, -0.45, 0, 1.15], time_from_start=rospy.Duration(1)
            ),
            JointTrajectoryPoint(
                positions=[-0.25, -0.45, 0, 1.15], time_from_start=rospy.Duration(2)
            ),
            JointTrajectoryPoint(
                positions=[0, -0.45, 0, 1.15], time_from_start=rospy.Duration(3)
            ),
        ]

        self.wave2 = JointTrajectory()
        self.wave2.joint_names = [
            "arm_shoulder_pan_joint",
            "arm_shoulder_lift_joint",
            "arm_elbow_flex_joint",
            "arm_wrist_flex_joint",
        ]
        self.wave2.points = [
            JointTrajectoryPoint(
                positions=[0.5, -0.45, 0, 1.15], time_from_start=rospy.Duration(1)
            ),
            JointTrajectoryPoint(
                positions=[0.25, -0.45, 0, 1.15], time_from_start=rospy.Duration(2)
            ),
            JointTrajectoryPoint(
                positions=[0, -0.45, 0, 1.15], time_from_start=rospy.Duration(3)
            ),
        ]

    def new_user_command(self, data: String) -> None:
        """
        Receives a new command from the user.

        Args:
            data (String): The new command.
        """
        self.user_command = data.data.strip()
        self.send_command = True

    def update_position(self) -> None:
        """
        Updates the arm position.
        """
        if self.send_command:
            if self.user_command == "retract":
                self.arm_movement_pub.publish(self.retract)
                print("Retracted arm!")
            elif self.user_command == "extend":
                self.arm_movement_pub.publish(self.extend)
                print("Extended arm!")
            else:
                print("Unknown instruction:", self.user_command)
            self.send_command = False


if __name__ == "__main__":
    rospy.init_node("arm_mover", anonymous=True)
    am = ArmMover()
    time.sleep(0.5)
    # am.arm_movement_pub.publish(am.retract)
    # print('Retracted arm!')
    for wave in [am.wave1, am.wave2]:
        am.arm_movement_pub.publish(wave)
        rospy.sleep(wave.points[-1].time_from_start.to_sec())
    # am.arm_movement_pub.publish(am.extend_ring)
    # print('Retracted arm!')

    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        am.update_position()
        r.sleep()
