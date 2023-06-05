"""
Module for managing the cylinder detector data.
"""

from typing import List, NamedTuple

import rospy

from geometry_msgs.msg import Pose

from map_manager import MapManager
from combined.msg import CylinderGreetInstructions


class Cylinder(NamedTuple):
    """
    Class Representing a Cylinder object
    """

    cylinder_id: int
    pose: Pose
    color: str
    greet_pose: Pose


class CylinderManager:
    """
    Class for managing detected cylinders.
    """

    def __init__(self, map_manager: MapManager):
        self.detected_cylinders_subscriber = rospy.Subscriber(
            "unique_cylinder_greet",
            CylinderGreetInstructions,
            self.detected_cylinder_callback,
        )
        self.detected_cylinders: List[Cylinder] = []
        self.map_manager = map_manager

    def detection_count(self) -> int:
        """
        Returns the number of detected cylinders.
        """
        return len(self.detected_cylinders)

    def detected_cylinder_callback(self, data):
        """
        Called when unique cylinder greet instructions published on topic.
        """
        cylinder_pose = data.object_pose
        cylinder_color = data.object_color

        rospy.loginfo(f"Received cylinder with color {cylinder_color}")

        x_cylinder = cylinder_pose.position.x
        y_cylinder = cylinder_pose.position.y

        cylinder_greet_pose = self.map_manager.get_object_greet_pose(
            x_cylinder, y_cylinder, erosion=7
        )

        new_cylinder = Cylinder(
            pose=cylinder_pose,
            color=cylinder_color,
            greet_pose=cylinder_greet_pose,
            cylinder_id=data.object_id,
        )

        rospy.loginfo(
            f"New cylinder with color {cylinder_color} and id {data.object_id} added to list"
        )

        self.detected_cylinders.append(new_cylinder)
