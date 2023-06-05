"""
Module for managing the cylinder detector data.
"""

import threading
from typing import List

import rospy

from geometry_msgs.msg import Pose

from map_manager import MapManager
from combined.msg import CylinderGreetInstructions


class Cylinder:
    """
    Class representing a cylinder.
    """

    def __init__(self, pose: Pose, color: str, cylinder_id: int, greet_pose: Pose) -> None:
        self.cylinder_pose = pose
        self.cylinder_color = color
        self.cylinder_id = cylinder_id
        self.cylinder_greet_pose = greet_pose


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
        self.detected_cylinders = []
        self.detected_cylinders_lock = threading.Lock()

        self.cylinder_list: List[Cylinder] = []

        self.cylinder_coords = []
        self.cylinder_colors = []
        self.cylinder_greet_poses = []
        # self.num_all_cylinders = 10
        self.num_all_cylinders = 4
        self.all_cylinders_found = False
        self.num_discovered_cylinders = 0
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

        # compute greet location and orientation
        x_cylinder = cylinder_pose.position.x
        y_cylinder = cylinder_pose.position.y

        cylinder_greet_pose = self.map_manager.get_object_greet_pose(
            x_cylinder, y_cylinder, erosion=7
        )

        self.cylinder_coords.append(cylinder_pose)
        self.cylinder_colors.append(cylinder_color)
        self.cylinder_greet_poses.append(cylinder_greet_pose)

        new_cylinder = Cylinder(
            pose=cylinder_pose,
            color=cylinder_color,
            greet_pose=cylinder_greet_pose,
            cylinder_id=data.object_id,
        )
        rospy.loginfo(
            f"New cylinder with color {cylinder_color} and id {data.object_id} added to list"
        )
        self.cylinder_list.append(new_cylinder)

        self.num_discovered_cylinders = self.num_discovered_cylinders + 1

        if self.num_discovered_cylinders >= self.num_all_cylinders:
            self.all_cylinders_found = True

    def have_cylinders_to_visit(self):
        """
        Returns true if there are still cylinders to visit.
        """

        return len(self.cylinder_coords) > 0
