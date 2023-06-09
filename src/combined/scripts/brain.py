#!/usr/bin/python3

# TODO: fix pylint errors for modularity in this class # pylint: disable=fixme
# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-statements, too-many-branches, too-many-public-methods, too-many-lines

"""
Module representing brain of the turtle bot.
"""

import math
import signal
import sys
import threading
from typing import Tuple, List
from types import FrameType
import numpy as np
import actionlib
import rospy
from map_manager import MapManager
from sound import SoundPlayer
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist, Pose
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker
from move_arm import ArmMover
from laser_manager import LaserManager
from nav_msgs.msg import Odometry

from dialogue import PosterDialogue, PersonDialogue
from combined.msg import (
    DetectedFaces,
    DetectedRings,
    CylinderGreetInstructions,
    UniqueRingCoords,
)
from combined.srv import IsPoster


# pylint: disable=too-few-public-methods
class Cylinder:
    """
    Class representing a cylinder.
    """

    def __init__(self, pose: Pose, color: str, cylinder_id: int, greet_pose: Pose) -> None:
        self.cylinder_pose = pose
        self.cylinder_color = color
        self.cylinder_id = cylinder_id
        self.cylinder_greet_pose = greet_pose


class MercenaryInfo:
    """
    This class represents the information that the mercenary has about the task.
    """

    def __init__(
        self,
        name: str = None,
        cylinder_color: str = None,
        ring_color: str = None,
        wanted_price: int = None,
    ) -> None:
        self.name: str = name
        self.cylinder_color: str = cylinder_color
        self.ring_color: str = ring_color
        self.wanted_price: int = wanted_price

    def __str__(self) -> str:
        return (
            f"Name: {self.name}, Cylinder color: {self.cylinder_color}, Ring color:"
            f" {self.ring_color}, Wanted price: {self.wanted_price}"
        )

    def is_complete(self) -> bool:
        """
        Returns true if mercenary has enough information to complete the task, false otherwise
        """
        return (
            self.name is not None
            and self.cylinder_color is not None
            and self.ring_color is not None
            and self.wanted_price is not None
        )

    @staticmethod
    def are_complete(mercenary_info_list) -> bool:
        """
        Returns true if all mercenaries in the list have enough
        information to complete the task, false otherwise
        """
        valid = True
        for mercenary_info in mercenary_info_list:
            if not mercenary_info.is_complete():
                valid = False
        return valid


def signal_handler(sig: signal.Signals, frame: FrameType) -> None:
    """
    Handles the SIGINT signal, which is sent when the user presses Ctrl+C.

    Args:
        sig (signal.Signals): The signal that was received.
        frame (FrameType): The current stack frame.
    """
    signal_name = signal.Signals(sig).name
    frame_info = (
        f"File {frame.f_code.co_filename}, line {frame.f_lineno}, in {frame.f_code.co_name}"
    )
    rospy.loginfo(f"Program interrupted by {signal_name} signal, shutting down.")
    rospy.loginfo(f"Signal received at: {frame_info}")
    rospy.signal_shutdown(f"{signal_name} received")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class Brain:
    """
    This class represents the brain of the turtle bot. It is responsible for
    moving the bot to all keypoints, rotating 360 degrees at each of them, and
    managing detected faces.
    """

    def __init__(self):
        self.map_manager = MapManager(show_plot=False)
        rospy.loginfo("Waiting for map manager to be ready...")
        while self.map_manager.is_ready() is False:
            rospy.sleep(0.1)
        rospy.loginfo("Map manager is ready")
        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base server.")
        self.move_base_client.wait_for_server()
        self.velocity_publisher = rospy.Publisher(
            "mobile_base/commands/velocity", Twist, queue_size=10
        )
        self.init_planner()
        self.markers_timer = rospy.Timer(rospy.Duration(1), lambda event: self.map_show_markers())
        self.detected_faces_subscriber = rospy.Subscriber(
            "/detected_faces", DetectedFaces, self.faces_callback
        )
        self.searched_space_timer = rospy.Timer(
            rospy.Duration(0.4), lambda event: self.map_manager.update_searched_space()
        )

        self.detected_rings_subscriber = rospy.Subscriber(
            "/detected_ring_coords", DetectedRings, self.ring_callback
        )
        self.detected_ground_rings_subscriber = rospy.Subscriber(
            "/detected_ground_ring_coords", DetectedRings, self.ground_ring_callback
        )

        # for cylinder handling
        rospy.Subscriber(
            "unique_cylinder_greet",
            CylinderGreetInstructions,
            self.detected_cylinder_callback,
        )
        self.cylinder_coords = []
        self.cylinder_colors = []
        self.cylinder_greet_poses = []
        # self.num_all_cylinders = 10
        self.num_all_cylinders = 4
        self.all_cylinders_found = False
        self.num_discovered_cylinders = 0

        self.is_ready = False
        self.detected_faces = []
        self.detected_faces_lock = threading.Lock()
        self.detected_rings: List[UniqueRingCoords] = []
        self.detected_rings_lock = threading.Lock()
        self.detected_ground_rings: List[UniqueRingCoords] = []
        self.detected_ground_rings_lock = threading.Lock()
        self.sound_player = SoundPlayer()
        self.additional_goals = []

        self.current_goal_pub = rospy.Publisher("brain_current_goal", Marker, queue_size=10)
        self.current_goal_marker_id = 0

        # for moving arm
        self.arm_mover = ArmMover()
        rospy.sleep(1)
        self.arm_mover.arm_movement_pub.publish(self.arm_mover.extend_ring)

        self.arm_pose = "extend_ring"

        # /green_ring_coords
        self.green_ring_coords_pub = rospy.Publisher(
            "green_ring_coords", UniqueRingCoords, queue_size=10
        )

        # /parking subscribe
        rospy.Subscriber("parking_spot", Pose, self.parking_callback)
        self.parking_lock = threading.Lock()
        self.parking_spot = None

        self.laser_manager = LaserManager()

        self.current_robot_pose = None
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.current_robot_pose_callback)

        # add service
        rospy.loginfo("Waiting for poster info service")
        rospy.wait_for_service("is_poster", timeout=20)
        rospy.loginfo("Poster info service is ready")
        self.poster_info_proxy = rospy.ServiceProxy("is_poster", IsPoster)

        self.mercenary_infos: List[MercenaryInfo] = []
        self.cylinder_list: List[Cylinder] = []

    def init_planner(self) -> None:
        """
        Initializes and configure the parameters for the DWA (Dynamic Window Approach)
        planner, which is a local motion planning algorithm used for the mobile robot's navigation.
        """
        rospy.set_param("/move_base/DWAPlannerROS/acc_lim_theta", 2)
        rospy.set_param("/move_base/DWAPlannerROS/min_vel_x", 7)
        rospy.set_param("/move_base/DWAPlannerROS/max_vel_x", 10)
        rospy.set_param("/move_base/DWAPlannerROS/acc_lim_x", 2)
        rospy.set_param("/move_base/DWAPlannerROS/max_vel_theta", 10)
        rospy.set_param("/move_base/DWAPlannerROS/min_vel_theta", 5)
        rospy.set_param("/move_base/DWAPlannerROS/min_in_place_vel_theta", 5)
        rospy.set_param("/move_base/DWAPlannerROS/escape_vel", -0.3)
        rospy.set_param("/move_base/DWAPlannerROS/xy_goal_tolerance", 0.15)
        rospy.set_param("/move_base/max_planning_retries", 3)
        rospy.set_param("/move_base/clearing_rotation_allowed", False)

    def map_show_markers(self) -> None:
        """
        Publishes markers of initial goals on the map.
        """
        goals = self.map_manager.get_goals()
        if len(self.additional_goals) > 0:
            goals.extend(self.additional_goals)

        self.map_manager.publish_markers_of_goals(goals)

    def faces_callback(self, msg: DetectedFaces) -> None:
        """
        Callback function for the faces subscriber. Stores
        detected faces in a thread-safe manner.

        Args:
            msg (DetectedFaces): The message containing the detected faces.
        """
        with self.detected_faces_lock:
            # if the array is smaller than before update only the existing faces

            # join the 2 arrays based on group_id
            # Create a dictionary of existing faces with group_id as key
            existing_faces_dict = {face.group_id: face for face in self.detected_faces}

            # Create a dictionary of new faces with group_id as key
            new_faces_dict = {face.group_id: face for face in msg.array}

            # Update existing_faces_dict with new_faces_dict. This will overwrite
            # existing entries with the same group_id and add new entries with new group_ids.
            existing_faces_dict.update(new_faces_dict)

            # Convert the updated dictionary back to a list
            self.detected_faces = list(existing_faces_dict.values())

    def ring_callback(self, msg: DetectedRings) -> None:
        """
        Callback function for the ring subscriber. Stores
        detected rings in a thread-safe manner.

        Args:
            msg (DetectedRings): The message containing the detected rings.
        """
        with self.detected_rings_lock:
            self.detected_rings = msg.array

    def ground_ring_callback(self, msg: DetectedRings):
        """
        Callback function for the ring subscriber. Stores
        detected rings in a thread-safe manner.

        Args:
            msg (DetectedRings): The message containing the detected rings.
        """
        with self.detected_ground_rings_lock:
            self.detected_ground_rings = msg.array

    def parking_callback(self, msg: Pose):
        """
        Callback function for the parking subscriber. Stores the parking spot

        Args:
            msg (Pose): The message containing the parking spot.
        """
        with self.parking_lock:
            self.parking_spot = msg

    def current_robot_pose_callback(self, data: Odometry) -> None:
        """
        Callback function for the current robot pose subscriber. Stores the current robot pose.

        Args:
            data (Odometry): Stores the current robot pose.
        """
        self.current_robot_pose = data.pose.pose

    def move_to_goal(
        self,
        x_position: float,
        y_position: float,
        rr_x: float,
        rr_y: float,
        rr_z: float,
        rr_w: float,
    ) -> int:
        """
        Moves the turtle bot to the goal with the given coordinates and orientation.

        Args:
            x (float): The x-coordinate of the goal position.
            y (float): The y-coordinate of the goal position.
            rr_x (float): The x-component of the quaternion representing the goal orientation.
            rr_y (float): The y-component of the quaternion representing the goal orientation.
            rr_z (float): The z-component of the quaternion representing the goal orientation.
            rr_w (float): The w-component of the quaternion representing the goal orientation.

        Returns:
            int: The state of the action after it has finished executing, one of the following:
                * GoalStatus.PENDING
                * GoalStatus.ACTIVE
                * GoalStatus.PREEMPTED
                * GoalStatus.SUCCEEDED
                * GoalStatus.ABORTED
                * GoalStatus.REJECTED
                * GoalStatus.PREEMPTING
                * GoalStatus.RECALLING
                * GoalStatus.RECALLED
                * GoalStatus.LOST
        """
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x_position
        goal.target_pose.pose.position.y = y_position
        goal.target_pose.pose.orientation.x = rr_x
        goal.target_pose.pose.orientation.y = rr_y
        goal.target_pose.pose.orientation.z = rr_z
        goal.target_pose.pose.orientation.w = rr_w

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "brain"
        marker.id = self.current_goal_marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = goal.target_pose.pose
        marker.scale.x = 0.5
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.lifetime = rospy.Duration(20)

        self.current_goal_pub.publish(marker)
        self.current_goal_marker_id += 1

        self.move_base_client.send_goal(goal)

        wait_result = self.move_base_client.wait_for_result()

        if not wait_result:
            rospy.logerr("Not able to set goal.")
            return -1

        res = self.move_base_client.get_state()
        return res

    def get_object_greet_pose(self, x_obj, y_obj, erosion: int = 0) -> Pose:
        """
        Returns Pose with proper greet location and orientation
        for ring / cylinder at x_obj, y_obj.
        """
        # compute position
        x_greet, y_greet = self.map_manager.get_nearest_accessible_point_with_erosion(
            x_obj, y_obj, erosion
        )

        # compute orientation from greet point to object point
        q_dest = self.map_manager.quaternion_from_points(x_greet, y_greet, x_obj, y_obj)

        # create pose for greet
        # pylint: disable=R0801
        pose = Pose()
        pose.position.x = x_greet
        pose.position.y = y_greet
        pose.position.z = 0
        pose.orientation.x = q_dest[0]
        pose.orientation.y = q_dest[1]
        pose.orientation.z = q_dest[2]
        pose.orientation.w = q_dest[3]
        return pose

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

        cylinder_greet_pose = self.get_object_greet_pose(x_cylinder, y_cylinder, erosion=7)

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

    def degrees_to_rad(self, deg):
        """
        Converts degrees to radians.

        Args:
            deg (float): The angle in degrees.

        Returns:
            float: The angle in radians.
        """
        return deg * math.pi / 180

    def rotate(self, angle_deg, angular_speed=0.7, clockwise=True):
        """
        Rotates the turtle bot by the specified angle at the given angular speed.

        Args:
            angle_deg (float): The angle in degrees the turtle bot should rotate.
            angular_speed (float, optional): The angular speed at which the turtle bot rotates.
                                            Defaults to 0.7.
            clockwise (bool, optional): The rotation direction. True for clockwise,
                                        False for counterclockwise. Defaults to True.
        """
        rospy.loginfo("Rotating.")

        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = angular_speed if not clockwise else -angular_speed

        # total angle in radians
        total_angle = self.degrees_to_rad(angle_deg)

        t_0 = rospy.Time.now().to_sec()
        angle_rotated = 0
        while angle_rotated < total_angle:
            self.velocity_publisher.publish(twist)
            t_1 = rospy.Time.now().to_sec()
            angle_rotated = abs(angular_speed * (t_1 - t_0))

        # stop the robot spinning
        twist.angular.z = 0
        self.velocity_publisher.publish(twist)
        rospy.loginfo(f"Done rotating. angle: {str(total_angle / math.pi * 180)}")

    def nearest_neighbor_path(self, vertices: list, start_vertex: Tuple[float, float]):
        """
        Given a list of vertices and a starting vertex, returns the nearest neighbor path
        as a list of vertices ordered by the path.

        Args:
            vertices (list of tuples): A list of vertices, each represented as a tuple (x, y)
            start_vertex (tuple): The starting vertex represented as a tuple (x, y)

        Returns:
            list of tuples: The nearest neighbor path as a list of vertices
        """
        unvisited_vertices = set(vertices)
        unvisited_vertices.remove(start_vertex)
        current_vertex = start_vertex
        path = [start_vertex]

        while unvisited_vertices:
            in_sight_vertices = [
                vertex
                for vertex in unvisited_vertices
                if self.map_manager.has_clear_path(current_vertex, vertex)
            ]
            if in_sight_vertices:
                in_sight_vertices.sort(
                    key=lambda vertex: math.sqrt(
                        (current_vertex[0] - vertex[0]) ** 2 + (current_vertex[1] - vertex[1]) ** 2
                    )
                )
                nearest_vertex = in_sight_vertices[0]
            else:
                nearest_vertex = None
                nearest_distance = sys.float_info.max

                for vertex in unvisited_vertices:
                    distance = math.sqrt(
                        (current_vertex[0] - vertex[0]) ** 2 + (current_vertex[1] - vertex[1]) ** 2
                    )
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_vertex = vertex

            path.append(nearest_vertex)
            unvisited_vertices.remove(nearest_vertex)
            current_vertex = nearest_vertex

        return path

    def orientation_between_points(
        self, first_goal: Tuple[float, float], second_goal: Tuple[float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Returns the orientation of the turtle bot as a quaternion between two goals.

        Args:
            first_goal (Tuple[float, float]): The first goal as a tuple (x, y)
            second_goal (Tuple[float, float]): The second goal as a tuple (x, y)

        Returns:
            Tuple[float, float, float, float]: quaternion representing the orientation
        """
        angle = math.atan2(second_goal[1] - first_goal[1], second_goal[0] - first_goal[0])
        angle = math.atan2(second_goal[1] - first_goal[1], second_goal[0] - first_goal[0])
        quaternion = quaternion_from_euler(0, 0, angle)
        return quaternion

    def visit_found_cylinders(self):
        """
        Visits the rings found until now
        """
        while self.have_cylinders_to_visit():
            if rospy.is_shutdown():
                return

            rospy.loginfo(f"Cylinder queue: {str(self.cylinder_colors)}")
            current_cylinder_color = self.cylinder_colors.pop(0)
            current_greet_pose = self.cylinder_greet_poses.pop(0)
            self.cylinder_coords.pop(0)
            self.greet_cylinder(current_cylinder_color, current_greet_pose)

    def greet_cylinder(self, color, current_greet_pose):
        """
        Greets a cylinder

        Args:
            object_pose (Pose): Pose of the cylinder
            color (str): Color of the cylinder
            current_greet_pose (Pose): Pose of the greeting position
            person_obj (_type_): _description_

        Returns:
            _type_: _description_
        """
        rospy.loginfo("Greeting cylinder")

        self.move_to_goal(
            current_greet_pose.position.x,
            current_greet_pose.position.y,
            current_greet_pose.orientation.x,
            current_greet_pose.orientation.y,
            current_greet_pose.orientation.z,
            current_greet_pose.orientation.w,
        )

        self.sound_player.say(color)
        rospy.sleep(1)

    def get_closest_ring(self) -> Tuple[UniqueRingCoords, float]:
        """
        get closest ring to current position
        """
        with self.detected_rings_lock:
            closest_ring = None
            closest_distance = float("inf")
            for ring in self.detected_rings:
                distance = self.map_manager.euclidean_distance(
                    self.current_robot_pose.position.x,
                    self.current_robot_pose.position.y,
                    ring.ring_pose.position.x,
                    ring.ring_pose.position.y,
                )
                if distance < closest_distance:
                    closest_distance = distance
                    closest_ring = ring

            return closest_ring, closest_distance

    def get_robot_distance_to_point(self, position_x: float, position_y: float) -> float:
        """
        Returns robot distance to point (x,y)
        """
        return self.map_manager.euclidean_distance(
            self.current_robot_pose.position.x,
            self.current_robot_pose.position.y,
            position_x,
            position_y,
        )

    def move_as_close_to_as_possible(
        self, position_x: float, position_y: float, speed: float = 0.1
    ) -> float:
        """
        Moves the robot as close to point (x,y) ass possible, moving in straight direction,
        without hitting any obstacles using twist messages.
        """

        twist_msg = Twist()

        # we will be moving forward
        twist_msg.linear.x = abs(speed)

        twist_msg.linear.y = 0
        twist_msg.linear.z = 0
        twist_msg.angular.x = 0
        twist_msg.angular.y = 0
        twist_msg.angular.z = 0

        t_0 = rospy.Time.now().to_sec()
        # we use traveled distance just in case if we will need it to return back
        traveled_distance = 0

        # dist_prev = self.get_robot_distance_to_point(y, x)
        derivative_threshold = 0.0013

        prev_ten_distances = []
        while True:
            dist_to_obj = self.get_robot_distance_to_point(position_x, position_y)

            rospy.loginfo(
                f"distance to goal:{dist_to_obj},"
                f" laser_wall:{self.laser_manager.distance_to_obstacle}"
            )

            if self.laser_manager.is_too_close_to_obstacle():
                rospy.loginfo("laser is too close to obstacle. Stopping")
                break

            if dist_to_obj <= 0.03:
                rospy.loginfo("distance is below 0.03 stopping")
                break
            prev_ten_distances.append(dist_to_obj)

            if len(prev_ten_distances) > 20:
                prev_ten_distances.pop(0)

            if len(prev_ten_distances) >= 20:
                derivative = np.gradient(prev_ten_distances)[-1]
                if derivative > 0:
                    rospy.loginfo(f"Derivative: {derivative}")

                if derivative > derivative_threshold:
                    rospy.loginfo("Derivative threshold exceeded. Stopping.")
                    break

            self.velocity_publisher.publish(twist_msg)
            t_1 = rospy.Time.now().to_sec()

            traveled_distance = abs(speed) * (t_1 - t_0)

        twist_msg.linear.x = 0
        self.velocity_publisher.publish(twist_msg)

        return traveled_distance

    def auto_adjust_arm_camera(self, _: rospy.Timer) -> None:
        """
        auto adjust arm camera
        """
        closest_ring, closest_distance = self.get_closest_ring()

        if closest_ring is not None:
            distance_to_wall = self.laser_manager.distance_to_obstacle

            if (
                self.arm_pose == "extend_ring"
                and closest_distance < 0.8
                and (0.2 < distance_to_wall < 0.7)
            ):
                rospy.loginfo("Adjusting arm camera")
                self.arm_pose = "adjust_ring_close"
                self.arm_mover.arm_movement_pub.publish(self.arm_mover.extend_ring_close)

            elif (
                self.arm_pose == "adjust_ring_close"
                and closest_distance >= 0.8
                and distance_to_wall > 0.7
            ):
                rospy.loginfo("Adjusting arm camera")
                self.arm_pose = "extend_ring"
                self.arm_mover.arm_movement_pub.publish(self.arm_mover.extend_ring)

    def think_rings_cylinders(self):
        """
        search and greet cylinders and search rings
        """

        detected_rings_count = 0
        target_ring_detections = 4
        detected_rings_group_ids = set()

        goals = self.map_manager.get_goals()
        arm_cam_timer = rospy.Timer(rospy.Duration(0.5), self.auto_adjust_arm_camera)

        while not rospy.is_shutdown():
            optimized_path = self.nearest_neighbor_path(goals, goals[0])

            for i, goal in enumerate(optimized_path):
                rospy.loginfo(
                    f"Moving to goal {i + 1}/{len(optimized_path)}. rings detected:"
                    f" {len(self.detected_rings)}"
                )

                # At each goal adjust orientation to the next goal
                quaternion = (0, 0, 0, 1)
                if i < len(optimized_path) - 1:
                    next_goal = optimized_path[i + 1]
                    quaternion = self.orientation_between_points(goal, next_goal)

                self.move_to_goal(goal[0], goal[1], *quaternion)

                self.rotate(360, angular_speed=0.7)

                self.visit_found_cylinders()

                with self.detected_rings_lock:
                    if len(self.detected_rings) > detected_rings_count:
                        rospy.loginfo(
                            f"I have detected {len(self.detected_rings) - detected_rings_count} new"
                            " rings during this iteration."
                        )

                        new_rings: List[UniqueRingCoords] = [
                            ring
                            for ring in self.detected_rings
                            if ring.group_id not in detected_rings_group_ids
                        ]

                        for new_ring in new_rings:
                            rospy.loginfo(
                                f"Saving ring with id: {new_ring.group_id}, color: {new_ring.color}"
                            )
                            detected_rings_group_ids.add(new_ring.group_id)

                        detected_rings_count = len(self.detected_rings)

                if detected_rings_count >= target_ring_detections and self.all_cylinders_found:
                    break

            if detected_rings_count < target_ring_detections or not self.all_cylinders_found:
                rospy.loginfo("Not all rings or cylinders have been detected. Will start EXPLORING")
                # get new goals now that we have explored the map
                self.additional_goals = self.map_manager.get_additional_goals()
                if len(self.additional_goals) < 1:
                    rospy.loginfo(
                        "No new goals found. Will stop i FAILED to find all rings or cylinders"
                    )
                    break
                rospy.loginfo(
                    f"Found {len(self.additional_goals )} new goals. Will continue exploring"
                )
                goals = self.additional_goals

            else:
                rospy.loginfo("ALL RINGS AND CYLINDERS have been detected. Will stop")

                with self.detected_rings_lock:
                    for ring in self.detected_rings:
                        rospy.loginfo(f"Ring: {ring.group_id}, {ring.color}")
                break

        green_ring: UniqueRingCoords = None
        with self.detected_rings_lock:
            for ring in self.detected_rings:
                if ring.color == "green":
                    green_ring = ring
                    break

        if green_ring is None:
            rospy.loginfo("No green ring found. ERROR")
            return

        self.green_ring_coords_pub.publish(green_ring)

        # compute approximate location to park
        aprox_park_location = self.get_object_greet_pose(
            green_ring.ring_pose.position.x, green_ring.ring_pose.position.y, erosion=6
        )
        self.move_to_goal(
            aprox_park_location.position.x,
            aprox_park_location.position.y,
            aprox_park_location.orientation.x,
            aprox_park_location.orientation.y,
            aprox_park_location.orientation.z,
            aprox_park_location.orientation.w,
        )

        arm_cam_timer.shutdown()

        self.arm_mover.arm_movement_pub.publish(self.arm_mover.extend_ring_close)

        rospy.loginfo(
            f"POSITION aprox_park_location: {aprox_park_location.position.x},"
            f" {aprox_park_location.position.y}"
        )
        rospy.loginfo(
            f"POSITION green_ring: {green_ring.ring_pose.position.x},"
            f" {green_ring.ring_pose.position.y}"
        )

        distance_to_green_ring = self.map_manager.euclidean_distance(
            green_ring.ring_pose.position.x,
            green_ring.ring_pose.position.y,
            aprox_park_location.position.x,
            aprox_park_location.position.y,
        )
        rospy.loginfo(
            f"Distance between green ring and approximate parking spot: {distance_to_green_ring}"
        )

        for i in range(10):
            # get robot distance to green ring
            robot_distance_to_wall = self.laser_manager.distance_to_obstacle

            rospy.loginfo(f"Distance between wall and robot: {robot_distance_to_wall}")

            # if distance is too big, move closer
            if robot_distance_to_wall > 0.75:
                rospy.loginfo("Distance to green ring is too big. Will move closer")
                twist = Twist()
                twist.linear.x = 0.2
                self.velocity_publisher.publish(twist)
                rospy.sleep(0.5)
                twist.linear.x = 0.0
                self.velocity_publisher.publish(twist)

            if robot_distance_to_wall < 0.7:
                rospy.loginfo("Robot is too close wall. Will move back")
                twist = Twist()
                twist.linear.x = -0.2
                self.velocity_publisher.publish(twist)
                rospy.sleep(0.5)
                twist.linear.x = 0.0
                self.velocity_publisher.publish(twist)

            # if distance is just right, stop
            if 0.7 < robot_distance_to_wall < 0.75:
                rospy.loginfo("Distance to wall is just right. Will stop")
                twist = Twist()
                twist.linear.x = 0.0
                self.velocity_publisher.publish(twist)
                break

        # rotate to find the green ring
        self.rotate(70, angular_speed=0.2)

        self.rotate(140, angular_speed=0.2, clockwise=False)

        # hide camera
        self.arm_mover.arm_movement_pub.publish(self.arm_mover.retract)

        # go to ground ring that is closest to the green ring
        with self.detected_ground_rings_lock:
            closest_ground_ring = None
            closest_distance = 100000
            for ground_ring in self.detected_ground_rings:
                distance = self.map_manager.euclidean_distance(
                    ground_ring.ring_pose.position.x,
                    ground_ring.ring_pose.position.y,
                    green_ring.ring_pose.position.x,
                    green_ring.ring_pose.position.y,
                )
                if distance < closest_distance:
                    closest_distance = distance
                    closest_ground_ring = ground_ring

        if closest_ground_ring is None:
            rospy.loginfo("No ground ring found. ERROR")
            return

        rospy.loginfo("Found best possible ground ring. Will move to it")

        aprox_park_location = self.get_object_greet_pose(
            closest_ground_ring.ring_pose.position.x,
            closest_ground_ring.ring_pose.position.y,
            erosion=6,
        )

        self.move_to_goal(
            aprox_park_location.position.x,
            aprox_park_location.position.y,
            aprox_park_location.orientation.x,
            aprox_park_location.orientation.y,
            aprox_park_location.orientation.z,
            aprox_park_location.orientation.w,
        )

        rospy.loginfo("Initiating parking procedure")

        dist_traveled = self.move_as_close_to_as_possible(
            closest_ground_ring.ring_pose.position.x,
            closest_ground_ring.ring_pose.position.y,
        )

        rospy.loginfo(f"The distance traveled to the ground ring is {dist_traveled}")
        rospy.loginfo("I have finished my task")

        self.sound_player.say("I have finished my task")

        for wave in [self.arm_mover.wave1, self.arm_mover.wave2]:
            self.arm_mover.arm_movement_pub.publish(wave)
            rospy.sleep(wave.points[-1].time_from_start.to_sec())

    def poster_manual_input(self, poster_text: str):
        """
        Function for manual input of the poster text.

        :param poster_text: The text on the poster. (used for auto completing the dialogue)
        """

        dialogue = PosterDialogue(poster_text)
        dialogue.start()

        if dialogue.is_valid():
            new_mercenary_info = None

            # add new mercenary info if it does not exist yet if it does exist, update it

            for mercenary_info in self.mercenary_infos:
                if mercenary_info.name == dialogue.name:
                    new_mercenary_info = mercenary_info
                    new_mercenary_info.ring_color = dialogue.ring_color
                    new_mercenary_info.wanted_price = dialogue.wanted_price

            if new_mercenary_info is None:
                new_mercenary_info = MercenaryInfo(
                    name=dialogue.name,
                    ring_color=dialogue.ring_color,
                    wanted_price=dialogue.wanted_price,
                )
                self.mercenary_infos.append(new_mercenary_info)

        rospy.loginfo(
            f"Finished poster dialogue. Person: {dialogue.name}, Ring(prison):"
            f" {dialogue.ring_color}, Price: {dialogue.wanted_price}"
        )

    def person_manual_dialogue(self):
        """
        Function for manual input of the person dialogue.
        """
        dialogue = PersonDialogue()
        dialogue.start()

        if dialogue.is_valid():
            new_mercenary_info = None
            for mercenary_info in self.mercenary_infos:
                if mercenary_info.name == dialogue.name:
                    new_mercenary_info = mercenary_info
                    new_mercenary_info.cylinder_color = dialogue.cylinder_color

            if new_mercenary_info is None:
                new_mercenary_info = MercenaryInfo(
                    name=dialogue.name,
                    cylinder_color=dialogue.cylinder_color,
                )
                self.mercenary_infos.append(new_mercenary_info)

        rospy.loginfo(
            f"Finished person dialogue. Person: {dialogue.name}, Cylinder:"
            f" {dialogue.cylinder_color}"
        )

    def think(self):
        """
        Main logic function for the turtle bot's brain. The turtle bot moves through the keypoints
        in the order determined by the nearest neighbor algorithm, performing a 360-degree rotation
        at each keypoint. If any new faces are detected during the rotation, the turtle bot moves to
        the face location and then continues to the next keypoint.
        """

        detected_faces_count = 0
        detected_faces_group_ids = set()
        detected_rings_group_ids = set()

        detected_rings_count = 0
        detected_rings_group_ids = set()

        goals = self.map_manager.get_goals()
        arm_cam_timer = rospy.Timer(rospy.Duration(0.5), self.auto_adjust_arm_camera)

        while not rospy.is_shutdown():
            optimized_path = self.nearest_neighbor_path(goals, goals[0])

            for i, goal in enumerate(optimized_path):
                rospy.loginfo(
                    f"Moving to goal {i + 1}/{len(optimized_path)}. Faces detected:"
                    f" {detected_faces_count}."
                )

                quaternion = (0, 0, 0, 1)
                if i < len(optimized_path) - 1:
                    # always face south
                    quaternion = (0, 0, -0.707, 0.707)

                self.move_to_goal(goal[0], goal[1], *quaternion)
                rospy.sleep(2.0)

                self.rotate(360, angular_speed=0.3)

                with self.detected_faces_lock:
                    rospy.loginfo(
                        f"I have detected {len(self.detected_faces) - detected_faces_count} new"
                        " faces during this iteration."
                    )

                    new_faces = [
                        face
                        for face in self.detected_faces
                        if face.group_id not in detected_faces_group_ids
                    ]

                    for new_face in new_faces:
                        self.move_to_goal(
                            new_face.x_coord,
                            new_face.y_coord,
                            new_face.rr_x,
                            new_face.rr_y,
                            new_face.rr_z,
                            new_face.rr_w,
                        )

                        rospy.sleep(2.0)

                        # Recognize poster here
                        response = self.poster_info_proxy(new_face.group_id)
                        for i in range(10):
                            if response.status == 1:
                                break
                            response = self.poster_info_proxy(new_face.group_id)

                            if i % 2 == 0:
                                twist = Twist()
                                twist.linear.x = -0.2
                                self.velocity_publisher.publish(twist)
                                rospy.sleep(0.8)
                                twist.linear.x = 0.0
                                self.velocity_publisher.publish(twist)

                            elif i % 3 == 0:
                                self.rotate(10, angular_speed=0.3)
                            elif i % 5 == 0:
                                self.rotate(10, angular_speed=0.3, clockwise=False)

                            else:
                                twist = Twist()
                                twist.linear.x = 0.2
                                self.velocity_publisher.publish(twist)
                                rospy.sleep(0.8)
                                twist.linear.x = 0.0
                                self.velocity_publisher.publish(twist)

                            rospy.sleep(0.5)

                        if response.status == 0:
                            rospy.logwarn("No poster info found this should not happen")
                            # if poster detection fails continue to next explore
                            # point and return to the face on the next iteration
                            rospy.sleep(2)
                            continue

                        if response.is_poster:
                            self.poster_manual_input(response.poster_text)

                        else:
                            self.person_manual_dialogue()

                        rospy.loginfo(f"Greeting face id: {new_face.group_id}")
                        self.sound_player.play_goodbye_sound()
                        rospy.sleep(2)
                        rospy.loginfo(f"Done greeting face id: {new_face.group_id}")

                        if new_face.group_id not in detected_faces_group_ids:
                            rospy.loginfo(f"Saving face with id: {new_face.group_id}")
                            detected_faces_group_ids.add(new_face.group_id)
                            detected_faces_count += 1

                with self.detected_rings_lock:
                    if len(self.detected_rings) > detected_rings_count:
                        rospy.loginfo(
                            f"I have detected {len(self.detected_rings) - detected_rings_count} new"
                            " rings during this iteration."
                        )

                        new_rings: List[UniqueRingCoords] = [
                            ring
                            for ring in self.detected_rings
                            if ring.group_id not in detected_rings_group_ids
                        ]

                        for new_ring in new_rings:
                            rospy.loginfo(
                                f"Saving ring with id: {new_ring.group_id}, color: {new_ring.color}"
                            )
                            detected_rings_group_ids.add(new_ring.group_id)

                        detected_rings_count = len(self.detected_rings)

            if not MercenaryInfo.are_complete(self.mercenary_infos):
                previous_goals = goals
                self.additional_goals = self.map_manager.get_additional_goals()
                rospy.loginfo("Mercenary Infos are not complete.")
                for mercenary_info in self.mercenary_infos:
                    rospy.loginfo(f"Mercenary Info: {mercenary_info}")

                # if the distance between the last goal
                # and the first goal is less than 0.1m remove the goal
                for goal in self.additional_goals:
                    if np.linalg.norm(np.array(goal) - np.array(previous_goals[0])) < 0.1:
                        rospy.loginfo(
                            f"Removing goal: {goal} because it is too close to the"
                            f" goal{previous_goals[0]}"
                        )
                        self.additional_goals.remove(goal)
                        rospy.loginfo("Removed goal because i wasn't able to reach it")

                if len(self.additional_goals) < 1:
                    rospy.loginfo("No new goals found. Will stop")
                    break
                rospy.loginfo(
                    f"Found {len(self.additional_goals )} new goals. Will continue exploring"
                )
                goals = self.additional_goals

            else:
                break

        if not MercenaryInfo.are_complete(self.mercenary_infos):
            rospy.logerr("I was not able to get all the mercenary info CRITICAL ERROR")

        arm_cam_timer.shutdown()
        with open("./debug/mercenary_data.txt", "w", encoding="utf-8") as file:
            file.write("\nMercenary data:\n")
            for mercenary_info in self.mercenary_infos:
                file.write(str(mercenary_info))

            file.write("\nRing data:\n")
            file.write(str(self.detected_rings))

            file.write("\nFace data:\n")
            file.write(str(self.detected_faces))

            file.write("\nCylinder data:\n")
            for cylinder in self.cylinder_list:
                file.write(
                    f"cylinder_id: {cylinder.cylinder_id}, color: {cylinder.cylinder_color},"
                    f" cylinder_greet_pose: {cylinder.cylinder_greet_pose} , cylinder_pose:"
                    f" {cylinder.cylinder_pose}\n"
                )

        # go through all mercenary infos and find one with highest wanted price
        highest_mercenary = max(self.mercenary_infos, key=lambda x: x.wanted_price)
        rospy.loginfo(
            f"Found mercenary with highest wanted price: {highest_mercenary}, thinking about which"
            " cylinder he is hiding at"
        )

        hiding_place_cylinder = None
        for cylinder in self.cylinder_list:
            if cylinder.cylinder_color == highest_mercenary.cylinder_color:
                hiding_place_cylinder = cylinder
                break

        if hiding_place_cylinder is None:
            rospy.logerr(
                "Could not find cylinder with same color as mercenary hiding place this should not"
                " happen"
            )
            return

        rospy.loginfo(
            f"Remembered about he {hiding_place_cylinder.cylinder_color} cylinder that someone"
        )

        prison_ring = None
        for ring in self.detected_rings:
            if ring.color == highest_mercenary.ring_color:
                prison_ring = ring

        if prison_ring is None:
            rospy.logerr(
                "Could not find ring with same color as prison ring this should not happen"
            )
            return

        # Go greet cylinder
        self.greet_cylinder(
            color=hiding_place_cylinder.cylinder_color,
            current_greet_pose=hiding_place_cylinder.cylinder_greet_pose,
        )

        aprox_park_location = self.get_object_greet_pose(
            prison_ring.ring_pose.position.x,
            prison_ring.ring_pose.position.y,
            erosion=6,
        )
        self.move_to_goal(
            aprox_park_location.position.x,
            aprox_park_location.position.y,
            aprox_park_location.orientation.x,
            aprox_park_location.orientation.y,
            aprox_park_location.orientation.z,
            aprox_park_location.orientation.w,
        )

        self.arm_mover.arm_movement_pub.publish(self.arm_mover.extend_ring_close)

        rospy.loginfo(
            f"POSITION aprox_park_location: {aprox_park_location.position.x},"
            f" {aprox_park_location.position.y}"
        )
        rospy.loginfo(
            f"POSITION  prison_ring: {prison_ring.ring_pose.position.x},"
            f" {prison_ring.ring_pose.position.y}"
        )

        distance_to_green_ring = self.map_manager.euclidean_distance(
            prison_ring.ring_pose.position.x,
            prison_ring.ring_pose.position.y,
            aprox_park_location.position.x,
            aprox_park_location.position.y,
        )
        rospy.loginfo(
            f"Distance between green ring and approximate parking spot: {distance_to_green_ring}"
        )

        for i in range(10):
            # get robot distance to green ring
            robot_distance_to_wall = self.laser_manager.distance_to_obstacle

            rospy.loginfo(f"Distance between wall and robot: {robot_distance_to_wall}")

            # if distance is too big, move closer
            if robot_distance_to_wall > 0.75:
                rospy.loginfo("Distance to green ring is too big. Will move closer")
                twist = Twist()
                twist.linear.x = 0.2
                self.velocity_publisher.publish(twist)
                rospy.sleep(0.5)
                twist.linear.x = 0.0
                self.velocity_publisher.publish(twist)

            if robot_distance_to_wall < 0.7:
                rospy.loginfo("Robot is too close wall. Will move back")
                twist = Twist()
                twist.linear.x = -0.2
                self.velocity_publisher.publish(twist)
                rospy.sleep(0.5)
                twist.linear.x = 0.0
                self.velocity_publisher.publish(twist)

            # if distance is just right, stop
            if 0.7 < robot_distance_to_wall < 0.75:
                rospy.loginfo("Distance to wall is just right. Will stop")
                twist = Twist()
                twist.linear.x = 0.0
                self.velocity_publisher.publish(twist)
                break

        # rotate to find the green ring
        self.rotate(70, angular_speed=0.2)

        self.rotate(140, angular_speed=0.2, clockwise=False)

        # hide camera
        self.arm_mover.arm_movement_pub.publish(self.arm_mover.retract)

        # go to ground ring that is closest to the green ring
        with self.detected_ground_rings_lock:
            closest_ground_ring = None
            closest_distance = 100000
            for ground_ring in self.detected_ground_rings:
                distance = self.map_manager.euclidean_distance(
                    ground_ring.ring_pose.position.x,
                    ground_ring.ring_pose.position.y,
                    prison_ring.ring_pose.position.x,
                    prison_ring.ring_pose.position.y,
                )
                if distance < closest_distance:
                    closest_distance = distance
                    closest_ground_ring = ground_ring

        if closest_ground_ring is None:
            rospy.loginfo("No ground ring found. ERROR")
            return

        rospy.loginfo("Found best possible ground ring. Will move to it")

        aprox_park_location = self.get_object_greet_pose(
            closest_ground_ring.ring_pose.position.x,
            closest_ground_ring.ring_pose.position.y,
            erosion=6,
        )

        self.move_to_goal(
            aprox_park_location.position.x,
            aprox_park_location.position.y,
            aprox_park_location.orientation.x,
            aprox_park_location.orientation.y,
            aprox_park_location.orientation.z,
            aprox_park_location.orientation.w,
        )

        rospy.loginfo("Initiating parking procedure")

        dist_traveled = self.move_as_close_to_as_possible(
            closest_ground_ring.ring_pose.position.x,
            closest_ground_ring.ring_pose.position.y,
        )

        rospy.loginfo(f"The distance traveled to the ground ring is {dist_traveled}")
        rospy.loginfo("I have finished my task")

        self.sound_player.say("I have finished my task")

        for wave in [self.arm_mover.wave1, self.arm_mover.wave2]:
            self.arm_mover.arm_movement_pub.publish(wave)
            rospy.sleep(wave.points[-1].time_from_start.to_sec())

        rospy.loginfo("I have finished my task")


if __name__ == "__main__":
    rospy.init_node("brain")
    brain = Brain()
    brain.is_ready = True
    brain.think()
    rospy.spin()
