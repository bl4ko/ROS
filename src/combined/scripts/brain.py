#!/usr/bin/python3

"""
Module representing brain of the turlte bot. Currently it moves the turtle bot 
to all of the keypoints and do a 360 degree rotation at each of them.
"""

import math
import signal
import sys
import threading
from typing import Tuple
from types import FrameType
import actionlib
import rospy
from map_manager import MapManager
from sound import SoundPlayer
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist, Pose
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker
from combined.msg import DetectedFaces
from combined.msg import DetectedRings
from combined.msg import CylinderGreetInstructions
from combined.msg import UniqueRingCoords
from move_arm import Arm_Mover
from typing import List
from laser_manager import LaserManager
from nav_msgs.msg import Odometry
import numpy as np


def signal_handler(sig: signal.Signals, frame: FrameType) -> None:
    """
    Handles the SIGINT signal, which is sent when the user presses Ctrl+C.

    Args:
        sig (signal.Signals): The signal that was received.
        frame (FrameType): The current stack frame.
    """
    signal_name = signal.Signals(sig).name
    frame_info = f"File {frame.f_code.co_filename}, line {frame.f_lineno}, in {frame.f_code.co_name}"
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
        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction
        )
        rospy.loginfo("Waiting for move_base server.")
        self.move_base_client.wait_for_server()
        self.velocity_publisher = rospy.Publisher(
            "mobile_base/commands/velocity", Twist, queue_size=10
        )
        self.init_planner()
        self.markers_timer = rospy.Timer(
            rospy.Duration(1), lambda event: self.map_show_markers()
        )
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
        self.aditional_goals = []

        self.current_goal_pub = rospy.Publisher(
            "brain_current_goal", Marker, queue_size=10
        )
        self.current_goal_marker_id = 0

        # for moving arm
        self.arm_mover = Arm_Mover()
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
        self.odom_sub = rospy.Subscriber(
            "/odom", Odometry, self.current_robot_pose_callback
        )

    def init_planner(self):
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

    def map_show_markers(self):
        """
        Publishes markers of initail goals on the map.
        """
        goals = self.map_manager.get_goals()
        if len(self.aditional_goals) > 0:
            goals.extend(self.aditional_goals)

        self.map_manager.publish_markers_of_goals(goals)

    def faces_callback(self, msg: DetectedFaces):
        """
        Callback function for the faces subscriber. Stores
        detected faces in a thread-safe manner.

        Args:
            msg (DetectedFaces): The message containing the detected faces.
        """
        with self.detected_faces_lock:
            self.detected_faces = msg.array

    def ring_callback(self, msg: DetectedRings):
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
        with self.parking_lock:
            self.parking_spot = msg

    def current_robot_pose_callback(self, data):
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

        # log
        # rospy.loginfo("Sending goal to move_base: " + str(goal))

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
        # set the lifetime of the marker
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
        pose = Pose()
        pose.position.x = x_greet
        pose.position.y = y_greet
        pose.position.z = 0

        # so that there are no warnings
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

        # rospy.loginfo("Received cylinder at position: %s with color %s" % (str(cylinder_pose), cylinder_color))
        rospy.loginfo("Received cylinder with color %s" % (cylinder_color))

        # compute greet location and orientation
        x_cylinder = cylinder_pose.position.x
        y_cylinder = cylinder_pose.position.y

        cylinder_greet_pose = self.get_object_greet_pose(
            x_cylinder, y_cylinder, erosion=5
        )

        self.cylinder_coords.append(cylinder_pose)
        self.cylinder_colors.append(cylinder_color)
        self.cylinder_greet_poses.append(cylinder_greet_pose)
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
            clockwise (bool, optional): The rotation direction. True for clockwise, False for counterclockwise.
                                        Defaults to True.
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

        # stop the robot spining
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
            # print(in_sight_vertices)
            if in_sight_vertices:
                in_sight_vertices.sort(
                    key=lambda vertex: math.sqrt(
                        (current_vertex[0] - vertex[0]) ** 2
                        + (current_vertex[1] - vertex[1]) ** 2
                    )
                )
                nearest_vertex = in_sight_vertices[0]
            else:
                nearest_vertex = None
                nearest_distance = sys.float_info.max

                for vertex in unvisited_vertices:
                    distance = math.sqrt(
                        (current_vertex[0] - vertex[0]) ** 2
                        + (current_vertex[1] - vertex[1]) ** 2
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
        angle = math.atan2(
            second_goal[1] - first_goal[1], second_goal[0] - first_goal[0]
        )
        angle = math.atan2(
            second_goal[1] - first_goal[1], second_goal[0] - first_goal[0]
        )
        quaternion = quaternion_from_euler(0, 0, angle)
        return quaternion

    def think(self):
        """
        Main logic function for the turtle bot's brain. The turtle bot moves through the keypoints
        in the order determined by the nearest neighbor algorithm, performing a 360-degree rotation
        at each keypoint. If any new faces are detected during the rotation, the turtle bot moves to
        the face location and then continues to the next keypoint.
        """

        detected_faces_count = 0
        target_face_detections = 3
        detected_faces_group_ids = set()

        goals = self.map_manager.get_goals()

        while not rospy.is_shutdown():
            optimized_path = self.nearest_neighbor_path(goals, goals[0])

            for i, goal in enumerate(optimized_path):
                rospy.loginfo(
                    f"Moving to goal {i + 1}/{len(optimized_path)}. Faces detected:"
                    f" {len(self.detected_faces)}"
                )

                # At each goal adjust orientation to the next goal
                quaternion = (0, 0, 0, 1)
                if i < len(optimized_path) - 1:
                    next_goal = optimized_path[i + 1]
                    quaternion = self.orientation_between_points(goal, next_goal)

                self.move_to_goal(goal[0], goal[1], *quaternion)

                self.rotate(360, angular_speed=0.7)

                with self.detected_faces_lock:
                    if len(self.detected_faces) > detected_faces_count:
                        rospy.loginfo(
                            f"I have detected {len(self.detected_faces) - detected_faces_count} new"
                            " faces during this iteration."
                        )

                        # get new faces based on group id!
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

                            rospy.loginfo(f"Greeting face id: {new_face.group_id}")
                            self.sound_player.play_greeting_sound()
                            rospy.sleep(2)
                            rospy.loginfo(f"Done greeting face id: {new_face.group_id}")
                            detected_faces_group_ids.add(new_face.group_id)

                        detected_faces_count = len(self.detected_faces)

                if detected_faces_count >= target_face_detections:
                    break

            if detected_faces_count < target_face_detections:
                rospy.loginfo("Not all faces have been detected. Will start EXPLORING")
                # get new goals now that we have explored the map
                self.aditional_goals = self.map_manager.get_get_aditional_goals()
                if len(self.aditional_goals) < 1:
                    rospy.loginfo(
                        "No new goals found. Will stop i failed to find all faces"
                    )
                    break
                else:
                    rospy.loginfo(
                        f"Found {len(self.aditional_goals )} new goals. Will continue exploring"
                    )
                    goals = self.aditional_goals

            else:
                rospy.loginfo("All faces have been detected. Will stop")
                break

        rospy.loginfo("I have finished my task")

    def visit_found_cylinders(self):
        """
        Visits the rings found until now
        """
        while self.have_cylinders_to_visit():
            if rospy.is_shutdown():
                return

            rospy.loginfo("Cylinder queue: %s" % (str(self.cylinder_colors)))
            current_cylinder_pose = self.cylinder_coords.pop(0)
            current_cylinder_color = self.cylinder_colors.pop(0)
            current_greet_pose = self.cylinder_greet_poses.pop(0)
            self.greet_cylinder(
                current_cylinder_pose, current_cylinder_color, current_greet_pose
            )

    # def move_as_close_to_as_possible(self, x_coordinate, y_coordinate, speed=0.3):
    #     """
    #     Moves the robot as close to point (x,y) ass possible, moving in straight direction,
    #     without hitting any obstacles using twist messages.

    #     Args:
    #         x (float): x coor
    #         y (float): y coor
    #         speed (float, optional): _description_. Defaults to 0.3.

    #     Returns:
    #         float: travveld distance
    #     """

    #     twist_msg = Twist()

    #     # we will be moving forward
    #     twist_msg.linear.x = abs(speed)

    #     twist_msg.linear.y = 0
    #     twist_msg.linear.z = 0
    #     twist_msg.angular.x = 0
    #     twist_msg.angular.y = 0
    #     twist_msg.angular.z = 0

    #     t0 = rospy.Time.now().to_sec()
    #     # we use travelled distance just in case if we will need it to return back
    #     travelled_distance = 0

    #     dist_prev = self.get_robot_distance_to_point(x_coordinate, y_coordinate)
    #     while not self.laser_manager.is_too_close_to_obstacle():
    #         dist_to_obj = self.get_robot_distance_to_point(x_coordinate, y_coordinate)
    #         if dist_to_obj > dist_prev:
    #             break
    #         dist_prev = dist_to_obj

    #         self.cmd_vel_pub.publish(twist_msg)
    #         t1 = rospy.Time.now().to_sec()
    #         travelled_distance = abs(speed) * (t1 - t0)

    #     # we now stop the robot immediately
    #     twist_msg.linear.x = 0
    #     self.cmd_vel_pub.publish(twist_msg)

    #     # we return the distance travelled
    #     return travelled_distance

    def greet_cylinder(self, object_pose, color, current_greet_pose):
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

        # # if neccessary move closer to object with twist messages
        # dist = self.move_as_close_to_as_possible(
        #     object_pose.position.x, object_pose.position.y
        # )

        # extend robot arm
        # self.arm_mover.arm_movement_pub.publish(self.arm_mover.extend)

        # says cylinder color
        self.sound_player.say(color)

        # get data from qr code

        # wait for arm to be extended
        rospy.sleep(1)

        # retract arm again

    def think_cylinder(self):
        """
        search and greet cylinders
        """

        goals = self.map_manager.get_goals()
        ring_count = 4
        rings_found = 0

        while not rospy.is_shutdown():
            optimized_path = self.nearest_neighbor_path(goals, goals[0])

            for i, goal in enumerate(optimized_path):
                rospy.loginfo(f"Moving to goal {i + 1}/{len(optimized_path)}.")

                # At each goal adjust orientation to the next goal
                quaternion = (0, 0, 0, 1)
                if i < len(optimized_path) - 1:
                    next_goal = optimized_path[i + 1]
                    quaternion = self.orientation_between_points(goal, next_goal)

                self.move_to_goal(goal[0], goal[1], *quaternion)
                self.rotate(360, angular_speed=0.7)

                # search for cylinders
                self.visit_found_cylinders()

            if not self.all_cylinders_found:
                rospy.loginfo(
                    "Not all cylinders have been detected. Will start EXPLORING"
                )
                # get new goals now that we have explored the map
                self.aditional_goals = self.map_manager.get_get_aditional_goals()
                if len(self.aditional_goals) < 1:
                    rospy.loginfo(
                        "No new goals found. Will stop i failed to find all cylinders"
                    )
                    break
                else:
                    rospy.loginfo(
                        f"Found {len(self.aditional_goals )} new goals. Will continue exploring"
                    )
                    goals = self.aditional_goals

            else:
                rospy.loginfo("All cylinders have been detected. Will stop")
                break

        rospy.loginfo("I have finished my task")

    def think_rings(self):
        """
        search and greet rings
        """

        detected_rings_count = 0
        target_ring_detections = 5
        detected_rings_group_ids = set()

        goals = self.map_manager.get_goals()

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

                with self.detected_rings_lock:
                    if len(self.detected_rings) > detected_rings_count:
                        rospy.loginfo(
                            f"I have detected {len(self.detected_rings) - detected_rings_count} new"
                            " rings during this iteration."
                        )

                        # get new rings based on group id!
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

                if detected_rings_count >= target_ring_detections:
                    break

            if detected_rings_count < target_ring_detections:
                rospy.loginfo("Not all rings have been detected. Will start EXPLORING")
                # get new goals now that we have explored the map
                self.aditional_goals = self.map_manager.get_get_aditional_goals()
                if len(self.aditional_goals) < 1:
                    rospy.loginfo(
                        "No new goals found. Will stop i failed to find all rings"
                    )
                    break
                else:
                    rospy.loginfo(
                        f"Found {len(self.aditional_goals )} new goals. Will continue exploring"
                    )
                    goals = self.aditional_goals

            else:
                rospy.loginfo("All rings have been detected. Will stop")
                break

        rospy.loginfo("I have finished my task")

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

    def get_robot_distance_to_point(self, x, y):
        """
        Returns robot distance to point (x,y)
        """

        # rospy.loginfo(
        #     f"Getting distance from robot to point ({x},{y}) in map frame, current robot pose: "
        #     f"{self.current_robot_pose}"
        # )

        return self.map_manager.euclidean_distance(
            self.current_robot_pose.position.x, self.current_robot_pose.position.y, x, y
        )

    def move_as_close_to_as_possible(self, x, y, speed=0.1):
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

        t0 = rospy.Time.now().to_sec()
        # we use travelled distance just in case if we will need it to return back
        travelled_distance = 0

        dist_prev = self.get_robot_distance_to_point(y, x)
        derivative_threshold = 0.0013

        prev_ten_distances = []
        while True:
            dist_to_obj = self.get_robot_distance_to_point(x, y)

            rospy.loginfo(
                f"distance to goal:{dist_to_obj}, laser_wall:{self.laser_manager.distance_to_obstacle}"
            )

            # if dist_to_obj > dist_prev:
            #     break

            # dist_prev = dist_to_obj

            if self.laser_manager.is_too_close_to_obstacle():
                rospy.loginfo("laser is too close to obstacle. Stopping")
                break

            if dist_to_obj <= 0.03:
                rospy.loginfo("distance is below 0.03 stopping")
                break
            prev_ten_distances.append(dist_to_obj)

            if len(prev_ten_distances) > 20:
                prev_ten_distances.pop(0)

            # Calculate the derivative of the distance over time
            if len(prev_ten_distances) >= 20:
                derivative = np.gradient(prev_ten_distances)[-1]
                if derivative > 0:
                    rospy.loginfo(f"Derivative: {derivative}")

                # Check if the derivative is above the threshold
                if derivative > derivative_threshold:
                    rospy.loginfo("Derivative threshold exceeded. Stopping.")
                    break

            self.velocity_publisher.publish(twist_msg)
            t1 = rospy.Time.now().to_sec()

            travelled_distance = abs(speed) * (t1 - t0)

        # we now stop the robot immediately
        twist_msg.linear.x = 0
        self.velocity_publisher.publish(twist_msg)

        # we return the distance travelled
        return travelled_distance

    def auto_adjust_arm_camera(self, event):
        """
        auto adjust arm camera
        """

        # get closest ring

        # rospy.loginfo(f"Finding closest rings")
        closest_ring, closest_distance = self.get_closest_ring()

        if closest_ring is not None:
            # rospy.loginfo(
            #     # f"Closest ring is {closest_ring.group_id} with color {closest_ring.color} found at"
            #     #f" {closest_distance} meters"
            # )

            # if closest ring is within 1 meter adjust camera

            distance_to_wall = self.laser_manager.distance_to_obstacle

            if (
                self.arm_pose == "extend_ring"
                and closest_distance < 0.6
                and distance_to_wall < 0.7 and distance_to_wall > 0.2
            ):
                rospy.loginfo("Adjusting arm camera")
                self.arm_pose = "adjust_ring_close"
                self.arm_mover.arm_movement_pub.publish(
                    self.arm_mover.extend_ring_close
                )

            elif (
                self.arm_pose == "adjust_ring_close"
                and closest_distance >= 0.6
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

                        # get new rings based on group id!
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

                if (
                    detected_rings_count >= target_ring_detections
                    and self.all_cylinders_found
                ):
                    break

            if (
                detected_rings_count < target_ring_detections
                or not self.all_cylinders_found
            ):
                rospy.loginfo(
                    "Not all rings or cylinders have been detected. Will start EXPLORING"
                )
                # get new goals now that we have explored the map
                self.aditional_goals = self.map_manager.get_get_aditional_goals()
                if len(self.aditional_goals) < 1:
                    rospy.loginfo(
                        "No new goals found. Will stop i FAILED to find all rings or cylinders"
                    )
                    break
                else:
                    rospy.loginfo(
                        f"Found {len(self.aditional_goals )} new goals. Will continue exploring"
                    )
                    goals = self.aditional_goals

            else:
                rospy.loginfo("ALL RINGS AND CYLINDERS have been detected. Will stop")

                # print the rings
                with self.detected_rings_lock:
                    for ring in self.detected_rings:
                        rospy.loginfo(f"Ring: {ring.group_id}, {ring.color}")

                #

                break

        # perform parking
        # find green ring

        green_ring: UniqueRingCoords = None
        with self.detected_rings_lock:
            for ring in self.detected_rings:
                if ring.color == "green":
                    green_ring = ring
                    break

        if green_ring is None:
            rospy.loginfo("No green ring found. ERROR")
            return

        # start searching for ground rings
        self.green_ring_coords_pub.publish(green_ring)

        # self.arm_mover.arm_movement_pub.publish(self.arm_mover.extend_ring_close)

        # compute approximate location to park
        aproxx_park_location = self.get_object_greet_pose(
            green_ring.ring_pose.position.x, green_ring.ring_pose.position.y, erosion=6
        )
        self.move_to_goal(
            aproxx_park_location.position.x,
            aproxx_park_location.position.y,
            aproxx_park_location.orientation.x,
            aproxx_park_location.orientation.y,
            aproxx_park_location.orientation.z,
            aproxx_park_location.orientation.w,
        )

        arm_cam_timer.shutdown()

        # close camera
        self.arm_mover.arm_movement_pub.publish(self.arm_mover.extend_ring_close)

        rospy.loginfo(
            f"POSITION aproxx_park_location: {aproxx_park_location.position.x}, {aproxx_park_location.position.y}"
        )
        rospy.loginfo(
            f"POSITION  green_ring: {green_ring.ring_pose.position.x}, {green_ring.ring_pose.position.y}"
        )

        # log distance to green ring
        disttance_to_green_ring = self.map_manager.euclidean_distance(
            green_ring.ring_pose.position.x,
            green_ring.ring_pose.position.y,
            aproxx_park_location.position.x,
            aproxx_park_location.position.y,
        )
        rospy.loginfo(
            f"Distance between green ring and approximate parking spot: {disttance_to_green_ring}"
        )

        # for i in range(1):
        #     # get robot distance to green ring
        #     robot_distance_to_green_ring = self.map_manager.euclidean_distance(
        #         green_ring.ring_pose.position.x,
        #         green_ring.ring_pose.position.y,
        #         self.current_robot_pose.position.x,
        #         self.current_robot_pose.position.y,
        #     )

        #     rospy.loginfo(
        #         f"POSITION green_ring: {green_ring.ring_pose.position.x}, {green_ring.ring_pose.position.y}"
        #     )
        #     rospy.loginfo(
        #         f"POSITION  robot: {self.current_robot_pose.position.x}, {self.current_robot_pose.position.y}"
        #     )

        #     rospy.loginfo(
        #         f"Distance between green ring and robot: {robot_distance_to_green_ring}"
        #     )

        #     # if distance is too big, move closer
        #     if robot_distance_to_green_ring > 0.55:
        #         rospy.loginfo("Distance to green ring is too big. Will move closer")
        #         twist = Twist()
        #         twist.linear.x = 0.2
        #         self.velocity_publisher.publish(twist)
        #         rospy.sleep(0.5)
        #         twist.linear.x = 0.0
        #         self.velocity_publisher.publish(twist)

        #     if robot_distance_to_green_ring < 0.45:
        #         rospy.loginfo("Robot is too close to the green ring. Will move back")
        #         twist = Twist()
        #         twist.linear.x = -0.2
        #         self.velocity_publisher.publish(twist)
        #         rospy.sleep(0.5)
        #         twist.linear.x = 0.0
        #         self.velocity_publisher.publish(twist)

        #     # if distance is just right, stop
        #     if (
        #         robot_distance_to_green_ring < 0.55
        #         and robot_distance_to_green_ring > 0.45
        #     ):
        #         rospy.loginfo("Distance to green ring is just right. Will stop")
        #         twist = Twist()
        #         twist.linear.x = 0.0
        #         self.velocity_publisher.publish(twist)
        #         break

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
            if robot_distance_to_wall < 0.75 and robot_distance_to_wall > 0.7:
                rospy.loginfo("Distance to wall is just right. Will stop")
                twist = Twist()
                twist.linear.x = 0.0
                self.velocity_publisher.publish(twist)
                break

        max_distance = 0.3

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

        aproxx_park_location = self.get_object_greet_pose(
            closest_ground_ring.ring_pose.position.x,
            closest_ground_ring.ring_pose.position.y,
            erosion=6,
        )

        self.move_to_goal(
            aproxx_park_location.position.x,
            aproxx_park_location.position.y,
            aproxx_park_location.orientation.x,
            aproxx_park_location.orientation.y,
            aproxx_park_location.orientation.z,
            aproxx_park_location.orientation.w,
        )

        rospy.loginfo("Initiating parking procedure")

        dist_traveled = self.move_as_close_to_as_possible(
            closest_ground_ring.ring_pose.position.x,
            closest_ground_ring.ring_pose.position.y,
        )

        rospy.loginfo(f"The distance traveled to the ground ring is {dist_traveled}")

        # wait for message from parking subscriber in while loop
        # while not rospy.is_shutdown():
        #     rospy.sleep(0.1)
        #     with self.parking_lock:
        #         if self.parking_spot is not None:
        #             rospy.loginfo("I have found a parking spot")

        #             # move to parking spot
        #             self.move_to_goal(
        #                 self.parking_spot.position.x,
        #                 self.parking_spot.position.y,
        #                 self.parking_spot.orientation.x,
        #                 self.parking_spot.orientation.y,
        #                 self.parking_spot.orientation.z,
        #                 self.parking_spot.orientation.w,
        #             )
        #             rospy.loginfo("I have parked")
        #             break

        rospy.loginfo("I have finished my task")


if __name__ == "__main__":
    rospy.init_node("brain")
    brain = Brain()
    brain.is_ready = True
    brain.think_rings_cylinders()
    rospy.spin()
