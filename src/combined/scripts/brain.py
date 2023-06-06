#!/usr/bin/python3


"""
Module representing brain of the turtle bot.
"""

import math
import signal
import sys
from typing import Tuple, List, Set
from types import FrameType
import numpy as np
import actionlib
import rospy
from map_manager import MapManager
from sound import SoundPlayer
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist, Pose, Quaternion, Point
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker

from laser_manager import LaserManager
from nav_msgs.msg import Odometry

from dialogue import PosterDialogue, PersonDialogue
from ring_manager import RingManager
from ground_ring_manager import GroundRingManager
from cylinder_manager import CylinderManager
from arm_manager import ArmManager
from face_manager import FaceManager
from mercenary import MercenaryInfo
from combined.msg import (
    UniqueRingCoords,
)
from combined.srv import IsPoster


def sigint_handler(sig: signal.Signals, frame: FrameType) -> None:
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


signal.signal(signal.SIGINT, sigint_handler)


# pylint: disable=too-many-instance-attributes
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

        # Object for managing ring detections
        self.ring_manager = RingManager()
        # Object for managing ground ring detections
        self.ground_ring_manager = GroundRingManager()
        # Object for managing face and posters detections
        self.face_manager = FaceManager()
        # Object for managing cylinders
        self.cylinder_manager = CylinderManager(map_manager=self.map_manager)
        # Object for managing the robot's arm
        self.arm_manager = ArmManager()

        # Object for playing sounds
        self.sound_player = SoundPlayer()

        self.current_goal_pub = rospy.Publisher("brain_current_goal", Marker, queue_size=10)
        self.current_goal_marker_id = 0

        self.laser_manager = LaserManager()

        # Subscriber for updating the current robot pose
        self.robot_pose_sub = rospy.Subscriber("/odom", Odometry, self.current_robot_pose_callback)
        self.current_robot_pose = None

        # add service
        rospy.loginfo("Waiting for poster info service")
        rospy.wait_for_service("is_poster", timeout=20)
        rospy.loginfo("Poster info service is ready")
        self.poster_info_proxy = rospy.ServiceProxy("is_poster", IsPoster)

        self.mercenary_infos: List[MercenaryInfo] = []

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

    def current_robot_pose_callback(self, data: Odometry) -> None:
        """
        Callback function for the current robot pose subscriber. Stores the current robot pose.

        Args:
            data (Odometry): Stores the current robot pose.
        """
        self.current_robot_pose = data.pose.pose

    def move_to_goal(self, goal_pose: Pose) -> int:
        """
        Moves the turtle bot to the goal with the given coordinates and orientation.

        Args:
            goal_pose (Pose): The goal pose.

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

        goal.target_pose.pose.position.x = goal_pose.position.x
        goal.target_pose.pose.position.y = goal_pose.position.y
        goal.target_pose.pose.orientation.x = goal_pose.orientation.x
        goal.target_pose.pose.orientation.y = goal_pose.orientation.y
        goal.target_pose.pose.orientation.z = goal_pose.orientation.z
        goal.target_pose.pose.orientation.w = goal_pose.orientation.w

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
        closest_ring, closest_distance = self.ring_manager.get_closest_ring(
            self.current_robot_pose.position.x, self.current_robot_pose.position.y
        )

        if closest_ring is not None:
            distance_to_wall = self.laser_manager.distance_to_obstacle

            if (
                self.arm_manager.arm_pose == "extend_above_down"
                and closest_distance < 0.8
                and (0.2 < distance_to_wall < 0.7)
            ):
                rospy.loginfo("Adjusting arm camera")
                self.arm_manager.move_arm("extend_front_down")

            elif (
                self.arm_manager.arm_pose == "extend_front_down"
                and closest_distance >= 0.8
                and distance_to_wall > 0.7
            ):
                rospy.loginfo("Adjusting arm camera")
                self.arm_manager.move_arm("extend_above_down")

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

    def process_new_cylinders(self, detected_cylinders_group_ids: Set[int]) -> None:
        """
        Function for processing new cylinders.

        Args:
            detected_cylinders_group_ids (Set[int]): Set containing
                            the group ids of the detected cylinders
        """

        for cylinder in self.cylinder_manager.detected_cylinders:
            if cylinder.cylinder_id not in detected_cylinders_group_ids:
                rospy.loginfo(
                    f"Saving cylinder with id: {cylinder.cylinder_id}, color: {cylinder.color}"
                )
                detected_cylinders_group_ids.add(cylinder.cylinder_id)

    def process_new_rings(self, detected_rings_group_ids: Set[int]) -> None:
        """
        Function for processing new rings.

        Args:
            detected_rings_group_ids (Set[int]): Set containing the group ids of the detected rings.
        """
        with self.ring_manager.detected_rings_lock:
            new_rings: List[UniqueRingCoords] = [
                ring
                for ring in self.ring_manager.detected_rings
                if ring.group_id not in detected_rings_group_ids
            ]

            for new_ring in new_rings:
                rospy.loginfo(f"Saving ring with id: {new_ring.group_id}, color: {new_ring.color}")
                detected_rings_group_ids.add(new_ring.group_id)

    def visit_new_faces(self, detected_faces_group_ids: Set[int]) -> None:
        """
        Args:
            detected_faces_group_ids (Set[int]): _description_
        """
        with self.face_manager.detected_faces_lock:
            for face in self.face_manager.detected_faces:
                if face.group_id not in detected_faces_group_ids:
                    face_goal = Pose(
                        position=Point(x=face.x_coord, y=face.y_coord, z=0.0),
                        orientation=Quaternion(face.rr_x, face.rr_y, face.rr_z, face.rr_w),
                    )
                    self.move_to_goal(face_goal)
                    rospy.sleep(2.0)
                    # Recognize poster here
                    response = self.poster_info_proxy(face.group_id)
                    for i in range(10):
                        if response.status == 1:
                            break
                        response = self.poster_info_proxy(face.group_id)

                        if i % 2 == 0:
                            twist = Twist()
                            twist.linear.x = -0.2
                            self.velocity_publisher.publish(twist)
                            rospy.sleep(0.8)
                            twist.linear.x = 0.0
                            self.velocity_publisher.publish(twist)

                        elif i % 3 == 0:
                            self.rotate(10, angular_speed=0.5)
                        elif i % 5 == 0:
                            self.rotate(10, angular_speed=0.5, clockwise=False)

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
                        rospy.sleep(2)
                        continue

                    rospy.loginfo(f"Greeting face id: {face.group_id}")
                    if response.is_poster:
                        self.poster_manual_input(response.poster_text)
                    else:
                        self.person_manual_dialogue()

                    self.sound_player.play_goodbye_sound()
                    rospy.sleep(2)
                    rospy.loginfo(f"Done greeting face id: {face.group_id}")

                    if face.group_id not in detected_faces_group_ids:
                        rospy.logdebug(f"Saving face with id: {face.group_id}")
                        detected_faces_group_ids.add(face.group_id)

    def get_additional_goals(
        self, previous_goals: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Returns a list of additional goals that the robot should visit.

        Args:
            previous_goals (List[Tuple[float, float]]): List of previous goals
            that the robot has visited.

        Returns:
            List[Tuple[float, float]]: List of additional goals that the robot should visit.
        """
        additional_goals = self.map_manager.get_additional_goals()

        for mercenary_info in self.mercenary_infos:
            rospy.loginfo(f"Mercenary Info: {mercenary_info}")

        # if the distance between the last goal
        # and the first goal is less than 0.1m remove the goal
        for goal in additional_goals.copy():
            if np.linalg.norm(np.array(goal) - np.array(previous_goals[0])) < 0.1:
                rospy.loginfo(
                    f"Removing goal: {goal} because it is too close to the goal{previous_goals[0]}"
                )
                additional_goals.remove(goal)
                rospy.loginfo("Removed goal because i wasn't able to reach it")

        if len(additional_goals) < 1:
            rospy.logerr("No new goals found. Will stop!")
            return []

        rospy.loginfo(f"Found {len(additional_goals )} new goals. Will continue exploring")

        return additional_goals

    def get_most_wanted_data(self):
        """
        Function for finding the most wanted mercenary hiding place and prison.

        Returns:
            Tuple: (hiding_place_cylinder, prison_ring)
        """
        # go through all mercenary infos and find one with highest wanted price
        highest_mercenary = max(self.mercenary_infos, key=lambda x: x.wanted_price)
        rospy.loginfo(
            f"Found mercenary with highest wanted price: {highest_mercenary}, thinking about which"
            " cylinder he is hiding at"
        )

        hiding_place_cylinder = None
        for cylinder in self.cylinder_manager.detected_cylinders:
            if cylinder.cylinder_color == highest_mercenary.cylinder_color:
                hiding_place_cylinder = cylinder
                break

        if hiding_place_cylinder is None:
            rospy.logerr("Could not find cylinder with same color as mercenary hiding place")
            return None, None

        prison_ring = None
        for ring in self.ring_manager.detected_rings:
            if ring.color == highest_mercenary.ring_color:
                prison_ring = ring

        if prison_ring is None:
            rospy.logerr("Could not find ring with the same color as a prison ring")
            return None, None

        return hiding_place_cylinder, prison_ring

    def park_in_ring(self, prison_ring: UniqueRingCoords):
        """
        Parks the robot in the prison ring.

        Args:
            prison_ring (UniqueRingCoords): The prison ring.
        """
        aprox_park_location = self.map_manager.get_object_greet_pose(
            prison_ring.ring_pose.position.x,
            prison_ring.ring_pose.position.y,
            erosion=6,
        )
        self.move_to_goal(aprox_park_location)

        self.arm_manager.move_arm("extend_front_down")

        rospy.loginfo(
            f"POSITION aprox_park_location: {aprox_park_location.position.x},"
            f" {aprox_park_location.position.y}"
            f"POSITION  prison_ring: {prison_ring.ring_pose.position.x},"
            f" {prison_ring.ring_pose.position.y}"
        )

        distance_to_prison_ring = self.map_manager.euclidean_distance(
            prison_ring.ring_pose.position.x,
            prison_ring.ring_pose.position.y,
            aprox_park_location.position.x,
            aprox_park_location.position.y,
        )
        rospy.loginfo(
            f"Distance between green ring and approximate parking spot: {distance_to_prison_ring}"
        )

        for _ in range(10):
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

        self.rotate(70, angular_speed=0.5)
        self.rotate(140, angular_speed=0.5, clockwise=False)

        # hide camera
        self.arm_manager.move_arm("retract_above_down")

        # go to ground ring that is closest to the green ring
        with self.ground_ring_manager.detected_ground_rings_lock:
            closest_ground_ring = None
            closest_distance = 100000
            for ground_ring in self.ground_ring_manager.detected_ground_rings:
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

        aprox_park_location = self.map_manager.get_object_greet_pose(
            closest_ground_ring.ring_pose.position.x,
            closest_ground_ring.ring_pose.position.y,
            erosion=6,
        )

        self.move_to_goal(
            aprox_park_location,
        )

        rospy.loginfo("Initiating parking procedure")

        dist_traveled = self.move_as_close_to_as_possible(
            closest_ground_ring.ring_pose.position.x,
            closest_ground_ring.ring_pose.position.y,
        )

        rospy.loginfo(f"The distance traveled to the ground ring is {dist_traveled}")
        rospy.loginfo("I have finished my task")

    def think(self):
        """
        Main logic function for the turtle bot's brain. The turtle bot moves through the keypoints
        in the order determined by the nearest neighbor algorithm, performing a 360-degree rotation
        at each keypoint. If any new faces are detected during the rotation, the turtle bot moves to
        the face location and then continues to the next keypoint.
        """

        detected_face_group_ids = set()
        detected_ring_group_ids = set()
        detected_cylinder_group_ids = set()

        goals = self.map_manager.get_goals()
        arm_cam_timer = rospy.Timer(rospy.Duration(0.5), self.auto_adjust_arm_camera)

        while not rospy.is_shutdown():
            optimized_path = self.nearest_neighbor_path(goals, goals[0])

            for i, goal in enumerate(optimized_path):
                self.map_manager.publish_markers_of_goals(goals=goals, duration=500)

                rospy.loginfo(
                    f"Moving to goal {i + 1}/{len(optimized_path)}. Faces detected:"
                    f" {len(detected_face_group_ids)}."
                )

                goal_pose = Pose()
                goal_pose.position.x, goal_pose.position.y = goal[0], goal[1]
                goal_pose.orientation = Quaternion(0, 0, -0.707, 0.707)  # always face south
                self.move_to_goal(goal_pose)

                rospy.sleep(2.0)

                self.rotate(360, angular_speed=0.6)

                # Process new faces
                new_face_count = self.face_manager.detection_count() - len(detected_face_group_ids)
                rospy.loginfo(f" {new_face_count} new faces detected.")
                if new_face_count > 0:
                    self.visit_new_faces(detected_face_group_ids)

                # Process new rings
                new_ring_count = self.ring_manager.detection_count() - len(detected_ring_group_ids)
                rospy.loginfo(f" {new_ring_count} new rings detected.")
                if new_ring_count > 0:
                    self.process_new_rings(detected_ring_group_ids)

                # Process new cylinders
                new_cylinder_count = self.cylinder_manager.detection_count() - len(
                    detected_cylinder_group_ids
                )
                rospy.loginfo(f" {new_cylinder_count} new cylinders detected.")
                if new_cylinder_count > 0:
                    self.process_new_cylinders(detected_cylinder_group_ids)

            # Check if enough data has been collected to complete the mission
            # If not then get additional goals
            if not MercenaryInfo.are_complete(self.mercenary_infos):
                rospy.loginfo("Mercenary Infos are not complete. Adding additional goals.")
                goals = self.get_additional_goals(previous_goals=goals)
            else:
                break

        arm_cam_timer.shutdown()

        hiding_place_cylinder, prison_ring = self.get_most_wanted_data()

        self.move_to_goal(hiding_place_cylinder.cylinder_greet_pose)
        self.sound_player.say(hiding_place_cylinder.cylinder_color)

        # Park inside the prison ring
        self.park_in_ring(prison_ring)

        self.arm_manager.move_arm("wave1")
        self.arm_manager.move_arm("wave2")

        self.sound_player.say("I have finished my task")


if __name__ == "__main__":
    rospy.init_node("brain")
    brain = Brain()
    brain.think()
    rospy.spin()
