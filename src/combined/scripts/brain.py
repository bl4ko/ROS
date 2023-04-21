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
import actionlib
import rospy
from map_manager import MapManager
from sound import SoundPlayer
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist
from tf.transformations import quaternion_from_euler
from combined.msg import DetectedFaces


def signal_handler(_: signal.Signals) -> None:
    """
    Handles the SIGINT signal, which is sent when the user presses Ctrl+C.

    Args:
        sig (signal.Signals): The signal that was received.
    """
    rospy.loginfo("Program interrupted, shutting down.")
    rospy.signal_shutdown("SIGINT received")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class Brain:
    """
    This class represents the brain of the turtle bot. It is responsible for
    moving the bot to all keypoints, rotating 360 degrees at each of them, and
    managing detected faces.
    """

    def __init__(self):
        self.map_manager = MapManager(show_plot=True)
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
        self.markers_timer = rospy.Timer(rospy.Duration(1), lambda event: brain.map_show_markers())
        self.detected_faces_subscriber = rospy.Subscriber(
            "/detected_faces", DetectedFaces, self.faces_callback
        )
        self.searched_space_timer = rospy.Timer(
            rospy.Duration(0.4), lambda event: self.map_manager.update_searched_space()
        )

        self.is_moving = False
        self.detected_faces = []
        self.detected_faces_lock = threading.Lock()
        self.sound_player = SoundPlayer()
        self.aditional_goals = []

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

    def rotate(self, angle_deg, angular_speed=0.7):
        """
        Rotates the turtle bot by the specified angle at the given angular speed.

        Args:
            angle_deg (float): The angle in degrees the turtle bot should rotate.
            angular_speed (float, optional): The angular speed at which the turtle bot rotates.
                                             Defaults to 0.7.
        """
        rospy.loginfo("Rotating.")

        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = angular_speed

        # counter clockwise
        total_angle = self.degrees_to_rad(angle_deg)

        t_0 = rospy.Time.now().to_sec()
        angle_rotated = 0
        while angle_rotated < total_angle:
            self.velocity_publisher.publish(twist)
            t_1 = rospy.Time.now().to_sec()
            angle_rotated = angular_speed * (t_1 - t_0)

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
                    rospy.loginfo("No new goals found. Will stop i failed to find all faces")
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


if __name__ == "__main__":
    rospy.init_node("brain")
    brain = Brain()
    brain.think()
    rospy.spin()
