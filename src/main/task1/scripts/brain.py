#!/usr/bin/python3

import math
import sys
import threading
from map_manager import MapManager
import rospy
from task1.msg import DetectedFaces
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist


class Brain:
    def __init__(self):
        self.map_manager = MapManager()
        rospy.loginfo("Waiting for map manager to be ready...")
        while self.map_manager.is_ready() == False:
            rospy.sleep(0.1)
        rospy.loginfo("Map manager is ready")
        self.mover = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base server.")
        self.mover.wait_for_server()
        self.cmd_vel_pub = rospy.Publisher(
            "mobile_base/commands/velocity", Twist, queue_size=10
        )
        self.init_planner()
        self.map_markers_timer = rospy.Timer(
            rospy.Duration(1), lambda event: brain.map_show_markers()
        )
        self.faces_subscriber = rospy.Subscriber(
            "/detected_faces", DetectedFaces, self.faces_callback
        )
        self.moving = False
        self.detected_faces = []
        self.detected_faces_lock = threading.Lock()

    def init_planner(self):
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

    # this function is used to show the markers on the map for the initial goals
    def map_show_markers(self):
        goals = self.map_manager.get_goals()
        self.map_manager.publish_markers_of_goals(goals)

    def faces_callback(self, msg):
        # rospy.loginfo("I have received a message from the face detector")
        # rospy.loginfo(msg)

        with self.detected_faces_lock:

            self.detected_faces = msg.array

    def move_to_goal_pose(self, pose, store_state=True):

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = pose.position.x
        goal.target_pose.pose.position.y = pose.position.y

        goal.target_pose.pose.orientation.x = pose.orientation.x
        goal.target_pose.pose.orientation.y = pose.orientation.y
        goal.target_pose.pose.orientation.z = pose.orientation.z
        goal.target_pose.pose.orientation.w = pose.orientation.w

        # rospy.loginfo("Sending next goal with xpose: %s" % str(pose))
        self.mover.send_goal(goal)

        wait_result = self.mover.wait_for_result()

        if not wait_result:
            rospy.logerr("Not able to set goal.")
        else:
            res = self.mover.get_state()
            # rospy.loginfo(str(res))
            return res

    def move_to_goal(self, x, y, rr_x, rr_y, rr_z, rr_w):

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        goal.target_pose.pose.orientation.x = rr_x
        goal.target_pose.pose.orientation.y = rr_y
        goal.target_pose.pose.orientation.z = rr_z
        goal.target_pose.pose.orientation.w = rr_w
        # rospy.loginfo("Sending next goal with x = %s and y = %s; rrx=%s, rry=%s, rrz=%s, rrw=%s " % (str(x), str(y), str(rr_x), str(rr_y), str(rr_z), str(rr_w)))
        self.mover.send_goal(goal)

        wait_result = self.mover.wait_for_result()

        # goal no longer in execution
        # if store_state:
        #    self.goal_forget()

        if not wait_result:
            rospy.logerr("Not able to set goal.")
        else:
            res = self.mover.get_state()
            # rospy.loginfo(str(res))
            return res

    def degrees_to_rad(self, deg):
        return deg * math.pi / 180

    def rotate(self, angle_deg, angular_speed=0.7):
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

        t0 = rospy.Time.now().to_sec()
        angle_rotated = 0
        while angle_rotated < total_angle:
            self.cmd_vel_pub.publish(twist)
            t1 = rospy.Time.now().to_sec()
            angle_rotated = angular_speed * (t1 - t0)

        # stop the robot spining
        twist.angular.z = 0
        self.cmd_vel_pub.publish(twist)
        rospy.loginfo("Done rotating. angle: %s" % str(total_angle / math.pi * 180))

    def nearest_neighbor_path(self, vertices, start_vertex):

        unvisited_vertices = set(vertices)
        unvisited_vertices.remove(start_vertex)
        current_vertex = start_vertex
        path = [start_vertex]

        while unvisited_vertices:
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

    def think(self):
        goals = self.map_manager.get_goals()
        nearest_neighbor_path = self.nearest_neighbor_path(goals, goals[0])

        df = 0

        for goal in nearest_neighbor_path:

            self.move_to_goal(goal[0], goal[1], 0, 0, 0, 1)
            self.rotate(360)
            with self.detected_faces_lock:
                if len(self.detected_faces) > df:
                    rospy.loginfo("I have detected a face")

                    new_faces = self.detected_faces[df:]
                    print(new_faces)
                    for new_face in new_faces:

                        self.move_to_goal(
                            new_face.x_coord,
                            new_face.y_coord,
                            new_face.rr_x,
                            new_face.rr_y,
                            new_face.rr_z,
                            new_face.rr_w,
                        )
                        # stop for 5 seconds
                        rospy.sleep(2)
                        rospy.loginfo("I have reached the face")

                    df = len(self.detected_faces)


if __name__ == "__main__":
    rospy.init_node("brain")
    brain = Brain()
    brain.think()
    rospy.spin()
