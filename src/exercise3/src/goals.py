#!/usr/bin/python3
import rospy
from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion


class MoveBaseClient:
    def __init__(self, pose_seq):
        self.pose_seq = pose_seq
        self.goal_cnt = 0
        self.client = SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to move_base action server")

    def done_cb(self, status, result):
        if status == 2:
            rospy.loginfo(
                "Goal pose "
                + str(self.goal_cnt + 1)
                + " received a cancel request after it started executing, completed execution!"
            )

        if status == 3:
            rospy.loginfo("Goal pose " + str(self.goal_cnt + 1) + " reached")
            self.goal_cnt += 1
            if self.goal_cnt < len(self.pose_seq):
                self.send_next_goal()

    def active_cb(self):
        rospy.loginfo(
            "Goal pose "
            + str(self.goal_cnt + 1)
            + " is now being processed by the Action Server"
        )

    def feedback_cb(self, feedback):
        pass

    def send_next_goal(self):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = self.pose_seq[self.goal_cnt]
        rospy.loginfo(
            "Sending goal pose " + str(self.goal_cnt + 1) + " to Action Server"
        )
        rospy.loginfo(str(self.pose_seq[self.goal_cnt]))
        self.client.send_goal(
            goal,
            done_cb=self.done_cb,
            active_cb=self.active_cb,
            feedback_cb=self.feedback_cb,
        )

    def movebase_client(self):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = self.pose_seq[self.goal_cnt]

        rospy.loginfo(
            "Sending goal pose " + str(self.goal_cnt + 1) + " to Action Server"
        )
        rospy.loginfo(str(self.pose_seq[self.goal_cnt]))

        # Send the goal and wait for the result
        status = self.client.send_goal_and_wait(
            goal, rospy.Duration(120), rospy.Duration(20)
        )

        # Reference for terminal status values: http://docs.ros.org/diamondback/api/actionlib_msgs/html/msg/GoalStatus.html
        if status == 2:
            rospy.loginfo(
                "Goal pose "
                + str(self.goal_cnt + 1)
                + " received a cancel request after it started executing, completed execution!"
            )

        if status == 3:
            rospy.loginfo("Goal pose " + str(self.goal_cnt + 1) + " reached")
            self.check_goal_count()
        else:
            rospy.logwarn("Failed to reach goal pose " + str(self.goal_cnt + 1))
            rospy.loginfo("Moving to the next goal pose")
            self.check_goal_count()

    def check_goal_count(self):
        self.goal_cnt += 1
        if self.goal_cnt < len(self.pose_seq):
            self.movebase_client()
        else:
            rospy.loginfo("All goals reached!")
            rospy.signal_shutdown("My work is done, time to go home!")


# Create the pose_seq list with the given points and a default orientation
points = [
    -0.000000,
    -0.050000,
    0.0,
    -0.200000,
    -1.350000,
    0.0,
    0.750000,
    -1.800000,
    0.0,
    2.600000,
    -1.450000,
    0.0,
    1.300000,
    0.550000,
    0.0,
]
default_orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
pose_seq = []

for i in range(0, len(points), 3):
    pose = Pose()
    pose.position = Point(points[i], points[i + 1], points[i + 2])
    pose.orientation = default_orientation
    pose_seq.append(pose)

rospy.init_node("move_base_client")
move_base_client = MoveBaseClient(pose_seq)
move_base_client.movebase_client()

rospy.spin()
