import rospy
from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


class MoveBaseClient:
    def __init__(self, pose_seq):
        self.pose_seq = pose_seq
        self.goal_cnt = 0
        self.client = SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to move_base action server")

    # Call back for when a goal is completed
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

    # Callback for when a goal becomes active
    def active_cb(self):
        rospy.loginfo(
            "Goal pose "
            + str(self.goal_cnt + 1)
            + " is now being processed by the Action Server"
        )

    # Callback for goal feedback
    def feedback_cb(self, feedback):
        pass

    # Sends the next goal in the pose sequence
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

    # MoveBase client main function
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

    # Checks if there are more goals in the sequence and sends the next goal if available
    def check_goal_count(self):
        self.goal_cnt += 1
        if self.goal_cnt < len(self.pose_seq):
            self.movebase_client()
        else:
            rospy.loginfo("All goals reached!")
            rospy.signal_shutdown("My work is done, time to go home!")
