"""
This file contains the CylinderController class.
"""

import rospy
from map_manager import MapManager
from combined.msg import CylinderGreetInstructions
# import Marker
from visualization_msgs.msg import MarkerArray

class CylinderController:
    """
    Class for controlling cylinder detections.
    """

    def __init__(self):
        rospy.Subscriber(
            "unique_cylinder_greet", CylinderGreetInstructions, self.detected_cylinder_callback
        )
        self.cylinder_coords = []
        self.cylinder_colors = []
        self.cylinder_greet_poses = []
        # self.num_all_cylinders = 10
        self.num_all_cylinders = 4
        self.all_cylinders_found = False
        self.marker_publisher = rospy.Publisher("cylinder_marker", MarkerArray, queue_size=10)
        self.num_discovered_cylinders = 0
        self.map_manager = MapManager(init_node=False)

    def detected_cylinder_callback(self, data: CylinderGreetInstructions) -> None:
        """
        Called when unique cylinder greet instructions published on topic.
        """
        cylinder_pose = data.object_pose
        cylinder_color = data.object_color

        rospy.loginfo(f"Received cylinder at position: {str(cylinder_pose)} with color {cylinder_color}")
        # rospy.loginfo(f"Received cylinder with color {cylinder_color}")

        # compute greet location and orientation
        x_cylinder = cylinder_pose.position.x
        y_cylinder = cylinder_pose.position.y

        cylinder_greet_pose = self.map_manager.get_object_greet_pose(x_cylinder, y_cylinder)

        self.cylinder_coords.append(cylinder_pose)
        self.cylinder_colors.append(cylinder_color)
        self.cylinder_greet_poses.append(cylinder_greet_pose)
        self.num_discovered_cylinders = self.num_discovered_cylinders + 1

        if self.num_discovered_cylinders >= self.num_all_cylinders:
            self.all_cylinders_found = True

    # def have_cylinders_to_visit(self) -> bool:
    #     """
    #     check if there are cylinders to visit

    #     Returns:
    #         bool: True if there are cylinders to visit else False
    #     """
    #     return len(self.cylinder_coords) > 0

    # def backupManouversCylinderQr(self, object_pose, current_greet_pose):
    #     """
    #     Conducts manouvers to read qr from different angles
    #     """
    #     print("QR not found. Performing additional actions.")
    #     x_c = object_pose.position.x
    #     y_c = object_pose.position.y

    #     x_g = current_greet_pose.position.x
    #     y_g = current_greet_pose.position.y

    #     # get candidates on circle
    #     # r = 0.5
    #     r_margin = 0.05
    #     r = self.map_manager.euclidean_distance(x_c, y_c, x_g, y_g) + r_margin
    #     circle_candidates = []
    #     step_deg = 30

    #     for i in range(0, 360, step_deg):
    #         y = y_c + r * math.sin(i)
    #         x = x_c + r * math.cos(i)
    #         if self.map_manager.can_move_to(x, y):
    #             circle_candidates.append((x, y))

    #     # print(circle_candidates)

    #     # go through all candidates and look at cylinder from that direction
    #     for can in circle_candidates:
    #         current_greet_pose.position.x = can[0]
    #         current_greet_pose.position.y = can[1]

    #         # set rotation
    #         q_dest = self.map_manager.quaternion_from_points(can[0], can[1], x_c, y_c)
    #         r_x = q_dest[0]
    #         r_y = q_dest[1]
    #         r_z = q_dest[2]
    #         r_w = q_dest[3]
    #         current_greet_pose.orientation.x = r_x
    #         current_greet_pose.orientation.y = r_y
    #         current_greet_pose.orientation.z = r_z
    #         current_greet_pose.orientation.w = r_w

    #         self.send_goal_pose(current_greet_pose)
    #         qr_read_data = self.qr_reader.readQrCode()
    #         if qr_read_data != None:
    #             return qr_read_data

    #     print("Did not manage to find qr code for cylinder.")
    #     return None

    # def backupManouversCylinderQrSort(self, object_pose, current_greet_pose):
    #     """
    #     Conducts manouvers to read qr from different angles
    #     """
    #     print("QR not found. Performing additional actions.")
    #     x_c = object_pose.position.x
    #     y_c = object_pose.position.y

    #     x_g = current_greet_pose.position.x
    #     y_g = current_greet_pose.position.y

    #     # get candidates on circle
    #     # r = 0.5
    #     r_margin = 0.0
    #     r = self.map_manager.euclidean_distance(x_c, y_c, x_g, y_g) + r_margin
    #     circle_candidates = []
    #     candidate_greet_dists = []
    #     step_deg = 90

    #     for i in range(0, 360, step_deg):
    #         y = y_c + r * math.sin(i)
    #         x = x_c + r * math.cos(i)
    #         circle_candidates.append((x, y))
    #         candidate_greet_dists.append(self.map_manager.euclidean_distance(x, y, x_g, y_g))

    #     # print(circle_candidates)

    #     # sorted_candidates = [x for _,x in sorted(zip(circle_candidates, candidate_greet_dists))]
    #     sorted_candidates = [x for _, x in sorted(zip(candidate_greet_dists, circle_candidates))]
    #     # print("candidates: ")
    #     # print(sorted_candidates)
    #     selected_candidates = sorted_candidates[0:-1]
    #     # print(selected_candidates)

    #     # sort candidates according to distance from greet_pose

    #     # go through all candidates and look at cylinder from that direction
    #     for can in circle_candidates:
    #         x_pos, y_pos = self.map_manager.get_nearest_accessible_point(can[0], can[1])
    #         current_greet_pose.position.x = x_pos
    #         current_greet_pose.position.y = y_pos

    #         # set rotation
    #         q_dest = self.map_manager.quaternion_from_points(
    #             current_greet_pose.position.x, current_greet_pose.position.y, x_c, y_c
    #         )
    #         r_x = q_dest[0]
    #         r_y = q_dest[1]
    #         r_z = q_dest[2]
    #         r_w = q_dest[3]
    #         current_greet_pose.orientation.x = r_x
    #         current_greet_pose.orientation.y = r_y
    #         current_greet_pose.orientation.z = r_z
    #         current_greet_pose.orientation.w = r_w

    #         self.send_goal_pose(current_greet_pose)
    #         qr_read_data = self.qr_reader.readQrCode()
    #         if qr_read_data != None:
    #             return qr_read_data

    #     print("Did not manage to find qr code for cylinder.")
    #     return None

    # def greet_cylinder(self, object_pose, color, current_greet_pose, person_obj):
    #     """
    #     Greets the cylinder and returns the vaccine color that was predicted
    #     from data in classifier
    #     """
    #     rospy.loginfo("Greeting cylinder")
    #     # self.send_goal(-1.7, 1.8, 1.0, store_state=False)
    #     # self.send_goal_pose(pose)
    #     self.send_goal_pose(current_greet_pose)

    #     # if neccessary move closer to object with twist messages
    #     dist = self.move_as_close_to_as_possible(object_pose.position.x, object_pose.position.y)

    #     # extend robot arm
    #     self.arm_mover.arm_movement_pub.publish(self.arm_mover.extend)

    #     # says cylinder color
    #     self.say_something(color)

    #     # get data from qr code
    #     qr_read_data = self.qr_reader.readQrCode()
    #     print("Brain read the following from QR: " + str(qr_read_data))

    #     # wait for arm to be extended
    #     rospy.sleep(1)

    #     # retract arm again
    #     self.arm_mover.arm_movement_pub.publish(self.arm_mover.retract)

    #     # move back to position where you started closing in
    #     self.move_forward(dist + 0.05, 0.1, False)

    #     if qr_read_data == None:
    #         # try reading from further away
    #         qr_read_data = self.qr_reader.readQrCode()

    #     if qr_read_data == None:
    #         # if we failed to read qr data
    #         # qr_read_data = self.backupManouversCylinderQr(object_pose, current_greet_pose)
    #         qr_read_data = self.backupManouversCylinderQrSort(object_pose, current_greet_pose)

    #     # if still nothing found
    #     if qr_read_data == None:
    #         return None

    #     # read file from link in qr code and train a classifier
    #     csm = ClassificationManager()
    #     csm.train_classifier(qr_read_data)

    #     # classify patient
    #     # find the patient with doctor color
    #     vaccine_color = csm.do_prediction(person_obj.age, person_obj.exercise_time)

    #     print("The following vaccine color was predicted: " + vaccine_color)

    #     return vaccine_color

    # def get_data_from_cylinder(self, object_pose, color, current_greet_pose):
    #     """
    #     Greets the cylinder and returns the data that it read from classifier
    #     """
    #     rospy.loginfo("Greeting cylinder")
    #     # self.send_goal(-1.7, 1.8, 1.0, store_state=False)
    #     # self.send_goal_pose(pose)
    #     self.send_goal_pose(current_greet_pose)

    #     # if neccessary move closer to object with twist messages
    #     dist = self.move_as_close_to_as_possible(object_pose.position.x, object_pose.position.y)

    #     # extend robot arm
    #     self.arm_mover.arm_movement_pub.publish(self.arm_mover.extend)

    #     # says cylinder color
    #     self.say_something(color)

    #     # get data from qr code
    #     qr_read_data = self.qr_reader.readQrCode()
    #     print("Brain read the following from QR: " + str(qr_read_data))

    #     # wait for arm to be extended
    #     rospy.sleep(1)

    #     # retract arm again
    #     self.arm_mover.arm_movement_pub.publish(self.arm_mover.retract)

    #     # move back to position where you started closing in
    #     self.move_forward(dist + 0.05, 0.1, False)

    #     return qr_read_data

    # def visit_found_cylinders(self):
    #     """
    #     Visits the rings found until now
    #     """
    #     while self.have_cylinders_to_visit():
    #         if rospy.is_shutdown():
    #             return

    #         rospy.loginfo("Ring queue: %s" % (str(self.cylinder_colors)))
    #         current_cylinder_pose = self.cylinder_coords.pop(0)
    #         current_cylinder_color = self.cylinder_colors.pop(0)
    #         current_greet_pose = self.cylinder_greet_poses.pop(0)
    #         self.greet_cylinder(
    #             current_cylinder_pose, current_cylinder_color, current_greet_pose, None
    #         )
