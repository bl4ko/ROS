#!/usr/bin/python3
"""
Module for finding center of the parking spot under the pose of the green ring.
"""

import sys
from typing import Tuple, List
import time
import rospy
import cv2
import numpy as np
import tf2_ros
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from tf2_geometry_msgs import PointStamped
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from combined.msg import UniqueRingCoords

LAST_PROCESSED_IMAGE_TIME = 0  # Variable for storing the time of the last processed image
Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]


class ParkingDetector:
    """
    Class for detecting center of the ring for the robot to park.
    """

    def __init__(self, log_level: int = rospy.INFO):
        rospy.init_node("parking_detector", log_level=log_level, anonymous=True)

        # Publisher for the parking spot
        self.parking_spot_publisher = rospy.Publisher("parking_spot", Pose, queue_size=10)

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # A help variable for holding the dimensions of the image
        self.dims = (0, 0, 0)

        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()

        rospy.loginfo("Waiting for message from brain to start parking...")
        data = rospy.wait_for_message("green_ring_coords", UniqueRingCoords)
        rospy.loginfo("Received message from brain. Starting to search for parking spot...")

        self.parking_pose = data.pose
        # Max distance between parking pose and found parking spot
        self.max_distance = 0.3
        self.arm_image_sub = rospy.Subscriber("arm_camera/rgb/image_raw", Image)
        self.arm_depth_sub = rospy.Subscriber("arm_camera/depth/image_raw", Image)
        time_synchronizer = message_filters.TimeSynchronizer(
            [self.arm_image_sub, self.arm_depth_sub], 100
        )
        time_synchronizer.registerCallback(self.image_callback)

    def image_callback(self, rgb_image_message: Image, depth_image_message: Image) -> None:
        """
        Callback function for processing received image data.

        Args:
           rgb_image_message (Image): Image message
           depth_image_message (Image): Depth image message
        """
        global LAST_PROCESSED_IMAGE_TIME  # pylint: disable=global-statement

        current_time = time.time()

        if current_time - LAST_PROCESSED_IMAGE_TIME >= 1:
            LAST_PROCESSED_IMAGE_TIME = current_time

            rospy.loginfo("I got a new image!")

            # Convert the image messages to OpenCV formats
            try:
                rgb_img = self.bridge.imgmsg_to_cv2(rgb_image_message, "bgr8")
            except CvBridgeError as err:
                print(err)

            try:
                depth_img = self.bridge.imgmsg_to_cv2(depth_image_message, "32FC1")
            except CvBridgeError as err:
                print(err)

            # Get the time stamp of the depth image
            depth_img_time = depth_image_message.header.stamp

            # Set the dimensions of the image
            self.dims = rgb_img.shape

            # Tranform image to gayscale
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

            # Do histogram equlization
            img = cv2.equalizeHist(gray)

            # Binarize the image using adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 25
            )

            # Extract contours from the binary image
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Fit elipses to all extracted contours with enough points
            ellipses = [cv2.fitEllipse(cnt) for cnt in contours if cnt.shape[0] >= 20]

            # Find pairs of ellipses with centers close to each other
            candidates = [
                (e1, e2)
                for i, e1 in enumerate(ellipses)
                for e2 in ellipses[i + 1 :]
                if np.sqrt((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2) < 5
            ]

            rospy.loginfo(f"Found {len(candidates)} candidates for parking spot")

            if len(candidates) > 0:
                self.process_candidates_and_update_ring_groups(
                    candidates, rgb_img, depth_img, depth_img_time
                )

    def process_candidates_and_update_ring_groups(
        self,
        candidates: List[Tuple[Ellipse, Ellipse]],
        rgb_img: np.ndarray,
        depth_img: np.ndarray,
        depth_img_time: rospy.Time,
    ) -> None:
        """
        Process the candidates, extract their depth and position, and update the ring groups.

        Args:
            candidates (List[Tuple[Ellipse, Ellipse]]): List of candidate pairs of ellipses.
            rgb_img (np.ndarray): RGB image.
            depth_img (np.ndarray): Depth image.
            depth_img_time (rospy.Time): Time stamp of the depth image.
        """
        # Iterate over the ellipse candidate pairs to find the rings
        for inner_ellipse, outer_ellipse in candidates:
            # Calculcate the average size and center of the inner ellipse
            inner_avg_size = (inner_ellipse[1][0] + inner_ellipse[1][1]) / 2
            inner_center = (inner_ellipse[0][1], inner_ellipse[0][0])

            # Calculate the minimum and maximum x-coordinates of the inner ellipse
            inner_x_min = max(0, int(inner_center[0] - inner_avg_size / 2))
            inner_x_max = min(rgb_img.shape[0], int(inner_center[0] + inner_avg_size / 2))

            # Calculate the minimum and maximum y-coordinates of the inner ellipse
            inner_y_min = max(0, int(inner_center[1] - inner_avg_size / 2))
            inner_y_max = min(rgb_img.shape[1], int(inner_center[1] + inner_avg_size / 2))

            # Calculate the center of the candidate ellipse pair
            candidate_center_x = round((inner_x_min + inner_x_max) / 2)
            candidate_center_y = round((inner_y_min + inner_y_max) / 2)

            # Create a squared slice of the center (size center_neigh x center_neigh)
            center_neigh = 4
            center_depth_slice = depth_img[
                (candidate_center_x - center_neigh) : (candidate_center_x + center_neigh),
                (candidate_center_y - center_neigh) : (candidate_center_y + center_neigh),
            ]
            # self.debug_image_with_mouse(center_depth_slice)
            # Convert nan to 0 in center_depth_slice
            center_depth_slice = np.nan_to_num(center_depth_slice)

            if np.mean(center_depth_slice) > 0.1:
                rospy.logdebug("Not a valid ring, because inside values are not far away.")
                continue

            # Create a mask using both ellipses
            ring_mask = self.ellipse2array(
                inner_ellipse, outer_ellipse, rgb_img.shape[0], rgb_img.shape[1]
            )

            # Calculate median depth value of the ellipse, where ellipse is on the image
            median_ring_depth = float(np.nanmedian(depth_img[ring_mask == 255]))

            # Print debugging information
            rospy.logdebug(f"Median ring depth: {str(median_ring_depth)}")

            # From here on we have a valid detection -> true ring
            ring_pose = self.get_pose(
                ellipse=inner_ellipse, dist=median_ring_depth, stamp=depth_img_time
            )

            if ring_pose is not None:
                rospy.loginfo("Valid ring pose found!")
                rospy.logdebug(
                    f"Pose: (x={ring_pose.position.x}, y={ring_pose.position.y},"
                    f" z={ring_pose.position.z})"
                )
                if (
                    np.sqrt(ring_pose.position.x - self.parking_pose.position.x) ** 2
                    + (ring_pose.position.y - self.parking_pose.position.y) ** 2
                    < self.max_distance
                ):
                    rospy.loginfo("Ring is close to parking pose!")
                    self.parking_spot_publisher.publish(ring_pose)
                    rospy.loginfo("Ring pose published. My job is done. Bye!")
                    sys.exit(0)

    def ellipse2array(
        self,
        ell_in: Tuple[Tuple[float, float], Tuple[float, float], float],
        ell_out: Tuple[Tuple[float, float], Tuple[float, float], float],
        height: float,
        width: float,
    ) -> np.array:
        """
        Returns an array with ones where the ellipse is and zeros elsewhere.

        Args:
            ell_in (Tuple[Tuple[float, float], Tuple[float, float], float]): inner ellipse
            ell_out (Tuple[Tuple[float, float], Tuple[float, float], float]): outer ellipse
            height (float): height of the image
            width (float): width of the image

        Returns:
            np.array: Array with 255 where the ellipse is and 0 elsewhere.
        """
        cv_image = np.zeros((height, width, 3), np.uint8)
        cv2.ellipse(cv_image, ell_out, (255, 255, 255), -1)
        cv2.ellipse(cv_image, ell_in, (0, 0, 0), -1)

        return cv_image[:, :, 1]

    def get_pose(
        self,
        ellipse: Ellipse,
        dist: float,
        stamp: rospy.Time,
    ) -> Pose:
        """
        Calculate the pose of the detected ring.

        Args:
            elipse (Tuple[float, float]): Ellipse object
            dist (float): Distance to the ring

        Returns:
            Pose: Pose of the ring
        """

        k_f = 525  # kinect focal length in pixels

        ellipse_x = self.dims[1] / 2 - ellipse[0][0]
        # elipse_y = self.dims[0] / 2 - elipse[0][1]

        angle_to_target = np.arctan2(ellipse_x, k_f)

        # Debugging print statements
        print("dist: ", dist)
        print("ellipse_x: ", ellipse_x)
        print("angle_to_target: ", angle_to_target)

        # Get the angles in the base_link relative coordinate system
        x, y = dist * np.cos(angle_to_target), dist * np.sin(angle_to_target)
        print("x, y: ", x, y)

        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped()
        point_s.point.x = -y
        point_s.point.y = 0
        point_s.point.z = x
        point_s.header.frame_id = "camera_rgb_optical_frame"
        point_s.header.stamp = stamp
        # print("point_s: ", point_s)

        try:
            # Get the point in the "map" coordinate system
            point_world = self.tf_buf.transform(point_s, "map")
            # print("point_world: ", point_world)

            # Create a Pose object with the same position
            pose = Pose()
            pose.position.x = point_world.point.x
            pose.position.y = point_world.point.y
            pose.position.z = point_world.point.z

            # Dummy orientation to make rviz happy
            pose.orientation.x = 0
            pose.orientation.y = 0
            pose.orientation.z = 0
            pose.orientation.w = 1

        except Exception as err:  # pylint: disable=broad-except
            rospy.logwarn(
                "Transformation into real world coordinates not available, will try again later:"
                f" {err}"
            )
            pose = None
            return

        return pose


def main(log_level=rospy.INFO) -> None:
    """
    This node is used to detect rings in the image.
    """

    ParkingDetector(log_level=log_level)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    LOG_LEVEL = rospy.INFO
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        print("Debug mode enabled!")
        LOG_LEVEL = rospy.DEBUG
    main(log_level=LOG_LEVEL)
