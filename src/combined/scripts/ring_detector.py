#!/usr/bin/python3
"""
This script is used for detecting rings in images.
"""

import sys
from collections import Counter
from typing import Tuple, List
import time
import rospy
import cv2
import numpy as np
import tf2_ros
import message_filters
from tf2_geometry_msgs import PointStamped
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


LAST_PROCESSED_IMAGE_TIME = 0  # Variable for storing the time of the last processed image


class DetectedRing:
    """
    Class for holding information about detected rings.
    """

    def __init__(
        self,
        pose: Pose,
        color: ColorRGBA,
    ):
        """
        Args:
            pose (Pose): Pose of the ring
            color (ColorRGBA): Color of the ring
        """
        self.pose: Pose = pose
        self.color: ColorRGBA = color


class RingGroup:
    """
    Class to store the information of a group of rings.
    A group of rings is a group of rings that are close to each other
    and are on the same plane (they represent the same ring in real world).
    """

    def __init__(self, initial_ring: DetectedRing, group_id: int) -> None:
        self.group_id: int = group_id
        self.rings: List[DetectedRing] = [initial_ring]
        self.detections: int = 1
        self.color: ColorRGBA = initial_ring.color
        self.avg_pose: Pose = initial_ring.pose

    def __str__(self) -> str:
        return (
            f"Ring group [{self.group_id}]: detections={self.detections}, color={self.color},"
            f" pose={self.avg_pose}"
        )

    def update_avg_pose(self) -> None:
        """
        Updates the average pose of the group of rings.
        """
        avg_pose = Pose()
        avg_pose.position.x = 0
        avg_pose.position.y = 0
        avg_pose.position.z = 0
        avg_pose.orientation = (0, 0, 0, 1)  # Just to quiet the rviz

        for ring in self.rings:
            avg_pose.position.x += ring.pose.position.x
            avg_pose.position.y += ring.pose.position.y
            avg_pose.position.z += ring.pose.position.z

        avg_pose.position.x /= len(self.rings)
        avg_pose.position.y /= len(self.rings)
        avg_pose.position.z /= len(self.rings)

        self.avg_pose = avg_pose

    def update_avg_color(self) -> None:
        """
        Updates the average color of the group.
        """
        color_counter = Counter([ring.color for ring in self.rings])
        most_common_color = color_counter.most_common(1)[0][0]
        self.color = most_common_color

    def add_ring(self, ring: DetectedRing) -> None:
        """
        Adds a ring to the group of rings.

        Args:
            ring (DetectedRing): Detected ring to be added to the group.
        """
        self.rings.append(ring)
        self.update_avg_pose()
        self.update_avg_color()
        self.detections += 1


class RingDetector:
    """
    Class for detecting rings in images.
    """

    def __init__(self):
        rospy.init_node("image_converter", anonymous=True)

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # A help variable for holding the dimensions of the image
        self.dims = (0, 0, 0)

        # Subscribe to rgb/depth image, and also synchronize the topics
        image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
        depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
        time_synchronizer = message_filters.TimeSynchronizer([image_sub, depth_sub], 100)
        time_synchronizer.registerCallback(self.image_callback)

        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher("ring_markers", MarkerArray, queue_size=1000)

        self.min_num_of_detections = 3

        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        # Max distance for the ring to be considered part of the group
        self.ring_max_distance: float = 0.3
        # Max depth difference for the ring to be considered valid
        self.depth_difference_threshold: float = 0.1

        self.ring_groups: List[RingGroup] = []

        # Number of all the rings to be found
        # After this number is achieved this node exits
        self.num_of_all_rings: int = 10

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

            rospy.loginfo(f"Found {len(candidates)} candidates for rings")

            # Extract the depth from the depth image
            for candidate in candidates:
                # the centers of the ellipses
                e_1 = candidate[0]
                e_2 = candidate[1]

                # drawing the ellipses on the image
                cv2.ellipse(rgb_img, e_1, (0, 255, 0), 2)
                cv2.ellipse(rgb_img, e_2, (0, 255, 0), 2)

                size = (e_1[1][0] + e_1[1][1]) / 2
                center = (e_1[0][1], e_1[0][0])

                x1 = int(center[0] - size / 2)
                x2 = int(center[0] + size / 2)
                x_min = x1 if x1 > 0 else 0
                x_max = x2 if x2 < rgb_img.shape[0] else rgb_img.shape[0]

                y1 = int(center[1] - size / 2)
                y2 = int(center[1] + size / 2)
                y_min = y1 if y1 > 0 else 0
                y_max = y2 if y2 < rgb_img.shape[1] else rgb_img.shape[1]

                # For outer ellipse
                size_outer = (e_2[1][0] + e_2[1][1]) / 2
                center_outer = (e_1[0][1], e_1[0][0])

                x1_outer = int(center_outer[0] - size_outer / 2)
                x2_outer = int(center_outer[0] + size_outer / 2)
                x_min_outer = x1_outer if x1_outer > 0 else 0
                x_max_outer = x2_outer if x2_outer < rgb_img.shape[0] else rgb_img.shape[0]

                y1_outer = int(center_outer[1] - size_outer / 2)
                y2_outer = int(center_outer[1] + size_outer / 2)
                y_min_outer = y1_outer if y1_outer > 0 else 0
                y_max_outer = y2_outer if y2_outer < rgb_img.shape[1] else rgb_img.shape[1]

                center_x = round((x_min + x_max) / 2)
                center_y = round((y_min + y_max) / 2)

                center_neigh = 2
                center_image_depth_slice = depth_img[
                    (center_x - center_neigh) : (center_x + center_neigh),
                    (center_y - center_neigh) : (center_y + center_neigh),
                ]

                ellipse_mask = self.ellipse2array(e_1, e_2, rgb_img.shape[0], rgb_img.shape[1])
                depth_ring_content = float(np.nanmedian(depth_img[ellipse_mask == 255]))

                if len(center_image_depth_slice) <= 0:
                    continue

                depth_ring_center = (
                    np.NaN
                    if np.all(center_image_depth_slice != center_image_depth_slice)
                    else np.nanmean(center_image_depth_slice)
                )

                # Parameter to consider a hole in the middle if depth difference
                # greater than this threshold
                # from experience around 1.0 is usually the ones with holes
                # without them difference is 0
                depth_difference_threshold = 0.1
                depth_difference = abs(depth_ring_content - depth_ring_center)

                rospy.loginfo(f"Depth difference: {str(depth_difference)}")

                # if there is no hole in the middle -> proceed to next one
                if depth_difference < depth_difference_threshold:
                    # candidate not valid
                    continue

                # if object too far away -> do not consider detected (for better pose estimation)
                if depth_ring_content > self.ring_max_distance:
                    continue

                # From here on we have a valid detection -> true ring
                ring_pose = self.get_pose(e_1, depth_ring_content, depth_img_time)

                if ring_pose is not None:
                    rospy.loginfo("Valid ring pose found!")
                    ring_img = rgb_img[x_min_outer:x_max_outer, y_min_outer:y_max_outer]
                    self.add_new_ring(ring_img=ring_img, ring_pose=ring_pose)
                    self.print_ring_groups()

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
            np.array: array with ones where the ellipse is and zeros elsewhere, size of the image
        """
        cv_image = np.zeros((height, width, 3), np.uint8)
        cv2.ellipse(cv_image, ell_out, (255, 255, 255), -1)
        cv2.ellipse(cv_image, ell_in, (0, 0, 0), -1)

        return cv_image[:, :, 1]

    def get_pose(
        self,
        ellipse: Tuple[Tuple[float, float], Tuple[float, float], float],
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

        # Get the angles in the base_link relative coordinate system
        x, y = dist * np.cos(angle_to_target), dist * np.sin(angle_to_target)

        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped()
        point_s.point.x = -y
        point_s.point.y = 0
        point_s.point.z = x
        point_s.header.frame_id = "camera_rgb_optical_frame"
        point_s.header.stamp = stamp

        try:
            # Get the point in the "map" coordinate system
            point_world = self.tf_buf.transform(point_s, "map")
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

    def add_new_ring(self, ring_img: np.array, ring_pose: Pose) -> None:
        """
        Add a new ring_img to the list of detected rings.

        Args:
            TODO
        """
        # Get the ring color from ring_img
        ring_color: ColorRGBA = self.get_ring_color(ring_img=ring_img)
        new_ring = DetectedRing(pose=ring_pose, color=ring_color)

        # For each group compare if avg_group pose is smaller than 0.5m from the new ring
        # In that case that means that the new ring is part of the group
        for i, ring_group in enumerate(self.ring_groups):
            if (
                np.sqrt(
                    (ring_group.avg_pose.position.x - ring_pose.position.x) ** 2
                    + (ring_group.avg_pose.position.y - ring_pose.position.y) ** 2
                )
                < self.ring_max_distance
            ):
                self.ring_groups[i].add_ring(new_ring)
                rospy.loginfo(f"Ring added to group {str(ring_group.group_id)}")
                if ring_group.detections > self.min_num_of_detections:
                    self.publish_ring_group(ring_group)
                    # TODO: self.publish_greet_instructions(ring_group)

                    if len(self.ring_groups) > self.num_of_all_rings:
                        rospy.loginfo("All rings have been detected!")
                        rospy.loginfo("Shutting down the ring detector...")
                        sys.exit(0)

        # # If the ring is not part of any group, create a new group
        new_ring_group = RingGroup(new_ring, len(self.ring_groups))
        rospy.loginfo(f"New ring group created: {str(new_ring_group.group_id)}")
        self.ring_groups.append(new_ring_group)

    def get_ring_color(self, ring_img: np.array) -> ColorRGBA:
        """
        Returns the color of the ring.

        Args:
            ring_img (np.array): Image of the ring

        Returns:
            ColorRGBA: Color of the ring
        """
        ring_image_hsv = cv2.cvtColor(ring_img, cv2.COLOR_BGR2HSV)

        # Define color boundaries in HSV format
        color_boundaries = {
            "yellow": ([24, 80, 20], [32, 255, 255]),
            "green": ([36, 25, 25], [80, 255, 255]),
            "black": ([0, 0, 0], [180, 255, 50]),
            "blue": ([88, 120, 25], [133, 255, 255]),
            "red": ([0, 100, 20], [10, 255, 255]),
            "red2": ([160, 100, 20], [180, 255, 255]),
        }

        color_counts = []

        for lower, upper in color_boundaries.values():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            mask = cv2.inRange(ring_image_hsv, lower, upper)
            color_counts.append(np.sum(mask) / 255)

        # Merge red and red2 counts
        color_counts[-2] += color_counts[-1]
        color_counts.pop()

        color_ratios = [count / sum(color_counts) for count in color_counts]
        max_ratio_index = color_ratios.index(max(color_ratios))

        detected_color = list(color_boundaries.keys())[max_ratio_index]

        color_rgba_mapping = {
            "yellow": ColorRGBA(r=1, g=1, b=0, a=1),
            "green": ColorRGBA(r=0, g=1, b=0, a=1),
            "black": ColorRGBA(r=0, g=0, b=0, a=1),
            "blue": ColorRGBA(r=0, g=0, b=1, a=1),
            "red": ColorRGBA(r=1, g=0, b=0, a=1),
        }

        return color_rgba_mapping[detected_color]

    def publish_ring_group(self, ring_group: RingGroup) -> None:
        """
        Publishes a single ring group as a marker on the ring_markers topic.
        Args:
            ring_group (RingGroup): The ring group to be published.
        """
        marker = Marker()
        marker.header.stamp = rospy.Time(0)
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = ring_group.avg_pose
        marker.lifetime = rospy.Duration.from_sec(500)
        marker.scale = Vector3(0.1, 0.1, 0.1)
        marker.color = ring_group.color
        marker.id = ring_group.group_id

        self.markers_pub.publish(marker)

    def print_ring_groups(self) -> None:
        """
        Prints the current ring groups.
        """
        rospy.loginfo("Current ring groups:")
        for ring_group in self.ring_groups:
            rospy.loginfo(f"   {ring_group}")


def main() -> None:
    """
    This node is used to detect rings in the image.
    """

    RingDetector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
