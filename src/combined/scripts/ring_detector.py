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
from geometry_msgs.msg import Vector3, Pose, Quaternion
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from combined.msg import UniqueRingCoords, DetectedRings

# Tuple of center coordinates (x,y)
# Tuple of axes lengths (width, height)
# Float the angle of rotation of the elipse
Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]

LAST_PROCESSED_IMAGE_TIME = (
    0  # Variable for storing the time of the last processed image
)


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
            f"Ring group [{self.group_id}]: detections={self.detections}, color=(r={self.color.r},"
            f" g={self.color.g}, b={self.color.b}), pos=(x={self.avg_pose.position.x},"
            f" y={self.avg_pose.position.y}, z={self.avg_pose.position.z})"
        )

    def update_avg_pose(self) -> None:
        """
        Updates the average pose of the group of rings.
        """
        avg_pose = Pose()
        avg_pose.position.x = 0
        avg_pose.position.y = 0
        avg_pose.position.z = 0
        avg_pose.orientation = Quaternion(0, 0, 0, 1)  # Just to quiet the rviz

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
        color_counter = Counter(
            [
                (ring.color.r, ring.color.g, ring.color.b, ring.color.a)
                for ring in self.rings
            ]
        )
        most_common_color = color_counter.most_common(1)[0][0]
        self.color = ColorRGBA(*most_common_color)

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

    def convert_rgba_to_string(self) -> str:
        """
        Converts self.color to a string from rgba object.

        Returns:
            str: String representation of the color.
        """
        if self.color.r == 1 and self.color.g == 0 and self.color.b == 0:
            return "red"

        if self.color.r == 0 and self.color.g == 1 and self.color.b == 0:
            return "green"

        if self.color.r == 0 and self.color.g == 0 and self.color.b == 1:
            return "blue"

        if self.color.r == 0 and self.color.g == 0 and self.color.b == 0:
            return "black"

        return "unknown"


class RingDetector:
    """
    Class for detecting rings in images.
    """

    def __init__(self, log_level: int = rospy.INFO):
        rospy.init_node("image_converter", log_level=log_level, anonymous=True)

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # A help variable for holding the dimensions of the image
        self.dims = (0, 0, 0)

        # Subscribe to rgb/depth image, and also synchronize the topics
        image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
        depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
        time_synchronizer = message_filters.TimeSynchronizer(
            [image_sub, depth_sub], 100
        )
        time_synchronizer.registerCallback(self.image_callback)

        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher("ring_markers", MarkerArray, queue_size=1000)

        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        # Max distance for the ring to be considered part of the group
        self.group_max_distance: float = 0.3

        # Max distance for the ring detection to be considered valid
        self.max_distance: float = 5.0

        # Max depth difference for the ring to be considered valid
        self.depth_difference_threshold: float = 0.1

        self.ring_groups: List[RingGroup] = []

        self.min_num_of_detections: int = 3

        # Number of all the rings to be found
        # After this number is achieved this node exits
        self.num_of_all_rings: int = 10

        self.ring_group_publisher = rospy.Publisher(
            "detected_ring_coords", DetectedRings, queue_size=10
        )

    def image_callback(
        self, rgb_image_message: Image, depth_image_message: Image
    ) -> None:
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
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

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

            if len(candidates) > 0:
                self.process_candidates_and_update_ring_groups(
                    candidates, rgb_img, depth_img, depth_img_time
                )

            self.publish_ring_groups()
            self.publish_ring_groups_coords()

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
            inner_x_max = min(
                rgb_img.shape[0], int(inner_center[0] + inner_avg_size / 2)
            )

            # Calculate the minimum and maximum y-coordinates of the inner ellipse
            inner_y_min = max(0, int(inner_center[1] - inner_avg_size / 2))
            inner_y_max = min(
                rgb_img.shape[1], int(inner_center[1] + inner_avg_size / 2)
            )

            # Calculate the average size and center of the second ellipse (outer ellipse)
            outer_avg_size = (outer_ellipse[1][0] + outer_ellipse[1][1]) / 2
            outer_center = (inner_ellipse[0][1], inner_ellipse[0][0])

            # Calculate the minium and maximum x-coordinates of the outer ellipse
            outer_x_min = max(0, int(outer_center[0] - outer_avg_size / 2))
            outer_x_max = min(
                rgb_img.shape[0], int(outer_center[0] + outer_avg_size / 2)
            )

            # Calculate the minimum and maximum y-coordinates of the outer ellipse
            outer_y_min = max(0, int(outer_center[1] - outer_avg_size / 2))
            outer_y_max = min(
                rgb_img.shape[1], int(outer_center[1] + outer_avg_size / 2)
            )

            # Calculate the center of the candidate ellipse pair
            candidate_center_x = round((inner_x_min + inner_x_max) / 2)
            candidate_center_y = round((inner_y_min + inner_y_max) / 2)

            # Create a squared slice of the center (size center_neigh x center_neigh)
            center_neigh = 4
            center_depth_slice = depth_img[
                (candidate_center_x - center_neigh) : (
                    candidate_center_x + center_neigh
                ),
                (candidate_center_y - center_neigh) : (
                    candidate_center_y + center_neigh
                ),
            ]
            # self.debug_image_with_mouse(center_depth_slice)
            # Convert nan to 0 in center_depth_slice
            center_depth_slice = np.nan_to_num(center_depth_slice)

            if np.mean(center_depth_slice) > 0.1:
                rospy.logdebug(
                    "Not a valid ring, because inside values are not far away."
                )
                continue

            # Create a mask using both ellipses
            ring_mask = self.ellipse2array(
                inner_ellipse, outer_ellipse, rgb_img.shape[0], rgb_img.shape[1]
            )

            # Calculate median depth value of the ellipse, where ellipse is on the image
            median_ring_depth = float(np.nanmedian(depth_img[ring_mask == 255]))

            # Print debugging information
            rospy.logdebug(f"Median ring depth: {str(median_ring_depth)}")

            # if object too far away -> do not consider detected (for better pose estimation)
            if median_ring_depth > self.max_distance:
                rospy.logdebug("Candidate not valid, because too far away")
                continue

            # If there are no valid depth values in the center slice, skip to the next candidate
            # if len(center_depth_slice) <= 0:
            #     rospy.logdebug("No valid depth values in center slice")
            #     continue

            # Calculate the mean depth value at the center of the candidate ellipse pair
            # mean_center_depth = (
            #     np.NaN
            #     if np.all(center_depth_slice != center_depth_slice)
            #     else np.nanmean(center_depth_slice)
            # )

            # Print debugging information
            # rospy.logdebug(f"Mean center depth: {str(mean_center_depth)}")

            # Set a depth difference threshold to consider an object as having a hole in the middle
            # depth_difference_threshold = 0.1
            # depth_difference = abs(median_ring_depth - mean_center_depth)

            # Print debugging information
            # rospy.logdebug(f"Depth difference: {str(depth_difference)}")

            # Check if the depth difference is NaN
            # if math.isnan(depth_difference):
            #     rospy.logdebug("Candidate not valid, because depth difference is NaN")
            #     continue

            # if there is no hole in the middle -> candidate is not valid
            # if depth_difference < depth_difference_threshold:
            #     rospy.logdebug("Candidate not valid, because no hole in the middle")
            #     continue

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
                ring_img = rgb_img[outer_x_min:outer_x_max, outer_y_min:outer_y_max]
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

    def add_new_ring(self, ring_img: np.array, ring_pose: Pose) -> None:
        """
        Add a new ring_img to the list of detected rings.

        Args:
            TODO
        """
        # Get the ring color from ring_img
        ring_color: ColorRGBA = self.get_ring_color(ring_img=ring_img)
        new_ring = DetectedRing(pose=ring_pose, color=ring_color)

        # For each group compare if avg_group pose is smaller than self.group_max_dsitance
        # from the new ring. In that case that means that the new ring is part of the group
        for i, ring_group in enumerate(self.ring_groups):
            if (
                np.sqrt(
                    (ring_group.avg_pose.position.x - ring_pose.position.x) ** 2
                    + (ring_group.avg_pose.position.y - ring_pose.position.y) ** 2
                )
                < self.group_max_distance
            ):
                self.ring_groups[i].add_ring(new_ring)
                rospy.loginfo(f"Ring added to group {str(ring_group.group_id)}")
                return

        # # If the ring is not part of any group, create a new group
        new_ring_group = RingGroup(new_ring, len(self.ring_groups))
        rospy.loginfo(f"New ring group created: id={str(new_ring_group.group_id)}")
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

    def publish_ring_groups_coords(self) -> None:
        """
        Publishes the coordinates of the ring groups.
        """
        rospy.logdebug("Publishing new ring group coordinates...")

        detected_rings_msg = DetectedRings()
        for ring_group in self.ring_groups:
            if ring_group.detections >= self.min_num_of_detections:
                ring_group_coords = UniqueRingCoords()
                ring_group_coords.group_id = ring_group.group_id
                ring_group_coords.ring_pose = ring_group.avg_pose
                ring_group_coords.color = ring_group.convert_rgba_to_string()
                detected_rings_msg.array.append(ring_group_coords)

        self.ring_group_publisher.publish(detected_rings_msg)

    def publish_ring_groups(self) -> None:
        """
        Publishes ring groups as markers, so they can be visualized in rviz.
        """
        rospy.logdebug("Publishing ring groups as markers...")

        markers = MarkerArray()

        for ring_group in self.ring_groups:
            if ring_group.detections >= self.min_num_of_detections:
                marker = Marker()
                marker.header.stamp = rospy.Time(0)
                marker.header.frame_id = "map"
                marker.pose = ring_group.avg_pose
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.lifetime = rospy.Duration.from_sec(10)
                marker.scale = Vector3(0.2, 0.2, 0.2)
                marker.color = ring_group.color
                marker.id = ring_group.group_id
                markers.markers.append(marker)

        self.markers_pub.publish(markers)

    def print_ring_groups(self) -> None:
        """
        Prints the current ring groups.
        """
        print("Current ring groups:")
        for ring_group in self.ring_groups:
            print(f"   {ring_group}")

    def debug_image_with_mouse(self, img: np.ndarray) -> None:
        """
        Helper function that displays an image and prints the pixel value at the mouse position.

        Args:
            img (np.ndarray): Image to be displayed.
        """
        cv2.namedWindow("img")

        def mouse_event(event, x: float, y: float, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                # print the pixel value at the x, y coordinate of the image
                print(param[y, x])

        # Set the mouse callback function to the window
        cv2.setMouseCallback("img", mouse_event, param=img)
        rospy.logdebug(f"Image: {str(img)}")
        cv2.imshow("img", img)
        cv2.waitKey(0)


def main(log_level=rospy.INFO) -> None:
    """
    This node is used to detect rings in the image.
    """

    RingDetector(log_level=log_level)

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
