#!/usr/bin/python3

# TODO: # pylint: disable=fixme
# pylint: disable=too-many-instance-attributes, disable=too-many-locals, disable=too-many-arguments, disable=too-many-locals, disable=duplicate-code

"""
This script is used for detecting rings in images.
"""

import sys
import threading
from typing import Tuple, List
import rospy
import cv2
import numpy as np
from tf2_ros import TransformException  # pylint: disable=no-name-in-module
import tf2_ros
import message_filters
from tf2_geometry_msgs import PointStamped
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3, Pose, Quaternion, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from combined.msg import UniqueRingCoords, DetectedRings

# Tuple of center coordinates (x,y)
# Tuple of axes lengths (width, height)
# Float the angle of rotation of the elipse
Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]

LAST_PROCESSED_IMAGE_TIME = 0  # Variable for storing the time of the last processed image


class DetectedGroundRing:
    """
    Class for holding information about detected rings.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        pose: Pose,
        robot_pose: PoseStamped = None,
    ):
        """
        Args:
            pose (Pose): Pose of the ring
            robot_pose (PoseStamped): Pose of the robot
        """
        self.pose: Pose = pose
        self.robot_pose: PoseStamped = robot_pose


class GroundRingGroup:
    """
    Class to store the information of a group of rings.
    A group of rings is a group of rings that are close to each other
    and are on the same plane (they represent the same ring in real world).
    """

    def __init__(self, initial_ring: DetectedGroundRing, group_id: int) -> None:
        self.group_id: int = group_id
        self.rings: List[DetectedGroundRing] = [initial_ring]
        self.detections: int = 1
        self.avg_pose: Pose = initial_ring.pose

    def __str__(self) -> str:
        return (
            f"Ring group [{self.group_id}]: detections={self.detections}"
            f" y={self.avg_pose.position.y}, z={self.avg_pose.position.z})"
        )

    def update_avg_pose(self) -> None:
        """
        Updates the average pose of the group of rings based on distance between
        ring pose and robot pose. Closer is better.
        """
        avg_pose = Pose()
        avg_pose.position.x = 0
        avg_pose.position.y = 0
        avg_pose.position.z = 0
        avg_pose.orientation = Quaternion(0, 0, 0, 1)  # Just to quiet the rviz

        total_weight = 0

        for ring in self.rings:
            robot_pose = ring.robot_pose.pose
            ring_pose = ring.pose

            # Calculate the distance between ring pose and robot pose
            distance = (
                (ring_pose.position.x - robot_pose.position.x) ** 2
                + (ring_pose.position.y - robot_pose.position.y) ** 2
            ) ** 0.5

            # Invert the distance to assign more weight to closer rings
            weight = 1 / distance

            total_weight += weight

            # Add the weighted positions
            avg_pose.position.x += ring_pose.position.x * weight
            avg_pose.position.y += ring_pose.position.y * weight
            avg_pose.position.z += ring_pose.position.z * weight

        # Divide by the total weight to get the weighted average
        avg_pose.position.x /= total_weight
        avg_pose.position.y /= total_weight
        avg_pose.position.z /= total_weight

        self.avg_pose = avg_pose

    def add_ring(self, ring: DetectedGroundRing) -> None:
        """
        Adds a ring to the group of rings.

        Args:
            ring (DetectedRing): Detected ring to be added to the group.
        """
        self.rings.append(ring)
        self.update_avg_pose()
        self.detections += 1


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
        image_sub = message_filters.Subscriber("/arm_camera/rgb/image_raw", Image)
        depth_sub = message_filters.Subscriber("/arm_camera/depth/image_raw", Image)
        time_synchronizer = message_filters.TimeSynchronizer([image_sub, depth_sub], 100)
        time_synchronizer.registerCallback(self.image_callback)
        self.image_lock = threading.Lock()

        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher("ground_ring_markers", MarkerArray, queue_size=1000)

        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        # Max distance for the ring to be considered part of the group
        self.group_max_distance: float = 0.3

        # Max distance for the ring detection to be considered valid
        self.max_distance: float = 5.0

        # Max depth difference for the ring to be considered valid
        self.depth_difference_threshold: float = 0.1

        self.ring_groups: List[GroundRingGroup] = []

        self.min_num_of_detections: int = 3

        # Number of all the rings to be found
        # After this number is achieved this node exits
        self.num_of_all_rings: int = 10

        self.ring_group_publisher = rospy.Publisher(
            "detected_ground_ring_coords", DetectedRings, queue_size=10
        )

        self.rgb_image_message: Image = None
        self.depth_image_message: Image = None

        # Threshold for the minimum size of the ellipse
        self.min_size_threshold: float = 40

    def image_callback(self, rgb_image_msg: Image, depth_image_msg: Image):
        """
        Callback for when a new image is received.

        Args:
            rgb_image_msg (np.ndarray): The RGB image.
            depth_image_msg (np.ndarray): The depth image.
        """
        with self.image_lock:
            self.rgb_image_message = rgb_image_msg
            self.depth_image_message = depth_image_msg

    def detect_rings(self) -> None:
        """
        Callback function for processing received image data.

        Args:
           rgb_image_message (Image): Image message
           depth_image_message (Image): Depth image message
        """

        with self.image_lock:
            rospy.loginfo("I got a new image!")

            # Convert the image messages to OpenCV formats
            try:
                rgb_img = self.bridge.imgmsg_to_cv2(self.rgb_image_message, "bgr8")
            except CvBridgeError as err:
                print(err)

            try:
                depth_img = self.bridge.imgmsg_to_cv2(self.depth_image_message, "32FC1")
            except CvBridgeError as err:
                print(err)

            depth_timestamp = self.depth_image_message.header.stamp

            try:
                base_position_transform = self.tf_buf.lookup_transform(
                    "map", "base_link", depth_timestamp
                )
                robot_x = base_position_transform.transform.translation.x
                robot_y = base_position_transform.transform.translation.y
                robot_z = base_position_transform.transform.translation.z
                robot_rotation_x = base_position_transform.transform.rotation.x
                robot_rotation_y = base_position_transform.transform.rotation.y
                robot_rotation_z = base_position_transform.transform.rotation.z
                robot_rotation_w = base_position_transform.transform.rotation.w

                robot_pose = PoseStamped()
                robot_pose.header.frame_id = "map"
                robot_pose.header.stamp = depth_timestamp
                robot_pose.pose.position.x = robot_x
                robot_pose.pose.position.y = robot_y
                robot_pose.pose.position.z = robot_z
                robot_pose.pose.orientation.x = robot_rotation_x
                robot_pose.pose.orientation.y = robot_rotation_y
                robot_pose.pose.orientation.z = robot_rotation_z
                robot_pose.pose.orientation.w = robot_rotation_w

            except TransformException as error:
                rospy.logerr(f"Error in base coords lookup: {error}")
                return

            # Get the time stamp of the depth image
            depth_img_time = self.depth_image_message.header.stamp

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

            if len(candidates) > 0:
                self.process_candidates_and_update_ring_groups(
                    candidates, rgb_img, depth_img, depth_img_time, robot_pose
                )

            self.publish_ring_groups()
            self.publish_ring_groups_coords()

    def process_candidates_and_update_ring_groups(
        self,
        candidates: List[Tuple[Ellipse, Ellipse]],
        rgb_img: np.ndarray,
        depth_img: np.ndarray,
        depth_img_time: rospy.Time,
        robot_pose: PoseStamped,
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

            if inner_avg_size < self.min_size_threshold:
                rospy.logdebug("Candidate not valid, inner ellipse too small")
                continue

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

            debug_img = rgb_img.copy()
            cv2.ellipse(debug_img, inner_ellipse, (0, 255, 0), 2)
            cv2.ellipse(debug_img, outer_ellipse, (0, 255, 0), 2)
            cv2.circle(debug_img, (candidate_center_y, candidate_center_x), 2, (0, 0, 255), 2)

            # self.debug_image_with_mouse(debug_img)
            # Convert nan to 0 in center_depth_slice
            center_depth_slice = np.nan_to_num(center_depth_slice)

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
                self.add_new_ring(ring_pose=ring_pose, robot_pose=robot_pose)
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
        rospy.logdebug("dist: ", dist)
        rospy.logdebug("ellipse_x: ", ellipse_x)
        rospy.logdebug("angle_to_target: ", angle_to_target)

        # Get the angles in the base_link relative coordinate system
        x_coord, y_coord = dist * np.cos(angle_to_target), dist * np.sin(angle_to_target)
        rospy.logdebug("x, y: ", x_coord, y_coord)

        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped()
        point_s.point.x = -y_coord
        point_s.point.y = 0
        point_s.point.z = x_coord
        point_s.header.frame_id = "arm_camera_rgb_optical_frame"
        point_s.header.stamp = stamp

        try:
            # Get the point in the "map" coordinate system
            point_world = self.tf_buf.transform(point_s, "map")
            # print("point_world: ", point_world)

            # Create a Pose object with the same position
            pose = Pose()
            pose.position.x = point_world.point.x
            pose.position.y = point_world.point.y
            pose.position.z = 0

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
            return None

        return pose

    def add_new_ring(self, ring_pose: Pose, robot_pose: PoseStamped) -> None:
        """
        Add a new ring_img to the list of detected rings.

        Args:
            TODO
        """
        new_ring = DetectedGroundRing(pose=ring_pose, robot_pose=robot_pose)

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
        new_ring_group = GroundRingGroup(new_ring, len(self.ring_groups))
        rospy.loginfo(f"New ring group created: id={str(new_ring_group.group_id)}")
        self.ring_groups.append(new_ring_group)

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
                ring_group_coords.color = "black"  # Dummy color
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
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.lifetime = rospy.Duration.from_sec(10)
                marker.scale = Vector3(0.1, 0.1, 0.1)
                marker.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=1)
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


def main(log_level=rospy.INFO) -> None:
    """
    This node is used to detect rings in the image.
    """

    ring_detector = RingDetector(log_level=log_level)

    try:
        rospy.Timer(rospy.Duration(0.4), lambda _: ring_detector.detect_rings())
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
