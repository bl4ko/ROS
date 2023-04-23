#!/usr/bin/python3
"""
This script is used for detecting rings in images.
"""

from typing import Tuple, List
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


class DetectedRing:
    """
    Class for holding information about detected rings.
    """

    def __init__(
        self,
        center: Tuple[float, float],
        radius: float,
        dist: float,
        pose: Pose,
        color: ColorRGBA,
    ):
        """
        Args:
            center (Tuple[float, float]): Center of the ring
            radius (float): Radius of the ring
            dist (float): Distance to the ring
        """
        self.center = center
        self.radius = radius
        self.dist = dist
        self.pose: Pose = pose
        self.color = color

    def __str__(self):
        return f"Center: {self.center}, Radius: {self.radius}, Distance: {self.dist}"

    def __repr__(self):
        return f"Center: {self.center}, Radius: {self.radius}, Distance: {self.dist}"


class RingGroup:
    """
    Class to store the information of a group of rings.
    A group of rings is a group of rings that are close to each other
    and are on the same plane (they represent the same ring in real world).
    """

    def __init__(self, initial_ring: DetectedRing, group_id: int) -> None:
        self.group_id = group_id
        self.rings = [initial_ring]
        self.detections = 1
        self.color = initial_ring.color
        self.avg_pose = initial_ring.pose

    def update_avg_pose(self) -> None:
        """
        Updates the average pose of the group of rings.
        """
        avg_x = 0
        avg_y = 0
        avg_z = 0
        for ring in self.rings:
            avg_x += ring.pose.position.x
            avg_y += ring.pose.position.y
            avg_z += ring.pose.position.z
        avg_x /= len(self.rings)
        avg_y /= len(self.rings)
        avg_z /= len(self.rings)
        self.avg_pose.position.x = avg_x
        self.avg_pose.position.y = avg_y
        self.avg_pose.position.z = avg_z

    def add_ring(self, ring: DetectedRing) -> None:
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

    def __init__(self):
        rospy.init_node("image_converter", anonymous=True)

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
        self.ring_max_distance: float = 0.5

        self.ring_groups: List[RingGroup] = []

    def image_callback(
        self, rgb_image_message: Image, depth_image_message: Image
    ) -> None:
        """
        Callback function for processing received image data.

        Args:
           rgb_image_message (Image): Image message
           depth_image_message (Image): Depth image message
        """

        print("I got a new image!")

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_message, "bgr8")
        except CvBridgeError as err:
            print(err)

        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_message, "32FC1")
        except CvBridgeError as err:
            print(err)

        # Set the dimensions of the image
        self.dims = rgb_image.shape

        # Tranform image to gayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Do histogram equlization
        img = cv2.equalizeHist(gray)

        # Binarize the image, there are different ways to do it
        # ret, thresh = cv2.threshold(img, 50, 255, 0)
        # ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 25
        )

        # Extract contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Example how to draw the contours, only for visualization purposes
        # cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
        # cv2.imshow("Contour window", img)
        # cv2.waitKey(1)

        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            #     print cnt
            #     print cnt.shape
            if cnt.shape[0] >= 20:
                ellipse = cv2.fitEllipse(cnt)
                elps.append(ellipse)

        # Find two elipses with same centers
        candidates = []
        for i, e_1 in enumerate(elps):
            for e_2 in elps[i + 1 :]:
                dist = np.sqrt(
                    ((e_1[0][0] - e_2[0][0]) ** 2 + (e_1[0][1] - e_2[0][1]) ** 2)
                )
                if dist < 5:
                    candidates.append((e_1, e_2))

        print("Processing is done! found", len(candidates), "candidates for rings")

        # Extract the depth from the depth image
        for candidate in candidates:
            # the centers of the ellipses
            e_1 = candidate[0]
            e_2 = candidate[1]

            # drawing the ellipses on the image
            # cv2.ellipse(rgb_image, e_1, (0, 255, 0), 2)
            # cv2.ellipse(rgb_image, e_2, (0, 255, 0), 2)

            size = (e_1[1][0] + e_1[1][1]) / 2
            center = (e_1[0][1], e_1[0][0])

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1 > 0 else 0
            x_max = x2 if x2 < rgb_image.shape[0] else rgb_image.shape[0]

            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < rgb_image.shape[1] else rgb_image.shape[1]

            color = rgb_image[int(center[0]), int(center[1])]
            color_rgb = color[::-1]
            color_rgba = ColorRGBA(*color_rgb, 1)

            self.get_pose(
                e_1,
                float(np.mean(depth_image[x_min:x_max, y_min:y_max])) / 1000.0,
                color_rgba=color_rgba,
            )

        # if len(candidates) > 0:
        #     cv2.imshow("Image window", cv_image)
        #     cv2.waitKey(1)

    def get_pose(
        self,
        ellipse: Tuple[Tuple[float, float], Tuple[float, float], float],
        dist: float,
        color_rgba: ColorRGBA,
    ) -> None:
        """
        Calculate the pose of the detected ring.

        Args:
            elipse (Tuple[float, float]): Ellipse object
            dist (float): Distance to the ring
        """
        k_f = 525  # kinect focal length in pixels

        ellipse_x = self.dims[1] / 2 - ellipse[0][0]
        # elipse_y = self.dims[0] / 2 - elipse[0][1]

        angle_to_target = np.arctan2(ellipse_x, k_f)

        # Get the angles in the base_link relative coordinate system
        x, y = dist * np.cos(angle_to_target), dist * np.sin(angle_to_target)

        ### Define a stamped message for transformation - directly in "base_frame"
        # point_s = PointStamped()
        # point_s.point.x = x
        # point_s.point.y = y
        # point_s.point.z = 0.3
        # point_s.header.frame_id = "base_link"
        # point_s.header.stamp = rospy.Time(0)

        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped()
        point_s.point.x = -y
        point_s.point.y = 0
        point_s.point.z = x
        point_s.header.frame_id = "camera_rgb_optical_frame"
        point_s.header.stamp = rospy.Time(0)

        # Get the point in the "map" coordinate system
        point_world = self.tf_buf.transform(point_s, "map")

        # Create a Pose object with the same position
        pose = Pose()
        pose.position.x = point_world.point.x
        pose.position.y = point_world.point.y
        pose.position.z = point_world.point.z

        new_ring = DetectedRing(
            center=ellipse[0],
            radius=(ellipse[1][0] + ellipse[1][1]) / 4,
            dist=dist,
            pose=pose,
            color=color_rgba,
        )
        self.add_new_ring(new_ring)

    def add_new_ring(self, ring: DetectedRing) -> None:
        """
        Add a new ring to the list of detected rings.

        Args:
            ring (DetectedRing): The ring to add
        """
        # For each group compare if avg_group pose is smaller than 0.5m from the new ring
        # In that case that means that the new ring is part of the group
        for i, ring_group in enumerate(self.ring_groups):
            if (
                np.sqrt(
                    (ring_group.avg_pose.position.x - ring.pose.position.x) ** 2
                    + (ring_group.avg_pose.position.y - ring.pose.position.y) ** 2
                    + (ring_group.avg_pose.position.z - ring.pose.position.z)
                )
                < self.ring_max_distance
            ):
                self.ring_groups[i].add_ring(ring)
                rospy.loginfo(f"Ring added to group {str(ring_group.group_id)}")
                return

        # If the ring is not part of any group, create a new group
        new_ring_group = RingGroup(ring, len(self.ring_groups))
        rospy.loginfo(f"New ring group created: {str(new_ring_group.group_id)}")
        self.ring_groups.append(new_ring_group)

    def publish_markers(self) -> None:
        """
        For each ring group publish a marker in available on ring_markers topic.
        """
        # Marker array object used for visualizations
        marker_array = MarkerArray()
        for ring_group in self.ring_groups:
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = "map"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = ring_group.avg_pose
            marker.scale = Vector3(
                ring_group.rings[0].radius * 2, ring_group.rings[0].radius * 2, 0.01
            )
            marker.color = ring_group.color
            marker.lifetime = rospy.Duration.from_sec(10)
            marker.id = ring_group.group_id
            marker_array.markers.append(marker)

        self.markers_pub.publish(marker_array)


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
