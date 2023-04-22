#!/usr/bin/python3
"""
This script is used for detecting rings in images.
"""

from typing import Tuple
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

        # Marker array object used for visualizations
        self.marker_array = MarkerArray()
        self.marker_num = 1

        # Subscribe to the image and/or depth topic
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        # self.depth_sub = rospy.Subscriber("/camera/depth_registered/image_raw", Image, self.depth_callback)
        image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
        depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
        ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 100)
        ts.registerCallback(self.image_callback)

        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher("ring_markers", MarkerArray, queue_size=1000)

        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

    def get_pose(self, elipse: Tuple[float, float], dist: float) -> None:
        """
        Calculate the pose of the detected ring.

        Args:
            elipse (Tuple[float, float]): Ellipse object
            dist (float): Distance to the ring
        """
        k_f = 525  # kinect focal length in pixels

        elipse_x = self.dims[1] / 2 - elipse[0][0]
        # elipse_y = self.dims[0] / 2 - elipse[0][1]

        angle_to_target = np.arctan2(elipse_x, k_f)

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

        # Create a marker used for visualization
        self.marker_num += 1
        marker = Marker()
        marker.header.stamp = point_world.header.stamp
        marker.header.frame_id = point_world.header.frame_id
        marker.pose = pose
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False
        marker.lifetime = rospy.Duration.from_sec(10)
        marker.id = self.marker_num
        marker.scale = Vector3(0.1, 0.1, 0.1)
        marker.color = ColorRGBA(0, 1, 0, 1)
        self.marker_array.markers.append(marker)

        self.markers_pub.publish(self.marker_array)

    def image_callback(self, rgb_image_message: Image, depth_image_message: Image) -> None:
        """
        Callback function for processing received image data.

        Args:
           rgb_image_message (Image): Image message
           depth_image_message (Image): Depth image message 
        """

        print("I got a new image!")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as err:
            print(err)

        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_message, "32FC1")
            #depth_image = self.bridge.imgmsg_to_cv2(depth_image_message, "16UC1")
        except CvBridgeError as e:
            print(e)

        # Set the dimensions of the image
        self.dims = cv_image.shape

        # Tranform image to gayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

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
            for e_2 in elps[i + 1:]:
                dist = np.sqrt(((e_1[0][0] - e_2[0][0]) ** 2 + (e_1[0][1] - e_2[0][1]) ** 2))
                if dist < 5:
                    candidates.append((e_1, e_2))

        print("Processing is done! found", len(candidates), "candidates for rings")

        try:
            depth_img = rospy.wait_for_message("/camera/depth/image_raw", Image)
        except rospy.ROSException as error:
            print(error)

        # Extract the depth from the depth image
        for candidate in candidates:
            # the centers of the ellipses
            e_1 = candidate[0]
            e_2 = candidate[1]

            # drawing the ellipses on the image
            cv2.ellipse(cv_image, e_1, (0, 255, 0), 2)
            cv2.ellipse(cv_image, e_2, (0, 255, 0), 2)

            size = (e_1[1][0] + e_1[1][1]) / 2
            center = (e_1[0][1], e_1[0][0])

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1 > 0 else 0
            x_max = x2 if x2 < cv_image.shape[0] else cv_image.shape[0]

            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < cv_image.shape[1] else cv_image.shape[1]

            depth_image = self.bridge.imgmsg_to_cv2(depth_img, "16UC1")

            self.get_pose(e_1, float(np.mean(depth_image[x_min:x_max, y_min:y_max])) / 1000.0)

        # if len(candidates) > 0:
        #     cv2.imshow("Image window", cv_image)
        #     cv2.waitKey(1)


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
