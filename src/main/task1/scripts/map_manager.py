#!/usr/bin/python3
import numpy as np
import rospy
import threading
import cv2
import tf2_geometry_msgs
from matplotlib import pyplot as plt
from nav_msgs.msg import OccupancyGrid
from skimage.morphology import skeletonize
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PointStamped, Vector3, Point, Quaternion, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_geometry_msgs import PointStamped
from typing import Tuple


# This script retrieves the map from the map_server and saves it in a 2D array.
# It also uses ekeletonize to get most important points to visit in the map.
# The points to visit are published as markers in rviz.


class MapManager:
    """
    A class for managaing a map, processing it and publishing markers for points to visit.
    """

    def __init__(self):
        self.map = None
        self.skeleton_overlay = None
        self.branch_points = None
        self.map_subscriber = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.map_lock = threading.Lock()  # Add a lock for the map attribute
        self.marker_publisher = rospy.Publisher(
            "goal_markers", MarkerArray, queue_size=100
        )
        self.goals_ready = False
        self.goal_points = []
        self.map_transform = TransformStamped()
        self.map_resolution = None
        self.size_x = None
        self.size_y = None
        self.map_frame_id = None

    def map_callback(self, data) -> None:
        """
        Callback function for the map subscriber. It processes the map data when received.
        """
        with self.map_lock:  # Acquire the lock before modifying the map attribute
            self.map_processing(data)

    def get_map(self) -> np.ndarray:
        """
        Get the current map.

        Returns:
            numpy.ndarray: The current map
        """
        with self.map_lock:  # Acquire the lock before accessing the map attribute
            return self.map

    def is_ready(self):
        """
        Check if the goals are ready.

        Returns:
            bool: True if the goals are ready, False otherwise.
        """
        with self.map_lock:
            return self.goals_ready

    def get_goals(self):
        """Get the list of goals

        Returns:
            list: List of goals.
        """
        with self.map_lock:
            return self.goal_points

    def map_processing(self, map_data):
        """
        Process the map data to find goals and publish markers

        Args:
            map_data (OccupancyGrid): Map data to process.
        """

        self.size_x = map_data.info.width
        self.size_y = map_data.info.height

        rospy.loginfo("Map size: x: %s, y: %s." % (str(self.size_x), str(self.size_y)))

        if self.size_x < 3 or self.size_y < 3:
            rospy.loginfo(
                "Map size only: x: %s, y: %s. NOT running map to image conversion."
                % (str(self.size_x), str(self.size_y))
            )
            return

        # Get the map properties from the map topic
        self.map_resolution = map_data.info.resolution
        self.map_frame_id = map_data.header.frame_id
        self.map_transform.transform.translation.x = map_data.info.origin.position.x
        self.map_transform.transform.translation.y = map_data.info.origin.position.y
        self.map_transform.transform.translation.z = map_data.info.origin.position.z
        self.map_transform.transform.rotation = map_data.info.origin.orientation

        # 2D array representing the map
        self.map = np.array(map_data.data).reshape((self.size_y, self.size_x))

        # Assign correct values to the map (0 = free, 100 = occupied, -1 = unknown)
        self.map[self.map == -1] = 127
        self.map[self.map == 0] = 255
        self.map[self.map == 100] = 0

        # Convert the map to uint8
        # Remove small obstacles and enlarge free spaces in the map
        # This makes it safer for the robot to navigate
        self.map = self.map.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        semi_safe_map = cv2.erode(self.map, kernel, iterations=3)
        self.map = semi_safe_map

        # Find the skeleton overlay of the map
        self.skeleton_overlay = self.skeletonize_map()

        # find branch points note the map is flipped
        self.branch_points = self.find_branch_points()

        # self.visualize_branch_points()

        self.init_goals()

    def init_goals(self) -> None:
        goals = []
        for i in range(len(self.branch_points)):
            goals.append(
                self.map_to_world_coords(
                    self.branch_points[i][0], self.branch_points[i][1]
                )
            )

        self.goal_points = goals
        self.publish_markers_of_goals(goals)
        self.goals_ready = True

    def skeletonize_map(self) -> np.ndarray:
        # Skeletonize the map: The map is flipped vertically, skletenozied
        # and then flipped back. The skeletonization process finds the "skeleton"
        # of the free space in the map, which is a thinned, single-pixel wide
        # version of the original free space
        flipped_map = np.flip(self.map, 0)
        skeleton = skeletonize(flipped_map)
        skeleton = np.flip(skeleton, 0)

        # Create a skeleton image
        skeleton_image = np.zeros_like(skeleton, dtype=np.uint8)
        skeleton_image[skeleton] = 255

        # Extract the skeleton from the original map: create a new
        # image containing only the skeleton pixels where the original map
        # was free (255)
        skeleton_overlay = np.zeros_like(self.map)
        skeleton_overlay[self.map == 255] = skeleton_image[self.map == 255]

        return skeleton_overlay

    def publish_markers_of_goals(self, goals: list) -> None:
        """
        Publish markers for the goals in RViz

        Args:
            goals (list): List of goals.
        """

        marker_array = MarkerArray()

        for i, goal in enumerate(goals):
            marker = Marker()
            marker.id = i
            marker.pose.position = Point(goal[0], goal[1], 0)
            marker.header.stamp = rospy.Time(0)
            marker.header.frame_id = "map"
            marker.pose.orientation = Quaternion(0.5, 0.5, 0.5, 0.5)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.frame_locked = False
            marker.lifetime = rospy.Duration(2)
            marker.scale = Vector3(0.1, 0.1, 0.1)
            marker.color = ColorRGBA(1.0, 0.5, 0.0, 1.0)  # orange color
            marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)

    def map_to_world_coords(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform map coordinates to world coordinates.

        Args:
            x (float): The x-coordinate in the map frame.
            y (float): The y-coordinate in the map frame.

        Returns:
            Tuple[float, float]: A tuple containing the x and y coordinates in the world frame.
        """
        pt = PointStamped()
        pt.point.x = x * self.map_resolution
        pt.point.y = y * self.map_resolution
        pt.point.z = 0.0

        # transform to goal space
        transformed_pt = tf2_geometry_msgs.do_transform_point(pt, self.map_transform)

        return (transformed_pt.point.x, transformed_pt.point.y)

    def find_branch_points(self) -> list:
        """
        Find branch points in a given skeleton image.

        Args:
            skeleton_image (np.ndarray): A binary image containing the skeleton.

        Returns:
            list: A list of tuples (x, y) containing the branch point coordinates.
        """
        # Create a copy of the input image to avoid modifying the original
        skeleton_copy = self.skeleton_overlay.copy()

        # Apply the Harris corner detector to find corners in the image
        harris_corners = cv2.cornerHarris(skeleton_copy, blockSize=9, ksize=5, k=0.04)

        # Dilate the result to make the corners more visible
        harris_corners_dilated = cv2.dilate(harris_corners, None)

        # Apply thresholding to identify the optimal corners
        threshold_value = 0.32 * harris_corners_dilated.max()
        threshold_image = cv2.threshold(
            harris_corners_dilated, threshold_value, 255, 0
        )[1]
        threshold_image_uint8 = np.uint8(threshold_image)

        # Find connected components and their centroids
        centroids = cv2.connectedComponentsWithStats(threshold_image_uint8)[3]

        # Define criteria for refining corner coordinates
        stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Refine corner coordinates to subpixel accuracy
        refined_corners = cv2.cornerSubPix(
            skeleton_copy, np.float32(centroids), (5, 5), (-1, -1), stop_criteria
        )

        # Extract the branch point coordinates from the refined corners
        branch_points = [
            (int(corner[0]), int(corner[1])) for corner in refined_corners[1:]
        ]

        return branch_points

    def visualize_branch_points(self) -> None:
        """
        Visualize the branch points in the map.
        """
        bp_map = np.zeros(self.map.shape, dtype=np.uint8)

        for i in range(len(self.branch_points)):
            bp_map[self.branch_points[i][1]][self.branch_points[i][0]] = 255

        overlayed_bp = cv2.addWeighted(self.map, 0.5, bp_map, 0.5, 0)

        cv2.imshow("overlayed branch points", overlayed_bp)
        cv2.waitKey(0)


def visualize_map(map_data, title="vis"):
    plt.imshow(map_data, cmap="gray", origin="lower")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def test():
    rospy.init_node("path_setter", anonymous=True)
    ps = MapManager()

    # run node and wait for map the nshow it in plt
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        if ps.is_ready():

            goals = ps.get_goals()
            if goals is not None and len(goals) > 0:
                print("goals ready")
                print(goals)
                ps.publish_markers_of_goals(goals)
            rate.sleep()


if __name__ == "__main__":
    print("start")
    test()
