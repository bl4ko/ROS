#!/usr/bin/python3
"""
Module for managing the map data.
"""

# TODO: pylint fix: pylint: disable=fixme
# pylint: disable=too-many-lines, disable=too-many-instance-attributes, disable=too-many-locals, disable=too-many-arguments, disable=too-many-locals, disable=too-many-arguments, disable=too-many-public-methods

import threading
import math
from typing import Tuple, List
import cv2
import numpy as np
import rospy
from matplotlib import pyplot as plt
import tf2_geometry_msgs
from nav_msgs.msg import OccupancyGrid
from skimage.morphology import skeletonize
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import (
    PointStamped,
    Vector3,
    Point,
    Quaternion,
    TransformStamped,
    PoseWithCovarianceStamped,
    Pose,
)
from visualization_msgs.msg import Marker, MarkerArray
from bresenham import bresenham  # pylint: disable=import-error
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import tf2_ros
from tf.transformations import (
    quaternion_from_euler,
    quaternion_matrix,
    translation_matrix,
    inverse_matrix,
    translation_from_matrix,
    quaternion_from_matrix,
    concatenate_matrices,
)


# This script retrieves the map from the map_server and saves it in a 2D array.
# It also uses ekeletonize to get most important points to visit in the map.
# The points to visit are published as markers in rviz.
LETHAL_OBSTACLE = 0
INFLATED_OBSTACLE = 99
UNKNOWN = 127
FREE_SPACE = 255
WALL_THRESHOLD = 38


class MapManager:
    """
    A class for managaing a map, processing it and publishing markers for points to visit.
    """

    def __init__(self, show_plot=False, init_node=True):
        if init_node:
            self.map = None  # Map from /map topic
            self.cost_map = None  # Cost map from move_base/global_costmap/costmap
            self.accessible_costmap = None  # Accesible points in the cost map
            self.skeleton_overlay = None  # Overlay of the skeleton on the map
            self.branch_points = None  # Branch points in the skeleton
            # self.map_subscriber = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
            # self.cost_map_subscriber = rospy.Subscriber(
            #     "/move_base/global_costmap/costmap", OccupancyGrid, self.cost_map_callback
            # )
            self.map_lock = threading.Lock()  # Add a lock for the map attribute
            self.cost_map_lock = threading.Lock()  # Add a lock for the cost map attribute
            self.marker_publisher = rospy.Publisher("goal_markers", MarkerArray, queue_size=100)
            self.goals_ready = False
            self.goal_points = []
            self.map_transform = TransformStamped()
            self.map_resolution = None
            self.size_x = None
            self.size_y = None
            self.map_frame_id = None
            self.cost_map_ready = False
            self.bridge = CvBridge()
            self.image_pub = rospy.Publisher("/map_manager_info", Image, queue_size=10)

            # wait for messages
            rospy.loginfo("Waiting for map and cost map messages...")
            map_msg = rospy.wait_for_message("/map", OccupancyGrid, timeout=100)
            cost_map_msg = rospy.wait_for_message(
                "/move_base/global_costmap/costmap", OccupancyGrid, timeout=100
            )
            rospy.loginfo("Map and cost map messages received.")

            self.searched_space = None
            self.tf_buf = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
            self.show_plot = show_plot

            # Process the map and cost map
            # first is cost map because it is used in map_callback
            self.cost_map_callback(cost_map_msg)
            self.map_callback(map_msg)

    def map_callback(self, map_data) -> None:
        """
        Process the map data to find goals and publish markers

        Args:
            map_data (OccupancyGrid): Map data to process.
        """
        with self.map_lock:  # Acquire the lock before modifying the map attribute
            self.size_x = map_data.info.width
            self.size_y = map_data.info.height

            rospy.loginfo(f"Map size: x: {str(self.size_x)}, y: {str(self.size_y)}.")

            if self.size_x < 3 or self.size_y < 3:
                rospy.loginfo(
                    f"Map size only: x: {str(self.size_x)}, y: {str(self.size_y)}. NOT running map"
                    " to image conversion."
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
            semi_safe_map = cv2.erode(self.map, kernel, iterations=4)
            self.map = semi_safe_map

            # Find the skeleton overlay of the map
            self.skeleton_overlay = self.skeletonize_map()

            # find branch points note the map is flipped
            self.branch_points = self.find_branch_points()
            self.branch_points = self.filter_branch_points()

            # self.visualize_branch_points()

            if self.show_plot:
                self.visualize(
                    self.skeleton_overlay,
                    self.branch_points,
                    self.accessible_costmap,
                )

            self.init_searched_space()
            self.init_goals()

    def init_searched_space(self):
        """
        Initialize the searched space to be the size of the accessible costmap
        and set all points that are not accessible to -1.
        60 = not searched
        255 = searched
        0 = not accessible
        """
        # initialize searched space
        self.searched_space = np.zeros((self.size_y, self.size_x))

        # set all points that are not accessible to -1
        for i in range(self.size_y):
            for j in range(self.size_x):
                if not self.map[i, j] == 255:  # free space
                    self.searched_space[i, j] = 0

                else:
                    self.searched_space[i, j] = 60

    def update_searched_space(self):
        """
        gets the current position of the robot and updates the searched space
        in a radius around the robot of 10 pixels around the robot
        """
        try:
            self.tf_buf.can_transform("map", "base_link", rospy.Time(0), rospy.Duration(3.0))

            coords = self.tf_buf.lookup_transform("map", "base_link", rospy.Time(0))
            world_x, world_y = self.world_to_map_coords(
                coords.transform.translation.x, coords.transform.translation.y
            )

            cv2.circle(self.searched_space, (world_x, world_y), 10, 255, -1)
            self.get_get_aditional_goals()

        except Exception as exception:  # pylint: disable=broad-except
            rospy.logwarn(exception)

    def get_get_aditional_goals(self):
        """
        Returns a list of goals from the searched space that have not been searched yet.
        """
        # get the unsarched space
        unsearched_space = self.searched_space.copy()
        unsearched_space[unsearched_space == 255] = 0
        unsearched_space[unsearched_space == 60] = 255
        unsearched_space[unsearched_space == 0] = 0
        unsearched_space.astype(np.uint8)

        # convert to 8-bit single-channel image
        unsearched_space = cv2.convertScaleAbs(unsearched_space)

        # get centroids of the unsearched space
        _, _, stats, centroids = cv2.connectedComponentsWithStats(unsearched_space)

        # get the centroids that are not the background
        centroids = centroids[1:]

        additional_goals = []

        # add a dot to the unsearched space
        for i, centroid in enumerate(centroids):
            # has to be in the map bounds
            # #has to include more than 40 pixels
            # print("centroid",stats[i])

            if self.map[int(centroid[1]), int(centroid[0])] == 255 and stats[i + 1][4] > 40:
                additional_goals.append((centroid[0], centroid[1]))
                cv2.circle(unsearched_space, (int(centroid[0]), int(centroid[1])), 1, 60, -1)
            elif stats[i + 1][4] > 40:
                # centroid center is not in the safe space
                # check if any of the points in the bounding box are in the safe space
                # if so add one of those points as a goal
                # rospy.loginfo("centroid center is
                # not in the safe space searching for a point in the safe space")
                x_map, y_map, width, height = stats[i + 1][0:4]
                for x_map in range(x_map, x_map + width):
                    for y_map in range(y_map, y_map + height):
                        if self.map[y_map, x_map] == 255:
                            # rospy.loginfo("found a point in the safe space")
                            additional_goals.append((x_map, y_map))
                            cv2.circle(unsearched_space, (x_map, y_map), 1, 60, -1)
                            break
                    else:
                        continue
                    break

        cv2.imwrite("unsearched_space.png", np.flip(unsearched_space, 0))
        # convert to map coordinates
        toret = []
        for goal in additional_goals:
            x_map, y_map = self.map_to_world_coords(goal[0], goal[1])
            toret.append((x_map, y_map))

        return toret

    def distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculates the distance between two points.

        Args:
            point1 (Tuple[float, float]): The first point.
            point2 (Tuple[float, float]): The second point.

        Returns:
            float: The distance between the two points.
        """
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def filter_branch_points(self):
        """
        Filter out branch points that are too close to each other.
        """
        filtered_branch_points = []
        for point in self.branch_points:
            # get cost of point
            if not (
                self.in_map_bounds(point[0], point[1]) and self.can_move_to(point[0], point[1])
            ):
                print("Point is not in map bounds or can not move to", point)
                continue

            # get closest point
            closest_point = None
            closest_distance = np.inf
            for filtered_point in filtered_branch_points:
                distance = self.distance(point, filtered_point)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_point = filtered_point

            # remove if there is a clear path between the two points
            to_meters = closest_distance * self.map_resolution
            if closest_point is not None and to_meters < 0.7:
                print("Removing point, because to close to another: ", point)
                continue

            # # #check that the point is not too close to black pixels
            # # if self.is_in_proximity_of_black_pixel(point,0.7):
            # #     continue
            filtered_branch_points.append(point)
            print("Adding prospect point: ", point)

        return filtered_branch_points

    def has_clear_path(self, point1, point2):
        """
        Check if there is a clear path between two points.

        The points are in world coordinates.
        """

        # world to map coordinates
        point1 = self.world_to_map_coords(point1[0], point1[1])
        point2 = self.world_to_map_coords(point2[0], point2[1])

        points = bresenham(point1[0], point1[1], point2[0], point2[1])

        for point in points:
            pixel_value = self.map[point[0], point[1]]
            if pixel_value == 0:
                return False

        return True

    def is_in_proximity_of_black_pixel(self, point, proximity=10):
        """
        Check if a point is in proximity of a black pixel.

        The proximity is in meters.

        """

        proximity = int(proximity / self.map_resolution)
        point_x, point_y = point
        rows, cols = self.map.shape

        for d_x in range(-proximity, proximity + 1):
            for d_y in range(-proximity, proximity + 1):
                if d_x == 0 and d_y == 0:
                    continue

                new_x, new_y = point_x + d_x, point_y + d_y

                if 0 <= new_x < rows and 0 <= new_y < cols:
                    pixel_value = self.map[new_x, new_y]

                    if pixel_value == 0:
                        return True

        return False

    def is_close_to_other(self, point, other_points, distance_threshold=0.7):
        """
        Check if a point is close to any other point.

        The distance threshold is in meters.

        """

        distance_threshold = int(distance_threshold / self.map_resolution)

        for other_point in other_points:
            distance = np.linalg.norm(np.array(point) - np.array(other_point))
            if distance <= distance_threshold and self.has_clear_path(point, other_point):
                return True

        return False

    def visualize(self, skeleton_overlay, branch_points, accessible_costmap):
        """
        Visualize the map, skeleton overlay and branch points in rviz.
        """
        # create a blank image
        img = np.zeros((self.size_y, self.size_x), np.uint8)
        # add each with a certain weight
        img = cv2.addWeighted(accessible_costmap, 0.5, skeleton_overlay, 0.38, 0)

        # conver branch points (x,y) to image where x y is white

        bp_img = np.zeros((self.size_y, self.size_x), np.uint8)
        for point in branch_points:
            bp_img[point[1], point[0]] = 255

        img = cv2.addWeighted(img, 0.5, bp_img, 0.77, 0)
        # convert to ros image

        # for each point in branch points add a circle around it

        circles = np.zeros((self.size_y, self.size_x), np.uint8)
        size = 1.6  # meters
        size = int(size / self.map_resolution)
        for point in branch_points:
            cv2.circle(circles, (point[0], point[1]), size, (255, 255, 255), -1)

        img = cv2.addWeighted(img, 0.5, circles, 0.13, 0)

        # flip image
        img = cv2.flip(img, 0)

        # zoom into center for better visualization
        zoomed = img[160 + 20 : 325, 160 + 20 : 325]
        img = zoomed

        # save image
        # cv2.imwrite("map.png", img)

        # plot
        plt.imshow(img, cmap="gray")
        plt.show()

        # wait for key press to continue
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        ros_img = self.bridge.cv2_to_imgmsg(img, "rgb8")
        # publish every second
        timer = threading.Timer(1, self.image_pub.publish, [ros_img])
        timer.start()

    def cost_map_callback(self, map_data) -> None:
        """
        Initializes the cost map and accessible cost map (only accessible positions).
        """

        with self.cost_map_lock:  # Acquire the lock before modifying the map attribute
            size_x = map_data.info.width
            size_y = map_data.info.height

            rospy.loginfo(f"CostMap size: x: {str(size_x)}, y: {str(size_y)}.")

            if size_x < 3 or size_y < 3:
                rospy.loginfo(
                    f"CostMap size only: x: {str(size_x)}, y: {str(size_y)}. NOT running CostMap to"
                    " image conversion."
                )
                return

            cost_map_resolution = map_data.info.resolution
            rospy.loginfo(f"cost_map resolution: {str(cost_map_resolution)}")

            self.cost_map = np.array(map_data.data).reshape((size_y, size_x)).astype(np.uint8)

            # get correct numbers
            self.cost_map[self.cost_map == -1] = 127
            self.cost_map[self.cost_map == 0] = 255
            self.cost_map[self.cost_map == 100] = 0

            # remember only accessible positions
            self.accessible_costmap = np.copy(self.cost_map)

            # threshold_available_map_point = 60
            # self.accessible_costmap[
            #     self.accessible_costmap > threshold_available_map_point
            # ] = 0
            # self.accessible_costmap[self.accessible_costmap > 0] = 255

            # erode accessible_costmap to make sure we get more central reachable points
            self.accessible_costmap = np.uint8(self.accessible_costmap)
            # kernel = np.ones((3, 3), np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            self.accessible_costmap = cv2.erode(self.accessible_costmap, kernel)

            # cv2.imshow("cost_map", self.cost_map)

            # cv2.imshow("accessible_costmap", self.accessible_costmap)
            # cv2.waitKey(0)

            self.cost_map_ready = True

            # save cost map
            # cv2.imwrite("cost_map.png", self.cost_map)
            # cv2.imwrite("accessible_costmap.png", self.accessible_costmap)

    def get_map(self) -> np.ndarray:
        """
        Get the current map.

        Returns:
            numpy.ndarray: The current map
        """
        with self.map_lock:  # Acquire the lock before accessing the map attribute
            return self.map

    def is_ready(self) -> bool:
        """
        Check if the goals are ready.

        Returns:
            bool: True if the goals are ready, False otherwise.
        """

        # both cost map and map must be ready
        with self.cost_map_lock:
            if not self.cost_map_ready:
                print("cost map not ready")
                return False

        with self.map_lock:
            print(f"goals ready: {self.goals_ready}")
            return self.goals_ready

    def get_goals(self) -> List[Tuple[float, float]]:
        """Get the list of goals

        Returns:
            list: List of goals.
        """
        with self.map_lock:
            return self.goal_points

    def get_robot_position(self) -> Tuple[float, float]:
        """
        Get the current robot position.

        Returns:
            Tuple[float, float]: The current robot position (x, y).
        """
        rospy.loginfo("Waiting for amcl_pose...")
        pose_msg = rospy.wait_for_message("/amcl_pose", PoseWithCovarianceStamped, timeout=5.0)
        rospy.loginfo("Received amcl_pose.")
        robot_position = (pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y)
        return robot_position

    def get_robot_position_pose(self) -> PoseWithCovarianceStamped:
        """
        Get the current robot position with pose.

        Returns:
            Tuple[float, float]: The current robot position (x, y).
        """
        rospy.loginfo("Waiting for amcl_pose...")
        pose_msg = rospy.wait_for_message("/amcl_pose", PoseWithCovarianceStamped, timeout=5.0)
        rospy.loginfo("Received amcl_pose.")

        return pose_msg.pose

    def init_goals(self) -> None:
        """
        Transforms branch points to world coordinates and sorts them by distance to the robot,
        and then sets them as the goals.
        """
        goals = []
        for _, branch_point in enumerate(self.branch_points):
            goals.append(self.map_to_world_coords(branch_point[0], branch_point[1]))

        # Sort goals by distance to the robot
        robot_x, robot_y = self.get_robot_position()

        # Sort goals by distance to the robot
        goals.sort(key=lambda goal: ((goal[0] - robot_x) ** 2 + (goal[1] - robot_y) ** 2))

        self.goal_points = goals
        self.publish_markers_of_goals(goals)
        self.goals_ready = True

    def skeletonize_map(self) -> np.ndarray:
        """
        Skeletonize the map, and return the skeleton overlay.

        Returns:
            np.ndarray: The skeleton overlay.
        """
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

    def map_to_world_coords(self, x_map: float, y_map: float) -> Tuple[float, float]:
        """
        Transform map coordinates to world coordinates.

        Args:
            x_map (float): The x-coordinate in the map frame.
            y_map (float): The y-coordinate in the map frame.

        Returns:
            Tuple[float, float]: A tuple containing the x and y coordinates in the world frame.
        """
        point = PointStamped()
        point.point.x = x_map * self.map_resolution
        point.point.y = y_map * self.map_resolution
        point.point.z = 0.0

        transformed_pt = tf2_geometry_msgs.do_transform_point(point, self.map_transform)

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
        # You can decrease the k value and/or increase blockSize to get more corners
        harris_corners = cv2.cornerHarris(skeleton_copy, blockSize=3, ksize=3, k=0.025)

        # Dilate the result to make the corners more visible
        harris_corners_dilated = cv2.dilate(harris_corners, None)

        # save the dilated image
        # cv2.imwrite("harris_corners_dilated.png", harris_corners_dilated)

        # Apply thresholding to identify the optimal corners
        # Decrease the threshold factor to detect more corners (e.g., from 0.32 to 0.2)
        threshold_value = 0.12 * harris_corners_dilated.max()
        threshold_image = cv2.threshold(harris_corners_dilated, threshold_value, 255, 0)[1]
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
        branch_points = [(int(corner[0]), int(corner[1])) for corner in refined_corners[1:]]
        return branch_points

    def visualize_branch_points(self) -> None:
        """
        Visualize the branch points in the map.
        """
        bp_map = np.zeros(self.map.shape, dtype=np.uint8)

        for _, branch_point in enumerate(self.branch_points):
            bp_map[branch_point[1]][branch_point[0]] = 255

        # overlayed_bp = cv2.addWeighted(self.map, 0.5, bp_map, 0.5, 0)
        # cv2.imshow("overlayed branch points", overlayed_bp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def get_face_greet_location_candidates_perpendicular(
        self, x_ce, y_ce, fpose_left, fpose_right, dist=30
    ):
        """
        Get the face greet location candidates perpendicular to the line between the two faces.
        """

        with self.cost_map_lock:  # lock the cost map
            x_left = fpose_left.position.x
            y_left = fpose_left.position.y
            x_right = fpose_right.position.x
            y_right = fpose_right.position.y

            d_x = x_right - x_left
            d_y = y_right - y_left

            perp_dx = -d_y / ((d_y * d_y + d_x * d_x) ** 0.5)
            perp_dy = d_x / ((d_y * d_y + d_x * d_x) ** 0.5)

            x_start = x_ce
            y_start = y_ce
            x_finish = round(-perp_dx * dist + x_ce)
            y_finish = round(-perp_dy * dist + y_ce)

            candidates = list(bresenham(x_start, y_start, x_finish, y_finish))
            candidates_reachable = []
            for candidate in candidates:
                col = candidate[0]
                row = candidate[1]
                if self.in_map_bounds(col, row) and self.can_move_to(col, row):
                    candidates_reachable.append(candidate)
                    break

            if len(candidates_reachable) == 0:
                rospy.loginfo("Searching for backup candidates")
                backup_candidate = candidates[3]
                x_close, y_close = self.nearest_nonzero_to_point(
                    self.accessible_costmap, backup_candidate[0], backup_candidate[1]
                )
                candidates_reachable.append((x_close, y_close))
                rospy.loginfo(candidates_reachable)

            return candidates_reachable

    def can_move_to(self, x_coord, y_coord) -> bool:
        """
        Check if the robot can move to a given cell.
        """
        cost = self.map_coord_cost(x_coord, y_coord)

        if cost == UNKNOWN:
            # Unknown cell
            return False
        if cost == INFLATED_OBSTACLE:
            # Not far enough from obstacle
            return False
        if cost == LETHAL_OBSTACLE:
            # Lethal obstacle (e.g., wall)
            return False
        if cost == FREE_SPACE:
            # Free space
            return True
        if cost > WALL_THRESHOLD:
            # Wall or close to wall
            return False

        #rospy.logwarn(f"Unknown cost value: {cost}")
        # You can choose to treat unknown values as obstacles or not:
        return False  # Treat unknown cost values as obstacles
        # return True  # Treat unknown cost values as free space

    def in_map_bounds(self, x_coord: float, y_coord: float) -> bool:
        """
        Check if a given cell is within the map bounds.

        Args:
            x_coord (float): First coordinate of the cell.
            y_coord (float): Second coordinate of the cell.

        Returns:
            bool: True if the cell is within the map bounds, False otherwise.
        """
        if (0 <= x_coord < self.size_x) and (0 <= y_coord < self.size_y):
            return True
        return False

    def nearest_nonzero_to_point(
        self, accessible_costmap: np.ndarray, x_coord: float, y_coord: float
    ) -> Tuple[int, int]:
        """
        Return indices of nonzero element closest to point (x,y) in array a
        """
        row, col = np.nonzero(accessible_costmap)
        min_idx = ((row - y_coord) ** 2 + (col - x_coord) ** 2).argmin()
        return col[min_idx], row[min_idx]

    def get_face_greet_location(self, x_c, y_c, x_r, y_r, fpose_left, fpose_right):
        """
        Get the face greet location.
        """
        #
        (x_c, y_c) = self.world_to_map_coords(x_c, y_c)
        (x_r, y_r) = self.world_to_map_coords(x_r, y_r)

        candidates = self.get_face_greet_location_candidates_perpendicular(
            x_c, y_c, fpose_left, fpose_right
        )

        res_point = None

        res_point = candidates[0]

        return self.map_to_world_coords(res_point[0], res_point[1])

    def map_coord_cost(self, x_coord: float, y_coord: float) -> int:
        """
        Get the cost of a given cell.

        Args:
            x_coord (float): x coordinate of the cell.
            y_coord (float): y coordinate of the cell.

        Returns:
            int: The cost of the cell.
        """
        if not self.in_map_bounds(x_coord, y_coord):
            rospy.logerr("Invalid map coordinates.")
            return None
        return self.accessible_costmap[y_coord][x_coord]

    def get_inverse_transform(self) -> TransformStamped:
        """
        Get the inverse transform of the map transform.

        Returns:
            TransformedStamped: The inverse transform.
        """
        # https://answers.ros.org/question/229329/what-is-the-right-way-to-inverse-a-transform-in-python/
        # https://www.programcreek.com/python/example/96799/tf.transformations
        # http://docs.ros.org/en/jade/api/tf/html/python/transformations.html
        transform_tmp = concatenate_matrices(
            translation_matrix(
                np.array(
                    [
                        self.map_transform.transform.translation.x,
                        self.map_transform.transform.translation.y,
                        self.map_transform.transform.translation.z,
                    ]
                )
            ),
            quaternion_matrix(
                np.array(
                    [
                        self.map_transform.transform.rotation.x,
                        self.map_transform.transform.rotation.y,
                        self.map_transform.transform.rotation.z,
                    ]
                )
            ),
        )
        inverse_transform = inverse_matrix(transform_tmp)
        translation = translation_from_matrix(inverse_transform)
        rotation = quaternion_from_matrix(inverse_transform)

        res = TransformStamped()
        res.transform.translation.x = translation[0]
        res.transform.translation.y = translation[1]
        res.transform.translation.z = translation[2]
        res.transform.rotation.x = rotation[0]
        res.transform.rotation.y = rotation[1]
        res.transform.rotation.z = rotation[2]
        res.transform.rotation.w = rotation[3]
        return res

    def world_to_map_coords(self, x_world: float, y_world: float) -> Tuple[int, int]:
        """
        Convert world coordinates to map coordinates.

        Args:
            x_world (float): world x coordinate
            y_world (float): world y coordinate

        Returns:
            Tuple[int, int]: map coordinate pair (x, y)
        """
        inverse_transform = self.get_inverse_transform()

        point = PointStamped()
        point.point.x = x_world
        point.point.y = y_world
        point.point.z = 0.0

        transformed_pt = tf2_geometry_msgs.do_transform_point(point, inverse_transform)

        x_res = round(transformed_pt.point.x / self.map_resolution)
        y_res = round(transformed_pt.point.y / self.map_resolution)

        return (x_res, y_res)

    def quaternion_from_points(self, x_1: float, y_1: float, x_2: float, y_2: float) -> Quaternion:
        """
        Returns quaternion representing rotation so that the
        robot will be pointing prom (x1,y1)to (x2,y2)

        Args:
            x1 (float): x coordinate of first point
            y1 (float): y coordinate of first point
            x2 (float): x coordinate of second point
            y2 (float): y coordinate of second point

        Returns:
            Quaternion: quaternion representing rotation
        """
        vector_to_second_point = np.array([x_2, y_2, 0]) - np.array([x_1, y_1, 0])
        base_vector = [1, 0, 0]

        yaw = np.arctan2(vector_to_second_point[1], vector_to_second_point[0]) - np.arctan2(
            base_vector[1], base_vector[0]
        )

        quaternion = quaternion_from_euler(0, 0, yaw)

        return quaternion

    def get_nearest_accessible_point(self, point_x: float, point_y: float) -> Tuple[float, float]:
        """
        Returns the indices of the accessible point closest to the point (x, y) in the costmap.

        Args:
            x (float): X-coordinate of the point.
            y (float): Y-coordinate of the point.

        Returns:
            Tuple[float, float]: The indices of the accessible point closest to the input point.
        """
        (c_x, c_y) = self.world_to_map_coords(point_x, point_y)

        x_close, y_close = self.nearest_nonzero_to_point(self.accessible_costmap, c_x, c_y)

        (x_transformed, y_transformed) = self.map_to_world_coords(x_close, y_close)
        return x_transformed, y_transformed

    def get_nearest_accessible_point_with_erosion(
        self, x_coord: float, y_coord: float, erosion_iter: int
    ) -> Tuple[float, float]:
        """
        Returns the indices of the accessible point closest to the point (x, y) in the costmap.
        with erosion

        Args:
            x (float): X-coordinate of the point.
            y (float): Y-coordinate of the point.

        Returns:
            Tuple[float, float]: The indices of the accessible point closest to the input point.
        """
        (c_x, c_y) = self.world_to_map_coords(x_coord, y_coord)

        accessible_map_copy = np.copy(self.accessible_costmap)
        kernel = np.ones((3, 3), np.uint8)
        accessible_map_copy = cv2.erode(accessible_map_copy, kernel, iterations=erosion_iter)

        # save_image(accessible_map_copy, "accessible_map_copy.png")

        debugimg = np.zeros_like(accessible_map_copy)
        # keep only non-zero values
        debugimg[accessible_map_copy > 0] = 255

        # cv2.imwrite("acemap_debug.png", debugimg)

        x_close, y_close = self.nearest_nonzero_to_point(accessible_map_copy, c_x, c_y)

        (x_transformed, y_transformed) = self.map_to_world_coords(x_close, y_close)
        return x_transformed, y_transformed

    def get_object_greet_pose(self, x_obj: float, y_obj: float) -> Pose:
        """
        Returns pose with proper greet location and orientation
        for ring / cylinder at x_obj, y_obj.
        """
        x_greet, y_greet = self.get_nearest_accessible_point(x_obj, y_obj)
        q_dest = self.quaternion_from_points(x_greet, y_greet, x_obj, y_obj)

        # create pose for greet
        pose = Pose()
        pose.position.x = x_greet
        pose.position.y = y_greet
        pose.position.z = 0
        pose.orientation.x = q_dest[0]
        pose.orientation.y = q_dest[1]
        pose.orientation.z = q_dest[2]
        pose.orientation.w = q_dest[3]
        return pose

    def euclidean_distance(self, x_1: float, y_1: float, x_2: float, y_2: float) -> float:
        """
        Returns euclidean distance between two points.

        Args:
            x1 (float): x coordinate of first point
            y1 (float): y coordinate of first point
            x2 (float): x coordinate of second point
            y2 (float): y coordinate of second point

        Returns:
            float: euclidean distance between two points
        """
        return np.linalg.norm(np.array([x_1, y_1]) - np.array([x_2, y_2]))


def test():
    """
    Test function for MapManager class.
    """
    rospy.init_node("path_setter", anonymous=True)
    map_manager = MapManager()

    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        if map_manager.is_ready():
            goals = map_manager.get_goals()
            if goals is not None and len(goals) > 0:
                map_manager.publish_markers_of_goals(goals)
            rate.sleep()


if __name__ == "__main__":
    test()
