#!/usr/bin/python3
import sys
import cv2
import numpy as np
import rospy
import move_base
from skimage.morphology import skeletonize
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from move_base_msgs.msg import MoveBaseGoal
from geometry_msgs.msg import (
    Vector3,
    Pose,
    Quaternion,
    Point,
    TransformStamped,
    Point,
)
from move_base_client import MoveBaseClient

cv_map = np.zeros((1, 1), dtype=np.uint8)
map_resolution = 0.0
map_transform = TransformStamped()
marker_pub = None
markers = None
waypoints = None


def publish_markers(waypoints):
    """Publishes a MarkerArray of waypoints to the /waypoint_markers topic

    Args:
        waypoints (list): List of waypoints
    """
    global marker_pub

    markers = MarkerArray()  # Create a Marker for each waypoint
    for i, waypoint in enumerate(waypoints):
        marker = Marker()
        marker.id = i
        marker.pose.position = Point(waypoint["x"], waypoint["y"], 0)
        marker.header.stamp = rospy.Time(0)
        marker.header.frame_id = "map"
        marker.pose.orientation = Quaternion(0.5, 0.5, 0.5, 0.5)
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False
        marker.lifetime = rospy.Duration.from_sec(777)
        marker.scale = Vector3(0.1, 0.1, 0.1)
        marker.color = ColorRGBA(1.0, 0.5, 0.0, 1.0)  # Orange color
        markers.markers.append(marker)
        print("marker", Point(waypoint["x"], waypoint["y"], 0))

    marker_pub.publish(markers)


def get_map(msg_map):
    """Gets a map from the map topic and converts it to an OpenCV image

    Args:
        msg_map (_type_): _description_
    """
    global cv_map, map_resolution, map_transform, markers, waypoints
    size_x = msg_map.info.width
    size_y = msg_map.info.height

    if size_x < 3 or size_y < 3:
        rospy.loginfo(
            "Map size is only x: %d, y: %d. Not running map to image conversion",
            size_x,
            size_y,
        )
        return

    if cv_map.shape != (size_y, size_x):
        cv_map = np.zeros((size_y, size_x), dtype=np.uint8)

    map_resolution = msg_map.info.resolution
    map_transform.transform.translation.x = msg_map.info.origin.position.x
    map_transform.transform.translation.y = msg_map.info.origin.position.y
    map_transform.transform.translation.z = msg_map.info.origin.position.z
    map_transform.transform.rotation = msg_map.info.origin.orientation
    map_msg_data = np.array(msg_map.data, dtype=np.int8).reshape((size_y, size_x))

    size_y_rev = size_y - 1

    for y in range(size_y_rev, -1, -1):
        idx_map_y = size_x * (size_y - y)
        idx_img_y = size_x * y
        for x in range(size_x):
            idx = idx_img_y + x
            value = map_msg_data[size_y - y - 1, x]
            if value == -1:
                cv_map[y, x] = 127
            elif value == 0:
                cv_map[y, x] = 255
            elif value == 100:
                cv_map[y, x] = 0

    # Make walls wider
    kernel = np.ones((3, 3), np.uint8)
    cv_map = cv2.erode(cv_map, kernel, iterations=3)

    # Extract the skeleton of the map
    skeleton = extract_skeleton(cv_map)
    skeleton_image = np.zeros_like(skeleton, dtype=np.uint8)
    skeleton_image[skeleton] = 255

    # Create a skeleton_overlay where cv_map == 255
    skeleton_overlay = np.zeros_like(cv_map)
    skeleton_overlay[cv_map == 255] = skeleton_image[cv_map == 255]

    waypoints_map = np.zeros_like(cv_map)
    bp = find_branch_points(skeleton_overlay)
    waypoints = []
    for vertex in bp:
        waypoints_map[vertex[1], vertex[0]] = 255
        mx = vertex[0] * map_resolution + map_transform.transform.translation.x
        my = (
            cv_map.shape[0] - vertex[1]
        ) * map_resolution + map_transform.transform.translation.y
        waypoints.append({"x": mx, "y": my, "z": 0.0})

    publish_markers(waypoints)
    execute_plan()


def execute_plan():
    global waypoints
    xy_marks = []
    for mark in waypoints:
        xy_marks.append((mark["x"], mark["y"]))
    path = nearest_neighbor(xy_marks, xy_marks[0])

    marks_path = []
    for point in path:
        marks_path.append({"x": point[0], "y": point[1], "z": 0.0})
    path_plan(marks_path)


def euclidean_distance(point1, point2):
    return np.sqrt((point1["x"] - point2["x"]) ** 2 + (point1["y"] - point2["y"]) ** 2)


def euclidean_distance2(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def nearest_neighbor(vertices, start_vertex):
    unvisited_vertices = set(vertices)
    unvisited_vertices.remove(start_vertex)
    current_vertex = start_vertex
    path = [start_vertex]

    while unvisited_vertices:
        nearest_vertex = None
        nearest_distance = sys.float_info.max

        for vertex in unvisited_vertices:
            distance = euclidean_distance2(current_vertex, vertex)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_vertex = vertex

        path.append(nearest_vertex)
        unvisited_vertices.remove(nearest_vertex)
        current_vertex = nearest_vertex

    return path


def find_branch_points(skeleton_image):
    sc = skeleton_image.copy()
    dst = cv2.cornerHarris(sc, 9, 5, 0.04)
    dst = cv2.dilate(dst, None)  # result is dilated for marking the corners

    # Threshold for an optimal value, it may vary depending on the image.
    img_thresh = cv2.threshold(dst, 0.32 * dst.max(), 255, 0)[1]
    img_thresh = np.uint8(img_thresh)

    # get the matrix with the x and y locations of each centroid
    centroids = cv2.connectedComponentsWithStats(img_thresh)[3]

    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # refine corner coordinates to subpixel accuracy
    corners = cv2.cornerSubPix(
        sc, np.float32(centroids), (5, 5), (-1, -1), stop_criteria
    )

    toret = []
    for i in range(1, len(corners)):
        toret.append((int(corners[i, 0]), int(corners[i, 1])))

    return toret


def path_plan(path):

    default_orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    pose_seq = []

    for i in range(0, len(path)):
        pose = Pose()
        pose.position = Point(path[i]["x"], path[i]["y"], 0.0)
        pose.orientation = default_orientation
        pose_seq.append(pose)

    move_base_client = MoveBaseClient(pose_seq)
    move_base_client.movebase_client()


def navigate_to_waypoint(waypoint):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = waypoint
    move_base.send_goal(goal)
    move_base.wait_for_result()
    result = move_base.get_state()
    return result


def extract_skeleton(map_data: np.ndarray) -> np.ndarray:
    """Extracts a skeleton from a map.

    Args:
        map_data (np.ndarray): A 2D numpy array representing the map.

    Returns:
        np.ndarray: A 2D numpy array representing the skeletonized map.
    """
    skeleton = skeletonize(cv_map)
    return skeleton


def skeleton_to_waypoints(skeleton):
    """Converts a skeleton image of the map to a list of waypoints.

    Args:
        skeleton (ndarray): Skeletonized map

    Returns:
        list: A list of endpoints in the skeleton, represented by (x, y) tuples.
    """
    skeleton2 = skeleton.copy()
    skeleton2 = skeleton2.astype(np.uint8)

    # Find the contours of the skeleton
    contours, _ = cv2.findContours(skeleton2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    vertices = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 3:
            continue
        moment = cv2.moments(contour)
        if moment["m00"] == 0:
            continue
        cx = int(moment["m10"] / moment["m00"])
        cy = int(moment["m01"] / moment["m00"])
        vertices.append((cx, cy))
    return vertices


def main():
    global map_sub, marker_pub
    rospy.init_node("navigate_to_skeleton_waypoints")
    map_sub = rospy.Subscriber("map", OccupancyGrid, get_map)
    marker_pub = rospy.Publisher("waypoint_markers", MarkerArray, queue_size=100)
    rospy.spin()


if __name__ == "__main__":
    main()
