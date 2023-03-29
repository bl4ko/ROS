#!/usr/bin/python3
import cv2
import numpy as np
import yaml
import rospy
import move_base
import rospkg
from nav_msgs.msg import OccupancyGrid
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose
from actionlib_msgs.msg import GoalStatus
from skimage.morphology import skeletonize  # Import Pose
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

cv_map = np.zeros((1, 1), dtype=np.uint8)
map_resolution = 0.0
map_transform = TransformStamped()


def publish_markers(waypoints):
    # Create a MarkerPublisher for publishing markers
    marker_pub = rospy.Publisher("waypoint_markers", Marker, queue_size=10)

    # Create a Marker message for the waypoints
    marker = Marker()
    marker.header.frame_id = "map"
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.lifetime = rospy.Duration()

    # Add the waypoints to the Marker message
    for waypoint in waypoints:
        point = Point()
        point.x = waypoint[0]
        point.y = waypoint[1]
        point.z = 0
        marker.points.append(point)
        print(waypoint)

    # Publish the Marker message
    marker_pub.publish(marker)

def get_map(msg_map):
    size_x = msg_map.info.width
    size_y = msg_map.info.height

    if size_x < 3 or size_y < 3:
        rospy.loginfo("Map size is only x: %d, y: %d. Not running map to image conversion", size_x, size_y)
        return

    global cv_map, map_resolution, map_transform
    if cv_map.shape != (size_y, size_x):
        cv_map = np.zeros((size_y, size_x), dtype=np.uint8)

    map_resolution = msg_map.info.resolution
    map_transform = TransformStamped()
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

    # if map_data is not None:
    skeleton = extract_skeleton(cv_map)
    # waypoints = skeleton_to_waypoints(skeleton, map_data)

    # Create an image of the skeleton map
    skeleton_image = np.zeros_like(skeleton, dtype=np.uint8)
    skeleton_image[skeleton] = 255

    final = skeleton_image[cv_map == 255]
    final_resized = cv2.resize(final, cv_map.shape[::-1], interpolation=cv2.INTER_NEAREST)
    final = final_resized

    # Create a new image from the skeleton where cv_map == 255
    new_image = np.zeros_like(cv_map)
    new_image[cv_map == 255] = skeleton_image[cv_map == 255]

    # Overlay the new image over the original map
    overlay = np.where(cv_map == 255, new_image, cv_map)

    waypoints = skeleton_to_waypoints(skeleton)
    img = np.zeros(cv_map.shape)
    for vertex in waypoints:
        img[vertex[1], vertex[0]] = 255

    publish_markers(skeleton_to_waypoints2(skeleton, msg_map))
    cv2.imshow("Skeleton", img)
    # Display the images
    cv2.imshow("Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def navigate_to_waypoint(waypoint):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = waypoint
    move_base.send_goal(goal)
    move_base.wait_for_result()
    result = move_base.get_state()
    return result


def extract_skeleton(map_data, scale=1):
    # Perform skeletonization
    skeleton = skeletonize(cv_map)
    return skeleton


def skeleton_to_waypoints2(skeleton, map_data):
    skeleton2 = skeleton.copy()
    skeleton2 = skeleton2.astype(np.uint8)
    map_width = map_data.info.width
    map_height = map_data.info.height
    map_resolution = map_data.info.resolution
    map_origin = map_data.info.origin.position

    # Find the contours of the skeleton
    contours, _ = cv2.findContours(skeleton2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Filter the contours based on their area
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
        # Convert to map coordinates
        mx = map_origin.x + cx * map_resolution
        my = map_origin.y + (map_height - cy) * map_resolution
        vertices.append((mx, my))
    return vertices

    
def skeleton_to_waypoints(skeleton):
    # map_width = map_data.info.width
    # map_height = map_data.info.height
    # map_resolution = map_data.info.resolution
    # map_origin = map_data.info.origin.position
    skeleton2 = skeleton.copy()
    skeleton2 = skeleton2.astype(np.uint8)

    # Find the contours of the skeleton
    contours, _ = cv2.findContours(skeleton2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Filter the contours based on their area
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

    # Convert the vertices to waypoints
    # waypoints = []
    # for vertex in vertices:
    #     x, y = vertex
    #     waypoint = Pose()
    #     waypoint.position.x = map_origin.x + x * map_resolution
    #     waypoint.position.y = map_origin.y + y * map_resolution
    #     waypoint.position.z = 0
    #     waypoint.orientation.w = 1
    #     waypoints.append(waypoint)
    # return waypoints

def load_map_from_files(pgm_file, yaml_file):
    with open(yaml_file, "r") as f:
        map_metadata = yaml.safe_load(f)

    # Load the normal map
    map_image = cv2.imread(pgm_file, cv2.IMREAD_GRAYSCALE)

    # Threshold the normal map to create a binary occupancy grid
    threshold = 128
    _, binary_map_image = cv2.threshold(map_image, threshold, 255, cv2.THRESH_BINARY)

    # Convert the binary occupancy grid to an OccupancyGrid message
    occupancy_grid = np.where(binary_map_image == 255, 0, 100)
    occupancy_grid = np.array(occupancy_grid, dtype=np.int8)

    map_data = OccupancyGrid()
    map_data.header.frame_id = map_metadata.get("frame_id", "map")
    map_data.info.resolution = map_metadata["resolution"]
    map_data.info.width = occupancy_grid.shape[1]
    map_data.info.height = occupancy_grid.shape[0]
    map_data.info.origin.position.x = map_metadata["origin"][0]
    map_data.info.origin.position.y = map_metadata["origin"][1]
    map_data.info.origin.position.z = 0
    map_data.info.origin.orientation.w = 1
    map_data.data = occupancy_grid.flatten().tolist()

    return map_data, map_image

def main():
    rospy.init_node("navigate_to_skeleton_waypoints")
    rospack = rospkg.RosPack()
    map_sub = rospy.Subscriber("map", OccupancyGrid, get_map)
    rospy.spin()


if __name__ == "__main__":
    main()
