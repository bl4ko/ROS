#!/usr/bin/python3
import math
import cv2
from matplotlib import pyplot as plt
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
import sys

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PointStamped, Vector3, Pose,Quaternion

from visualization_msgs.msg import Marker, MarkerArray
from tf2_geometry_msgs import PointStamped
from tf2_ros import TransformListener
from tf2_geometry_msgs import PointStamped
from tf2_ros import Buffer, TransformListener
import tf2_ros

import rospy
from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion


class MoveBaseClient:
    def __init__(self, pose_seq):
        self.pose_seq = pose_seq
        self.goal_cnt = 0
        self.client = SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to move_base action server")

    def done_cb(self, status, result):
        if status == 2:
            rospy.loginfo(
                "Goal pose "
                + str(self.goal_cnt + 1)
                + " received a cancel request after it started executing, completed execution!"
            )

        if status == 3:
            rospy.loginfo("Goal pose " + str(self.goal_cnt + 1) + " reached")
            self.goal_cnt += 1
            if self.goal_cnt < len(self.pose_seq):
                self.send_next_goal()

    def active_cb(self):
        rospy.loginfo(
            "Goal pose "
            + str(self.goal_cnt + 1)
            + " is now being processed by the Action Server"
        )

    def feedback_cb(self, feedback):
        pass

    def send_next_goal(self):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = self.pose_seq[self.goal_cnt]
        rospy.loginfo(
            "Sending goal pose " + str(self.goal_cnt + 1) + " to Action Server"
        )
        rospy.loginfo(str(self.pose_seq[self.goal_cnt]))
        self.client.send_goal(
            goal,
            done_cb=self.done_cb,
            active_cb=self.active_cb,
            feedback_cb=self.feedback_cb,
        )

    def movebase_client(self):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = self.pose_seq[self.goal_cnt]

        rospy.loginfo(
            "Sending goal pose " + str(self.goal_cnt + 1) + " to Action Server"
        )
        rospy.loginfo(str(self.pose_seq[self.goal_cnt]))

        # Send the goal and wait for the result
        status = self.client.send_goal_and_wait(
            goal, rospy.Duration(120), rospy.Duration(20)
        )

        # Reference for terminal status values: http://docs.ros.org/diamondback/api/actionlib_msgs/html/msg/GoalStatus.html
        if status == 2:
            rospy.loginfo(
                "Goal pose "
                + str(self.goal_cnt + 1)
                + " received a cancel request after it started executing, completed execution!"
            )

        if status == 3:
            rospy.loginfo("Goal pose " + str(self.goal_cnt + 1) + " reached")
            self.check_goal_count()
        else:
            rospy.logwarn("Failed to reach goal pose " + str(self.goal_cnt + 1))
            rospy.loginfo("Moving to the next goal pose")
            self.check_goal_count()

    def check_goal_count(self):
        self.goal_cnt += 1
        if self.goal_cnt < len(self.pose_seq):
            self.movebase_client()
        else:
            rospy.loginfo("All goals reached!")
            rospy.signal_shutdown("My work is done, time to go home!")


cv_map = np.zeros((1, 1), dtype=np.uint8)
map_resolution = 0.0
map_transform = TransformStamped()

marker_pub = None

def publish_markers(waypoints):
    # Create a MarkerPublisher for publishing markers

    global marker_pub

    # Create a Marker array 
    markers = MarkerArray()

    # Create a Marker for each waypoint
    for i, waypoint in enumerate(waypoints):
        marker = Marker()
        marker.id = i
        marker.pose.position = Point(waypoint['x'], waypoint['y'], 0)
        marker.header.stamp = rospy.Time(0)
        marker.header.frame_id = 'map'
        marker.pose.orientation = Quaternion(0.5, 0.5, 0.5, 0.5)
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False
        marker.lifetime = rospy.Duration.from_sec(777)
        marker.scale = Vector3(0.1, 0.1, 0.1)
        #orange color
        marker.color = ColorRGBA(1.0, 0.5, 0.0, 1.0)

        markers.markers.append(marker)

        print("marker",Point(waypoint['x'], waypoint['y'], 0))

    # Publish the Marker array
    marker_pub.publish(markers)

    
   

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
    global map_transform
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



    #make walls thinner
    kernel = np.ones((3,3),np.uint8)
    cv_map = cv2.erode(cv_map,kernel,iterations = 3)

    #show map
    #cv2.imshow("Map", cv_map)

    #save map
    #cv2.imwrite("map_ok.png", cv_map)


    


    # if map_data is not None:
    skeleton = extract_skeleton(cv_map)
    print("skeleton",skeleton.shape)
    plt.imsave("skeleton_OOOOOOO.png", skeleton, cmap='gray')

    # waypoints = skeleton_to_waypoints(skeleton, map_data)

    #save map
    

    # Create an image of the skeleton map
    skeleton_image = np.zeros_like(skeleton, dtype=np.uint8)
    skeleton_image[skeleton] = 255

    #save skeleton
    #cv2.imwrite("skeleton_ok.png", skeleton_image.reshape(skeleton_image.shape[0], skeleton_image.shape[1], 1))

    final = skeleton_image[cv_map == 255]
    final_resized = cv2.resize(final, cv_map.shape[::-1], interpolation=cv2.INTER_NEAREST)
    final = final_resized

    # Create a new image from the skeleton where cv_map == 255
    new_image = np.zeros_like(cv_map)
    new_image[cv_map == 255] = skeleton_image[cv_map == 255]

    # Overlay the new image over the original map
    overlay = np.where(cv_map == 255, new_image, cv_map)

    waypoints = skeleton_to_waypoints(new_image)
    img = np.zeros_like(cv_map)
    # for vertex in waypoints:
    #     img[vertex[1], vertex[0]] = 255


    

    #publish_markers(skeleton_to_waypoints2(skeleton, msg_map))
    cv2.imshow("Skeleton", img)
    # Display the images
    cv2.imshow("Overlay", overlay)

    cv2.imshow("Skeleton image", new_image)


    bp = find_branch_points(new_image)

    print("branch points", bp)


    marks=[]

    for vertex in bp:
        img[vertex[1], vertex[0]] = 255
        mx = vertex[0] * map_resolution + map_transform.transform.translation.x
        my = (cv_map.shape[0] - vertex[1]) * map_resolution + map_transform.transform.translation.y

        marks.append({"x": mx, "y": my, "z": 0.0})

    print("marks", marks)


    publish_markers(marks)

    #sort marks based on distance from start
    starting_point = {"x": 0.0, "y": 0.0}  # Replace with your actual starting point

# Sort the marks based on their distance from the starting point
    #marks = sorted(marks, key=lambda mark: euclidean_distance(starting_point, mark))

# Create a path using the sorted marks
    #path = [{"x": mark["x"], "y": mark["y"], "z": mark["z"]} for mark in sorted_marks]

    #marks to (x,y)
    xy_marks = []
    for mark in marks:
        xy_marks.append((mark["x"], mark["y"]))



    path = nearest_neighbor(xy_marks, xy_marks[0])

    #(x,y) to marks
    marks_path = []
    for point in path:
        marks_path.append({"x": point[0], "y": point[1], "z": 0.0})


    #publish the path
    path_plan(marks_path)

    cv2.imwrite("pts.png", img)
    cv2.imwrite("skeleton image.png", new_image) 
    cv2.imwrite("overlay.png", overlay)
    




    cv2.waitKey(0)
    cv2.destroyAllWindows()


def euclidean_distance(point1, point2):
    return np.sqrt((point1["x"] - point2["x"])**2 + (point1["y"] - point2["y"])**2)

def euclidean_distance2(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
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
    sc=skeleton_image.copy()

    #save skeleton
    cv2.imwrite("skeleton_oldok.png", sc)

    #0-1
    dst = cv2.cornerHarris(sc, 9, 5, 0.04)
    # result is dilated for marking the corners
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img_thresh = cv2.threshold(dst, 0.32*dst.max(), 255, 0)[1]
    img_thresh = np.uint8(img_thresh)

    # get the matrix with the x and y locations of each centroid
    centroids = cv2.connectedComponentsWithStats(img_thresh)[3]

    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# refine corner coordinates to subpixel accuracy
    corners = cv2.cornerSubPix(sc, np.float32(centroids), (5,5), (-1,-1), stop_criteria)

    toret=[]
    for i in range(1, len(corners)):
        toret.append((int(corners[i,0]), int(corners[i,1])))
    
    return toret

def path_plan(path):

    default_orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    pose_seq = []

    for i in range(0, len(path)):
        pose = Pose()
        pose.position = Point(path[i]['x'], path[i]['y'], 0.0)
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
        mx = cx * map_resolution + map_origin.x
        my = cy * map_resolution + map_origin.y


        vertices.append({"x": mx, "y": my, "z": 0.0})
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
    global map_sub, marker_pub
    rospy.init_node("navigate_to_skeleton_waypoints")
    rospack = rospkg.RosPack()
    map_sub = rospy.Subscriber("map", OccupancyGrid, get_map)
    marker_pub = rospy.Publisher("waypoint_markers", MarkerArray, queue_size=100)

    #publish test markers
    marker_array = MarkerArray()
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "waypoints"
    marker.id = 0
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1

    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    marker.lifetime = rospy.Duration(0)
    marker_array.markers.append(marker)
    marker_pub.publish(marker_array)



    rospy.spin()


if __name__ == "__main__":
    main()
