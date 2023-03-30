#!/usr/bin/python3

import math
import threading
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
import tf2_geometry_msgs

import rospy
from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion

from map_msgs.msg import OccupancyGridUpdate



#this file gets the map from the map_server and saves it in variable map
#it also uses skeletonize to get points to visit in the map
#it also publishes the points to visit as markers in rviz


class MapManager:
    def __init__(self):
        self.map = None
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.map_lock = threading.Lock()  # Add a lock for the map attribute

        self.marker_pub = rospy.Publisher("MapManager_markers", MarkerArray, queue_size=100)
        
        self.goals_ready = False
        self.goals = []

        self.map_transform = TransformStamped()
        self.map_resolution = None

        self.size_x = None
        self.size_y = None
        self.map_reference_frame = None




    
    def map_callback(self, data):
         with self.map_lock:  # Acquire the lock before modifying the map attribute
            self.map_processing(data)


    def get_map(self):
        with self.map_lock:  # Acquire the lock before accessing the map attribute
            return self.map
    

    def get_goals(self):
        with self.map_lock:
            return self.goals
        
    


    def map_processing(self,map_data):

        size_x = map_data.info.width
        size_y = map_data.info.height

        self.size_x = size_x
        self.size_y = size_y

        rospy.loginfo("Map size: x: %s, y: %s." % (str(size_x), str(size_y)))

        if size_x < 3 or size_y < 3:
            rospy.loginfo("Map size only: x: %s, y: %s. NOT running map to image conversion." % (str(size_x), str(size_y)))
            return
        
        self.map_resolution = map_data.info.resolution
        rospy.loginfo("Map resolution: %s" % str(self.map_resolution))

        self.map_reference_frame = map_data.header.frame_id

        self.map_transform.transform.translation.x = map_data.info.origin.position.x
        self.map_transform.transform.translation.y = map_data.info.origin.position.y
        self.map_transform.transform.translation.z = map_data.info.origin.position.z
        self.map_transform.transform.rotation = map_data.info.origin.orientation

        # deal with map
        #self.map = np.array(map_data.data, dtype = np.int8).reshape((size_y, size_x))
        self.map = np.array(map_data.data).reshape((size_y, size_x))

        # flip on the rows to get correct image (flip along y axis)
        # is it neccesary in our case - doe not seem to be because we will not transform y later at conversion
        # self.map = np.flip(self.map, 0)

        # get correct numbers
        self.map[self.map == -1] = 127
        self.map[self.map == 0] = 255
        self.map[self.map == 100] = 0

        # convert to uint8
        self.map = self.map.astype(np.uint8)

        kernel = np.ones((3,3),np.uint8)
        semi_safe_map = cv2.erode(self.map,kernel,iterations = 3)

        #make it safe to drive on 
        self.map = semi_safe_map

        #magic code do not touch
        skeleton_ = skeletonize(self.map)
        skeleton_image = np.zeros_like(skeleton_, dtype=np.uint8)
        skeleton_image[skeleton_] = 255

        final = skeleton_image[self.map == 255]
        final_resized = cv2.resize(final, self.map.shape[::-1], interpolation=cv2.INTER_NEAREST)
        final = final_resized

        new_image = np.zeros_like(self.map )
        new_image[self.map  == 255] = skeleton_image[self.map  == 255]

        #find branch points note the map is flipped
        bp = self.find_branch_points(new_image)
        bp_map = np.zeros(self.map.shape, dtype=np.uint8)
        
        # for i in range(len(bp)):
        #     bp_map[bp[i][1]][bp[i][0]] = 255
        
        #visualize_map(bp_map, "branch points")
        
        #overlay branch points on map
        #overlayed = cv2.addWeighted(self.map, 0.5, bp_map, 0.5, 0)
        
        #visualize_map(overlayed, "overlayed")

        goals = []
        for i in range(len(bp)):
            goals.append(self.map_to_world_coords(bp[i][0], bp[i][1]))

        
        self.goals = goals
        self.publish_markers_of_goals(goals)

        self.goals_ready = True

        



        

    
    def publish_markers_of_goals(self, goals):

        marker_array = MarkerArray()

        for i, goal in enumerate(goals):
            marker = Marker()
            marker.id = i
            marker.pose.position = Point(goal[0], goal[1], 0)
            marker.header.stamp = rospy.Time(0)
            marker.header.frame_id = 'map'
            marker.pose.orientation = Quaternion(0.5, 0.5, 0.5, 0.5)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.frame_locked = False
            marker.lifetime = rospy.Duration(2)
            marker.scale = Vector3(0.1, 0.1, 0.1)
            #orange color
            marker.color = ColorRGBA(1.0, 0.5, 0.0, 1.0)

            marker_array.markers.append(marker)
            print("added marker" + str(goal))


        self.marker_pub.publish(marker_array)



        
        





    """
    Returns coordinates transformed from map to world.
    """
    def map_to_world_coords(self, x, y):
        pt = PointStamped()
        pt.point.x = x * self.map_resolution
        pt.point.y = y * self.map_resolution
        pt.point.z = 0.0

        # transform to goal space
        transformed_pt = tf2_geometry_msgs.do_transform_point(pt, self.map_transform)

        return (transformed_pt.point.x, transformed_pt.point.y)



        
    def find_branch_points(self,skeleton_image):
        sc=skeleton_image.copy()
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


def visualize_map(map_data, title="vis"):
    plt.imshow(map_data, cmap='gray', origin='lower')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
        
        
def test():
    rospy.init_node('path_setter', anonymous=True)
    ps = MapManager()

    #run node and wait for map the nshow it in plt
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        map_data = ps.get_map()
        if map_data is not None:
            print("map ready")
            # visualize_map(map_data, "map")
            # #save map
            # cv2.imwrite("map.png", map_data)

        goals = ps.get_goals()
        if goals is not None and len(goals) > 0:
            print("goals ready")
            print(goals)
            ps.publish_markers_of_goals(goals)
        rate.sleep()


if __name__ == '__main__':
    print("start")
    test()

        
        
    