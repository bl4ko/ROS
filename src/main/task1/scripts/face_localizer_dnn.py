#!/usr/bin/python3
import sys
import threading
import rospy
import cv2
import numpy as np
import tf2_geometry_msgs
import tf2_ros
from os.path import dirname, join
#import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose,Quaternion
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import message_filters
from message_filters import ApproximateTimeSynchronizer
from geometry_msgs.msg import PointStamped, Vector3, Pose,Quaternion

from std_msgs.msg import ColorRGBA


class DetectedFace:
    def __init__(self, face_region, face_distance, depth_time, identity, pose, confidence):
        self.confidence = confidence
        self.face_region = face_region
        self.face_distance = face_distance
        self.depth_time = depth_time
        self.identity = identity
        self.pose = pose

class DetectedFacesTracker:
    
        def __init__(self):
            self.faces = []
            self.grouped_faces_by_distance = [] #list of objects
            self.distance_threshold = 0.5 #in meters

    
        def add_face(self, face):
            
            #check if the face is close to any of the existing faces
            for i in range(len(self.grouped_faces_by_distance)):
                avg_pose = self.grouped_faces_by_distance[i]['avg_pose']
                if self.is_close(avg_pose, face.pose):
                    print('face is close to existing face with id: ', i)
                    self.grouped_faces_by_distance[i]['faces'].append(face)
                    self.grouped_faces_by_distance[i]['avg_pose'] = self.update_avg_pose(self.grouped_faces_by_distance[i]['faces'])
                    self.grouped_faces_by_distance[i]['detections'] += 1
                    return
                
            #if the face is not close to any of the existing faces, add it as a new face
            self.grouped_faces_by_distance.append(
                    {
                        'faces': [face],
                        'avg_pose': face.pose,
                        'detections': 1,
                    }
                )
                
        def update_avg_pose(self, faces):
            #update the average pose of a face
            avg_pose = Pose()
            avg_pose.position.x = np.mean([face.pose.position.x for face in faces])
            avg_pose.position.y = np.mean([face.pose.position.y for face in faces])
            return avg_pose
        
        def is_close(self, pose1, pose2):
            #only check the x and y coordinates
            return np.linalg.norm(np.array([pose1.position.x, pose1.position.y]) - np.array([pose2.position.x, pose2.position.y])) < self.distance_threshold

    
        def get_grouped_faces(self):
            #return the faces grouped by distance
            return self.grouped_faces_by_distance
        
        
            
                
        

        
        



class face_localizer:
    def __init__(self):
        rospy.init_node('face_localizer', anonymous=True)
        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()
        # The function for performin HOG face detection
        #self.face_detector = dlib.get_frontal_face_detector()
        protoPath = join(dirname(__file__), "deploy.prototxt.txt")
        modelPath = join(dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")
        self.face_net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        # A help variable for holding the dimensions of the image
        self.dims = (0, 0, 0)
        # Marker array object used for showing markers in Rviz
        self.marker_array = MarkerArray()
        self.marker_num = 1
        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher('face_markers', MarkerArray, queue_size=1000)
        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        self.image_lock = threading.Lock()
         # Subscribe to the RGB and depth image topics
        self.rgb_image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
        self.depth_image_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
        # Synchronize the two image topics with a specified tolerance (e.g., 0.5 seconds)
        self.time_synchronizer = ApproximateTimeSynchronizer([self.rgb_image_sub, self.depth_image_sub], queue_size=5, slop=0.5)
        self.time_synchronizer.registerCallback(self.image_callback)
        self.latest_rgb_image_msg = None
        self.latest_depth_image_msg = None


        self.detected_faces_tracker = DetectedFacesTracker()



    def image_callback(self, rgb_image_msg, depth_image_msg):
        # Store the latest synchronized images
        with self.image_lock:
            self.latest_rgb_image_msg = rgb_image_msg
            self.latest_depth_image_msg = depth_image_msg
    def get_pose(self,coords,dist,stamp):
        # Calculate the position of the detected face
        k_f = 554 # kinect focal length in pixels
        x1, x2, y1, y2 = coords
        face_x = self.dims[1] / 2 - (x1+x2)/2.
        face_y = self.dims[0] / 2 - (y1+y2)/2.
        angle_to_target = np.arctan2(face_x,k_f)
        # Get the angles in the base_link relative coordinate system
        x, y = dist*np.cos(angle_to_target), dist*np.sin(angle_to_target)
        ### Define a stamped message for transformation - directly in "base_link"
        #point_s = PointStamped()
        #point_s.point.x = x
        #point_s.point.y = y
        #point_s.point.z = 0.3
        #point_s.header.frame_id = "base_link"
        #point_s.header.stamp = rospy.Time(0)
        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped()
        point_s.point.x = -y
        point_s.point.y = 0
        point_s.point.z = x
        point_s.header.frame_id = "camera_rgb_optical_frame"
        point_s.header.stamp = stamp
        # Get the point in the "map" coordinate system
        try:
            point_world = self.tf_buf.transform(point_s, "map")
            # Create a Pose object with the same position
            pose = Pose()
            pose.position.x = point_world.point.x
            pose.position.y = point_world.point.y
            pose.position.z = point_world.point.z

            
        except Exception as e:
            print(e)
            pose = None
        return pose
    def find_faces(self):
        #use lock to make sure we are not reading and writing to the images at the same time
        with self.image_lock:
            #check if we have received images
            if self.latest_rgb_image_msg is None or self.latest_depth_image_msg is None:
                return
            print('I got a new image!')
            # Convert the images into a OpenCV (numpy) format
            try:
                rgb_image = self.bridge.imgmsg_to_cv2(self.latest_rgb_image_msg, "bgr8")
            except CvBridgeError as e:
                print(e)
            time_difference = abs(self.latest_rgb_image_msg.header.stamp.to_sec() - self.latest_depth_image_msg.header.stamp.to_sec())
            if time_difference > 1:
                print("Timestamps are more than 1 second apart" + str(time_difference))
                return 0
            else:
                print("Timestamps are within 1 second apart" + str(time_difference))
            try:
                depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image_msg, "32FC1")
            except CvBridgeError as e:
                print(e)
            # Set the dimensions of the image
            self.dims = rgb_image.shape
            h = self.dims[0]
            w = self.dims[1]
            # Tranform image to gayscale
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            # Do histogram equlization
            #img = cv2.equalizeHist(gray)
            # Detect the faces in the image
            #face_rectangles = self.face_detector(rgb_image, 0)
            blob = cv2.dnn.blobFromImage(cv2.resize(rgb_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.face_net.setInput(blob)
            face_detections = self.face_net.forward()
            for i in range(0, face_detections.shape[2]):
                confidence = face_detections[0, 0, i, 2]
                if confidence>0.8:
                    box = face_detections[0,0,i,3:7] * np.array([w,h,w,h])
                    box = box.astype('int')
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    # Extract region containing face
                    face_region = rgb_image[y1:y2, x1:x2]
                    # Visualize the extracted face
                    cv2.imshow("ImWindow", face_region)
                    cv2.waitKey(1)
                    # Find the distance to the detected face
                    face_distance = float(np.nanmean(depth_image[y1:y2,x1:x2]))
                    print('Distance to face', face_distance)
                    # Get the time that the depth image was recieved
                    depth_time = self.latest_depth_image_msg.header.stamp
                    # Find the location of the detected face
                    pose = self.get_pose((x1,x2,y1,y2), face_distance, depth_time)
                    if pose is not None:

                        #face_region: Any,
                        # face_distance: Any,
                        # depth_time: Any,
                        # identity: Any,
                        # pose: Any,
                        # confidence: Any

                        detectedFace = DetectedFace(face_region, face_distance, depth_time, "unknown", pose, confidence)
                        self.detected_faces_tracker.add_face(detectedFace)

                       
                            # Publish the average pose of the group
                        # # Create a marker used for visualization
                        # self.marker_num += 1
                        # marker = Marker()
                        # marker.header.stamp = rospy.Time(0)
                        # marker.header.frame_id = 'map'
                        # marker.pose = pose
                        # marker.pose.orientation = Quaternion(0.5, 0.5, 0.5, 0.5)
                        # marker.type = Marker.CUBE
                        # marker.action = Marker.ADD
                        # marker.frame_locked = False
                        # marker.lifetime = rospy.Duration.from_sec(10)
                        # marker.id = self.marker_num
                        # marker.scale = Vector3(0.1, 0.1, 0.1)
                        # marker.color = ColorRGBA(0, 1, 0, 1)
                        # self.marker_array.markers.append(marker)
                        # self.markers_pub.publish(self.marker_array)

            self.show_markers(self.detected_faces_tracker.get_grouped_faces())

    def show_markers(self, grouped_faces):

        markers = MarkerArray()
        marker_num = 0
        
        for group in grouped_faces:
            avg_pose = group['avg_pose']
            print('Average pose', avg_pose, 'Number of faces here', len(group['faces']))
            # Create a marker used for visualization
            marker_num += 1
            marker = Marker()
            marker.header.stamp = rospy.Time(0)
            marker.header.frame_id = 'map'
            marker.pose = avg_pose
            marker.pose.orientation = Quaternion(0.5, 0.5, 0.5, 0.5)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.frame_locked = False
            marker.lifetime = rospy.Duration.from_sec(10)
            marker.id = marker_num
            marker.scale = Vector3(0.1, 0.1, 0.1)
            marker.color = ColorRGBA(0, 1, 0, 1)
            markers.markers.append(marker)

        self.markers_pub.publish(markers)

            
            





        
    def depth_callback(self,data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)
        # Do the necessairy conversion so we can visuzalize it in OpenCV
        image_1 = depth_image / np.nanmax(depth_image)
        image_1 = image_1*255
        image_viz = np.array(image_1, dtype=np.uint8)
        #cv2.imshow("Depth window", image_viz)
        #cv2.waitKey(1)
        #plt.imshow(depth_image)
        #plt.show()
def main():
    face_finder = face_localizer()
    rospy.Timer(rospy.Duration(1.0), lambda event: face_finder.find_faces())
    rospy.spin()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
