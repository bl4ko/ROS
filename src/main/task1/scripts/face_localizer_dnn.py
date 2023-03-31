#!/usr/bin/python3
import math
import threading
from os.path import dirname, join
import cv2
import numpy as np
from map_manager import MapManager
import tf2_ros
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose, Quaternion
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import message_filters
from message_filters import ApproximateTimeSynchronizer
from tf.transformations import quaternion_from_euler
from task1.msg import UniqueFaceCoords
from task1.msg import DetectedFaces


class DetectedFace:
    def __init__(
        self,
        face_region,
        face_distance,
        depth_time,
        identity,
        pose,
        confidence,
        pose_left,
        pose_right,
        xr,
        yr,
        rr_x,
        rr_y,
        rr_z,
        rr_w,
    ):
        self.confidence = confidence
        self.face_region = face_region
        self.face_distance = face_distance
        self.depth_time = depth_time
        self.identity = identity
        self.pose = pose
        self.pose_left = pose_left
        self.pose_right = pose_right
        self.xr = xr
        self.yr = yr
        self.rr_x = rr_x
        self.rr_y = rr_y
        self.rr_z = rr_z
        self.rr_w = rr_w


class DetectedFacesTracker:
    def __init__(self):
        self.faces = []
        self.grouped_faces_by_distance = []  # list of objects
        self.distance_threshold = 0.5  # in meters
        self.detection_threshold = (
            1  # number of detections to consider a face as a valid face
        )
        self.detection_history = 50  # number of detections to keep in the history
        self.map_manager = MapManager()

    def add_face(self, face):

        x_c = face.pose.position.x
        y_c = face.pose.position.y
        pose_face_left = face.pose_left
        pose_face_right = face.pose_right

        new_face_normal = self.get_face_normal(
            x_c, y_c, pose_face_left, pose_face_right
        )

        # check if the face is close to any of the existing faces and if it is on the same side of the wall
        for i in range(len(self.grouped_faces_by_distance)):
            avg_pose = self.grouped_faces_by_distance[i]["avg_pose"]

            # we store every normal just in case we need it later
            # we only need to check the first normal because we only add faces that are close to each other
            first_face_in_group_normal = self.grouped_faces_by_distance[i][
                "potential_faces_normals"
            ][0]

            n1, n2 = first_face_in_group_normal[0], first_face_in_group_normal[1]
            same_side = self.same_side_normal(
                new_face_normal[0], new_face_normal[1], n1, n2
            )

            if self.is_close(avg_pose, face.pose) and same_side:
                print("face is close to existing face with id: ", i)
                print("face normal: ", new_face_normal)

                if (
                    self.grouped_faces_by_distance[i]["detections"]
                    >= self.detection_history
                ):
                    
                    return # we don't want to add more faces than the detection history

                    # sort the faces by confidence
                    # remove the oldest face
                    # self.grouped_faces_by_distance[i]["faces"].pop(0)
                    # self.grouped_faces_by_distance[i]["potential_faces_normals"].pop(0)
                    # self.grouped_faces_by_distance[i]["detections"] -= 1

                # if there are more than
                self.grouped_faces_by_distance[i]["faces"].append(face)
                self.grouped_faces_by_distance[i]["avg_pose"] = self.update_avg_pose(
                    self.grouped_faces_by_distance[i]["faces"]
                )
                self.grouped_faces_by_distance[i]["detections"] += 1
                self.grouped_faces_by_distance[i]["potential_faces_normals"].append(
                    new_face_normal
                )
                return

        # if the face is not close to any of the existing faces, add it as a new face
        #print 
        print("new face detected")
        print("face normal: ", new_face_normal)
        print("face pose: ", face.pose)
        self.grouped_faces_by_distance.append(
            {
                "faces": [face],
                "avg_pose": face.pose,
                "detections": 1,
                "potential_faces_normals": [new_face_normal],
            }
        )

    def update_avg_pose(self, faces):
        # update the average pose of a face
        avg_pose = Pose()
        avg_pose.position.x = np.mean([face.pose.position.x for face in faces])
        avg_pose.position.y = np.mean([face.pose.position.y for face in faces])
        return avg_pose

    def is_close(self, pose1, pose2):
        # only check the x and y coordinates
        return (
            np.linalg.norm(
                np.array([pose1.position.x, pose1.position.y])
                - np.array([pose2.position.x, pose2.position.y])
            )
            < self.distance_threshold
        )

    def get_grouped_faces(self):
        # return the faces grouped by distance

        # remove faces that have not been detected enough times
        return [
            group
            for group in self.grouped_faces_by_distance
            if group["detections"] >= self.detection_threshold
        ]

    def get_greet_locations(self):
        loc = []
        for group in self.grouped_faces_by_distance:
            if group["detections"] >= self.detection_threshold:
                # Compute the location of the greeting based on the average robot pose and average face pose
                avg_pose = group["avg_pose"]
                avg_face_greet_location = (0, 0)
                total_weight = 0
                
                # Sort the faces by distance
                for face in group["faces"]:
                    xr = face.xr
                    yr = face.yr
                    pose_face_left = face.pose_left
                    pose_face_right = face.pose_right
                    confidence = face.confidence
                    face_distance = face.face_distance

                    xg, yg = self.map_manager.get_face_greet_location(
                        avg_pose.position.x,
                        avg_pose.position.y,
                        xr,
                        yr,
                        pose_face_left,
                        pose_face_right,
                    )
                    
                    # Weight the greet location based on the confidence and the inverse of the distance
                    weight = confidence / face_distance
                    avg_face_greet_location = (
                        avg_face_greet_location[0] + xg * weight,
                        avg_face_greet_location[1] + yg * weight,
                    )
                    total_weight += weight

                # Normalize the average face greet location by the total weight
                avg_face_greet_location = (
                    avg_face_greet_location[0] / total_weight,
                    avg_face_greet_location[1] / total_weight,
                )

                loc.append((avg_face_greet_location, group))

        return loc

    def get_face_normal(self, x_ce, y_ce, fpose_left, fpose_right):
        """
        Returns the normal vector of the face.
        """
        x_left = fpose_left.position.x
        y_left = fpose_left.position.y
        x_right = fpose_right.position.x
        y_right = fpose_right.position.y

        # print("x_left: ", x_left)
        # print("y_left: ", y_left)
        # print("x_right: ", x_right)
        # print("y_right: ", y_right)

        dx = x_right - x_left
        dy = y_right - y_left

        # print("dx: ", dx)
        # print("dy: ", dy)

        # get normalized perpendicular vector
        perp_dx = -dy / ((dy * dy + dx * dx) ** 0.5)
        perp_dy = dx / ((dy * dy + dx * dx) ** 0.5)

        # print("perp_dx: ", perp_dx)
        # print("perp_dy: ", perp_dy)

        return (perp_dx, perp_dy)

    def same_side_normal(self, normal_x1, normal_y1, normal_x2, normal_y2):
        """
        Returns true if detections were on same side of wall and false otherwise
        """
        dot_prod = normal_x1 * normal_x2 + normal_y1 * normal_y2

        if dot_prod < 0:
            return False
        else:
            return True


class FaceLocalizer:
    def __init__(self):
        self.detected_faces_publisher = rospy.Publisher(
            "detected_faces", DetectedFaces, queue_size=20
        )
        self.bridge = (
            CvBridge()
        )  # An object we use for converting images between ROS format and OpenCV format
        # The function for performin HOG face detection
        # self.face_detector = dlib.get_frontal_face_detector()
        protoPath = join(dirname(__file__), "deploy.prototxt.txt")
        modelPath = join(dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")
        self.face_net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        self.dims = (0, 0, 0)  # A help variable for holding the dimensions of the image
        self.marker_array = MarkerArray()  # Store markers for faces
        self.marker_num = 1
        self.markers_pub = rospy.Publisher("face_markers", MarkerArray, queue_size=1000)
        self.face_visit_locations_markers_pub = rospy.Publisher(
            "face_visit_locations_markers", MarkerArray, queue_size=1000
        )
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        self.image_lock = threading.Lock()
        self.rgb_image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
        self.depth_image_sub = message_filters.Subscriber(
            "/camera/depth/image_raw", Image
        )
        self.time_synchronizer = ApproximateTimeSynchronizer(
            [self.rgb_image_sub, self.depth_image_sub], queue_size=5, slop=0.5
        )
        self.time_synchronizer.registerCallback(self.image_callback)
        self.latest_rgb_image_msg = None
        self.latest_depth_image_msg = None

        self.detected_faces_tracker = DetectedFacesTracker()
        self.unique_groups = 0
        self.already_sent_ids = []

    def image_callback(self, rgb_image_msg, depth_image_msg):
        with self.image_lock:
            self.latest_rgb_image_msg = rgb_image_msg
            self.latest_depth_image_msg = depth_image_msg

    def get_pose(self, coords, dist, stamp):
        k_f = 554  # kinect focal length in pixels
        x1, x2, y1, y2 = coords
        face_x = self.dims[1] / 2 - (x1 + x2) / 2.0
        face_y = self.dims[0] / 2 - (y1 + y2) / 2.0
        angle_to_target = np.arctan2(face_x, k_f)
        x, y = dist * np.cos(angle_to_target), dist * np.sin(angle_to_target)
        point_s = PointStamped()
        point_s.point.x = -y
        point_s.point.y = 0
        point_s.point.z = x
        point_s.header.frame_id = "camera_rgb_optical_frame"
        point_s.header.stamp = stamp
        try:
            point_world = self.tf_buf.transform(point_s, "map")
            pose = Pose()
            pose.position.x = point_world.point.x
            pose.position.y = point_world.point.y
            pose.position.z = point_world.point.z

        except Exception as error:
            print(error)
            pose = None
        return pose

    def find_faces(self):
        with self.image_lock:
            if self.latest_rgb_image_msg is None or self.latest_depth_image_msg is None:
                return
            print("I got a new image!")
            try:
                rgb_image = self.bridge.imgmsg_to_cv2(self.latest_rgb_image_msg, "bgr8")
            except CvBridgeError as e:
                print(e)
            try:
                depth_image = self.bridge.imgmsg_to_cv2(
                    self.latest_depth_image_msg, "32FC1"
                )
            except CvBridgeError as e:
                print(e)
            self.dims = rgb_image.shape
            h = self.dims[0]
            w = self.dims[1]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(rgb_image, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0),
            )
            self.face_net.setInput(blob)
            face_detections = self.face_net.forward()
            for i in range(0, face_detections.shape[2]):
                confidence = face_detections[0, 0, i, 2]
                if confidence > 0.7:
                    box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    box = box.astype("int")
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    face_region = rgb_image[y1:y2, x1:x2]
                    face_distance = float(np.nanmean(depth_image[y1:y2, x1:x2]))
                    print("Distance to face", face_distance)
                    depth_time = self.latest_depth_image_msg.header.stamp

                    try:
                        base_pos_transform = self.tf_buf.lookup_transform(
                            "map", "base_link", depth_time
                        )
                        xr = base_pos_transform.transform.translation.x
                        yr = base_pos_transform.transform.translation.y
                        zr = base_pos_transform.transform.translation.z

                        # get robot rotation  it is determining the position and orientation
                        #  of the robot's base (i.e., the robot's chassis)
                        # relative to a global reference frame, which is typically
                        # referred to as the map frame in ROS-based robotic systems.
                        rr_x = base_pos_transform.transform.rotation.x
                        rr_y = base_pos_transform.transform.rotation.y
                        rr_z = base_pos_transform.transform.rotation.z
                        rr_w = base_pos_transform.transform.rotation.w
                        rospy.loginfo(
                            "x: %s, y: %s, z: %s" % (str(xr), str(yr), str(zr))
                        )
                    except Exception as e:
                        rospy.logerr("Error in base coords lookup: %s" % (str(e)))
                        return

                    pose = self.get_pose((x1, x2, y1, y2), face_distance, depth_time)
                    if pose is not None:
                        x_ce = int(round((x1 + x2) / 2))
                        print("center: " + str(x_ce))
                        face_distance_left = float(
                            np.nanmean(depth_image[y1:y2, x1 : (x1 + 1)])
                        )
                        pose_left = self.get_pose(
                            (x1, x1, y1, y1), face_distance_left, depth_time
                        )

                        face_distance_right = float(
                            np.nanmean(depth_image[y1:y2, (x2 - 1) : x2])
                        )
                        pose_right = self.get_pose(
                            (x2, x2, y1, y1), face_distance_right, depth_time
                        )

                        xpl = pose_left.position.x
                        ypl = pose_left.position.y
                        zpl = pose_left.position.z

                        xpr = pose_right.position.x
                        ypr = pose_right.position.y
                        zpr = pose_right.position.z

                        ### check if detection is a real face
                        if (
                            (pose is not None)
                            and (pose_left is not None)
                            and (pose_right is not None)
                            and not (
                                math.isnan(xpl)
                                or math.isnan(ypl)
                                or math.isnan(zpl)
                                or math.isnan(xpr)
                                or math.isnan(ypr)
                                or math.isnan(zpr)
                                or math.isnan(pose.position.x)
                                or math.isnan(pose.position.y)
                                or math.isnan(pose.position.z)
                                or math.isnan(pose_left.position.x)
                                or math.isnan(pose_left.position.y)
                                or math.isnan(pose_left.position.z)
                                or math.isnan(pose_right.position.x)
                                or math.isnan(pose_right.position.y)
                                or math.isnan(pose_right.position.z)

                            
                            )
                        ):
                            # rospy.logerr("Sending face.")
                            detectedFace = DetectedFace(
                                face_region,
                                face_distance,
                                depth_time,
                                "unknown",
                                pose,
                                confidence,
                                pose_left,
                                pose_right,
                                xr,
                                yr,
                                rr_x,
                                rr_y,
                                rr_z,
                                rr_w,
                            )
                            self.detected_faces_tracker.add_face(detectedFace)

            #self.show_markers(self.detected_faces_tracker.get_grouped_faces())
            data = self.detected_faces_tracker.get_greet_locations()

            coords = []
            for d in data:
                coords.append(d[0])

            


            groups = []
            ids = []
            i = 0
            for d in data:
                groups.append(d[1])
                ids.append(i)
                i += 1

            to_publish = DetectedFaces()

            for i in range(len(groups)):
                self.unique_groups = len(groups)
                self.already_sent_ids = ids


                fc_group = groups[i]
                x = fc_group["avg_pose"].position.x
                y = fc_group["avg_pose"].position.y

                face_c_x = coords[-1][0]
                face_c_y = coords[-1][1]


                normal_x = fc_group["potential_faces_normals"][0][0]
                normal_y = fc_group["potential_faces_normals"][0][1]

                q_dest = self.quaternion_for_face_greet(face_c_x, face_c_y, x, y)
                rr_x = q_dest[0]
                rr_y = q_dest[1]
                rr_z = q_dest[2]
                rr_w = q_dest[3]

                unfccoords=self.make_unfcoords_msg(i,
                    face_c_x, face_c_y, rr_x, rr_y, rr_z, rr_w, normal_x, normal_y, x, y
                )
                to_publish.array.append(unfccoords)

            self.detected_faces_publisher.publish(to_publish)
            self.show_markers_coords(coords)
            self.show_markers(groups)

    def quaternion_for_face_greet(self, x1, y1, x2, y2):
        v1 = np.array([x2, y2, 0]) - np.array([x1, y1, 0])
        # in the direction of z axis
        v0 = [1, 0, 0]

        # compute yaw - rotation around z axis
        yaw = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])

        rospy.loginfo("Yaw: %s" % str(yaw * 57.2957795))

        q = quaternion_from_euler(0, 0, yaw)

        rospy.loginfo("Got quaternion: %s" % str(q))

        return q

    def show_markers(self, grouped_faces):

        markers = MarkerArray()
        marker_num = 0

        for group in grouped_faces:
            avg_pose = group["avg_pose"]
            print("Average pose", avg_pose, "Number of faces here", len(group["faces"]))
            # Create a marker used for visualization
            marker_num += 1
            marker = Marker()
            marker.header.stamp = rospy.Time(0)
            marker.header.frame_id = "map"
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

    def show_markers_coords(self, coords):

        markers = MarkerArray()
        marker_num = 0

        for face in coords:
            # Create a marker used for visualization
            marker_num += 1
            marker = Marker()
            marker.header.stamp = rospy.Time(0)
            marker.header.frame_id = "map"
            marker.pose.position.x = face[0]
            marker.pose.position.y = face[1]
            marker.pose.orientation = Quaternion(0.5, 0.5, 0.5, 0.5)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.frame_locked = False
            marker.lifetime = rospy.Duration.from_sec(10)
            marker.id = marker_num
            marker.scale = Vector3(0.1, 0.1, 0.1)
            # blue
            marker.color = ColorRGBA(0, 0, 1, 1)
            markers.markers.append(marker)

        self.face_visit_locations_markers_pub.publish(markers)

    def make_unfcoords_msg(
        self,id, x, y, rr_x, rr_y, rr_z, rr_w, normal_x, normal_y, face_c_x, face_c_y
    ):
        msg = UniqueFaceCoords()
        msg.x_coord = x
        msg.y_coord = y
        msg.rr_x = rr_x
        msg.rr_y = rr_y
        msg.rr_z = rr_z
        msg.rr_w = rr_w

        msg.face_id = id

        msg.normal_x = normal_x
        msg.normal_y = normal_y

        msg.face_c_x = face_c_x
        msg.face_c_y = face_c_y

        # rospy.loginfo(
        #     "New unique face coords published on topic with id: %d."
        #     % self.unique_groups
        # )
        # rospy.loginfo(msg)
        #self.coords_publisher.publish(msg)

        return msg

    

    def depth_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)
        image_1 = depth_image / np.nanmax(depth_image)
        image_1 = image_1 * 255
        image_viz = np.array(image_1, dtype=np.uint8)


def main():
    rospy.init_node("face_localizer", anonymous=True)
    face_finder = FaceLocalizer()
    rospy.Timer(rospy.Duration(0.3), lambda event: face_finder.find_faces())
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
