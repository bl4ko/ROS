#!/usr/bin/python3

# TODO: # pylint: disable=fixme
# pylint: disable=too-many-instance-attributes, disable=too-many-arguments, disable=too-many-locals. disable=too-few-public-methods, disable=duplicate-code

"""
Module for the face localizer node, which uses Google's 
Mediapipe library to detect faces and publishes their position.
"""
import math
import threading
from typing import List, Tuple, Optional
import easyocr

import cv2
import numpy as np
import tf2_ros
import rospy
from tf2_ros import TransformException  # pylint: disable=no-name-in-module
from map_manager import MapManager
import mediapipe as mp
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose, Quaternion
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import message_filters
from message_filters import ApproximateTimeSynchronizer
from tf.transformations import quaternion_from_euler
from combined.msg import UniqueFaceCoords, DetectedFaces
from combined.srv import IsPoster, IsPosterResponse, IsPosterRequest


class DetectedFace:
    """
    Class to store the information of a detected face.
    """

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
        robot_x,
        robot_y,
        robot_z,
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
        self.robot_x = robot_x
        self.robot_y = robot_y
        self.robot_z = robot_z
        self.robot_rotation_x = rr_x
        self.robot_rotation_y = rr_y
        self.robot_rotation_z = rr_z
        self.robot_rotation_w = rr_w


class FaceGroup:
    """
    Class to store the information of a group of faces.
    A group of faces is a group of faces that are close to each other,
    and are on the same side of the wall (i.e. they represent the same person).
    """

    def __init__(self, initial_face: DetectedFace, group_id: int) -> None:
        self.group_id = group_id
        self.faces = [initial_face]
        self.avg_pose = initial_face.pose
        self.detections = 1
        self.potential_normals = [self.get_face_normal(initial_face)]
        self.avg_face_normal = self.potential_normals[0]
        self.is_poster = -1  # -1: unknown, 0: not a poster, 1: is a poster
        self.poster_text = ""

    def __str__(self) -> str:
        return f"FaceGroup(detections={self.detections})"

    def get_face_normal(self, face: DetectedFace) -> Tuple[float, float]:
        """
        Gets the normal of a face.

        Args:
            face (DetectedFace): face to get the normal of.

        Returns:
            Tuple[float, float]: normal of the face.
        """
        x_left = face.pose_left.position.x
        y_left = face.pose_left.position.y
        x_right = face.pose_right.position.x
        y_right = face.pose_right.position.y

        d_x = x_right - x_left
        d_y = y_right - y_left

        perp_dx = -d_y / ((d_y * d_y + d_x * d_x) ** 0.5)
        perp_dy = d_x / ((d_y * d_y + d_x * d_x) ** 0.5)

        return (perp_dx, perp_dy)

    def update_avg_pose(self) -> None:
        """
        Updates the average pose of the group of faces.
        """
        self.avg_pose.position.x = np.mean([face.pose.position.x for face in self.faces])
        self.avg_pose.position.y = np.mean([face.pose.position.y for face in self.faces])

    def add_face(self, face: DetectedFace) -> None:
        """
        Adds a face to the group of faces.

        Args:
            face (DetectedFace): face to add to the group.
        """
        self.faces.append(face)
        self.update_avg_pose()
        self.detections += 1
        new_normal = self.get_face_normal(face)
        self.potential_normals.append(new_normal)
        self.avg_face_normal = np.mean(self.potential_normals, axis=0)

    def update_poster_data(self, is_poster: int, poster_text: str) -> None:
        """
        Updates the poster data of the group of faces.

        Args:
            is_poster (int): is the group of faces a poster.
            poster_text (str): text on the poster.
        """
        self.is_poster = is_poster
        self.poster_text = poster_text


class DetectedFacesTracker:
    """
    Class to track the faces detected by the face detector.
    """

    def __init__(self):
        self.tracked_faces = []
        self.face_groups: List[FaceGroup] = []
        self.group_distance_threshold: float = 0.5  # in meters
        self.valid_detection_threshold: int = (
            3  # number of detections to consider a face as a valid face
        )
        self.history_limit: int = 50  # number of detections to keep in the history
        self.face_max_distance: float = 3  # in meters max distance to a valid face
        self.greeting_max_distance: float = 0.6  # in meters max distance to a valid face
        self.map_manager: MapManager = MapManager()

    def print_face_groups(self):
        """
        Prints the face groups to stdout.
        """
        print("  Group faces:")
        for i, face_group in enumerate(self.face_groups):
            print(f"    Group {i + 1}: {face_group}")

    def add_face(self, face: DetectedFace) -> None:
        """
        Adds a face to the group of faces, or creates a new group if
        the face is not close to any of the existing faces.

        Args:
            face (DetectedFace): face to add
        """
        if face.face_distance > self.face_max_distance:
            print(f"Face is too far away: {face.face_distance} not adding it")
            return

        for face_group in self.face_groups:
            avg_pose = face_group.avg_pose

            normal_1, normal_2 = (
                face_group.avg_face_normal[0],
                face_group.avg_face_normal[1],
            )
            fg_normal1, fg_normal2 = face_group.get_face_normal(face)
            same_side = self.same_side_normal(fg_normal1, fg_normal2, normal_1, normal_2)

            if self.is_close(avg_pose, face.pose) and same_side:
                if face_group.detections >= self.history_limit:
                    print("Face group has reached the history limit, not adding new faces")
                    return

                face_group.add_face(face)
                print("Face added to existing face group!")
                self.print_face_groups()
                return

        print("New face detected, new facegroup created and added to list of unique groups!")
        self.face_groups.append(FaceGroup(face, len(self.face_groups) + 1))
        self.print_face_groups()

    def update_poster_data(self, is_poster: int, poster_text: str, group_id: int) -> bool:
        """
        Updates the poster data of a group of faces.

        Args:
            is_poster (int): is the group of faces a poster.
            poster_text (str): text on the poster.
            group_id (int): id of the group of faces.
        """
        for face_group in self.face_groups:
            if face_group.group_id == group_id:
                face_group.update_poster_data(is_poster, poster_text)
                return True
        return False

    def is_close(self, pose1: Pose, pose2: Pose):
        """
        Checks if two poses are close to each other.

        Args:
            pose1 (Pose): First pose
            pose2 (Pose): Second pose

        Returns:
            bool: True if the poses are close, False otherwise
        """
        # only check the x and y coordinates
        return (
            np.linalg.norm(
                np.array([pose1.position.x, pose1.position.y])
                - np.array([pose2.position.x, pose2.position.y])
            )
            < self.group_distance_threshold
        )

    def get_grouped_faces(self):
        """
        Returns the list of faces that have been detected more than valid_detection_threshold times.

        Returns:
            List[List[DetectedFace]]: groups with enough detections.
        """
        return [
            group
            for group in self.face_groups
            if group.detections >= self.valid_detection_threshold
        ]

    def get_greet_locations(self) -> List[Tuple[Tuple[float, float], FaceGroup]]:
        """
        Calculate and return the greet locations for each valid face group.

        Returns:
            List[Tuple[Tuple[float, float], FaceGroup]]: A list of tuples, where the first element
                is a tuple representing the (x, y) coordinates of the average greet location,
                and the second element is the face group dictionary.
        """
        greet_locations = []

        # Iterate over valid face groups
        for group in (
            g for g in self.face_groups if g.detections >= self.valid_detection_threshold
        ):
            avg_pose = group.avg_pose
            total_weight = 0
            weighted_greet_locations = np.zeros(2)

            # No need to sort faces by distance as it's not used in the following loop
            for face in group.faces:
                robot_x, robot_y = face.robot_x, face.robot_y
                face_pose_left, face_pose_right = face.pose_left, face.pose_right
                confidence, face_distance = face.confidence, face.face_distance

                greet_x, greet_y = self.map_manager.get_face_greet_location(
                    avg_pose.position.x,
                    avg_pose.position.y,
                    robot_x,
                    robot_y,
                    face_pose_left,
                    face_pose_right,
                )

                greet_to_face_distance = np.linalg.norm(
                    np.array([greet_x, greet_y]) - np.array([robot_x, robot_y])
                )

                # Weight the greet location based on the confidence and the inverse of the distance
                weight = confidence / face_distance * greet_to_face_distance
                weighted_greet_locations += np.array([greet_x, greet_y]) * weight
                total_weight += weight

            # Normalize the average face greet location by the total weight
            avg_face_greet_location = tuple(weighted_greet_locations / total_weight)

            # Check distance of greet location to face
            greet_to_face_distance = np.linalg.norm(
                np.array(avg_face_greet_location)
                - np.array([avg_pose.position.x, avg_pose.position.y])
            )

            if greet_to_face_distance <= self.greeting_max_distance:
                greet_locations.append((avg_face_greet_location, group))

        return greet_locations

    def same_side_normal(
        self, normal_x1: float, normal_y1: float, normal_x2: float, normal_y2: float
    ) -> bool:
        """
        Checks if two normals are on the same side.

        Args:
            normal_x1 (float): First component of the first normal
            normal_y1 (float): Second component of the first normal
            normal_x2 (float): First component of the second normal
            normal_y2 (float): Second component of the second normal

        Returns:
            bool: True if the normals are on the same side, False otherwise
        """
        dot_prod = normal_x1 * normal_x2 + normal_y1 * normal_y2

        if dot_prod < 0:
            return False
        return True


class FaceLocalizer:
    """
    Class for localizing faces in the robot's camera image.
    """

    def __init__(self):
        self.detected_faces_publisher = rospy.Publisher(
            "detected_faces", DetectedFaces, queue_size=20
        )
        self.bridge = CvBridge()  # Object for converting between ROS and OpenCV image formats
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
        self.depth_image_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
        self.time_synchronizer = ApproximateTimeSynchronizer(
            [self.rgb_image_sub, self.depth_image_sub], queue_size=5, slop=0.5
        )
        self.time_synchronizer.registerCallback(self.image_callback)
        self.latest_rgb_image_msg = None
        self.latest_depth_image_msg = None
        self.detected_faces_tracker = DetectedFacesTracker()
        self.unique_groups = 0
        self.already_sent_ids = []
        self.reader = easyocr.Reader(["en"], gpu=True)
        self.is_poster_service = rospy.Service("is_poster", IsPoster, self.is_poster_callback)

    def image_callback(self, rgb_image_msg, depth_image_msg):
        """
        Callback for when a new image is received.

        Args:
            rgb_image_msg (np.ndarray): The RGB image.
            depth_image_msg (np.ndarray): The depth image.
        """
        with self.image_lock:
            self.latest_rgb_image_msg = rgb_image_msg
            self.latest_depth_image_msg = depth_image_msg

    def get_pose(
        self,
        bounding_box: Tuple[int, int, int, int],
        dist: float,
        time_stamp: rospy.Time,
    ) -> Optional[Pose]:
        """
        Get the pose of the detected face in the robot's coordinate system.

        Args:
            bounding_box (Tuple[int, int, int, int]): The face's bounding box coordinates
            (x1, x2, y1, y2).
            dist (float): The distance to the detected face.
            time_stamp (rospy.Time): The timestamp associated with the depth image

        Returns:
            Optional[Pose]: The pose of the detected face in the robot's coordinate frame
            or None if the transformation failed.
        """
        kinect_focal_length = 554
        x_1, x_2, _, _ = bounding_box
        face_center_x = self.dims[1] / 2 - (x_1 + x_2) / 2.0
        # face_center_y = self.dims[0] / 2 - (y1 + y2) / 2.0
        angle_to_target = np.arctan2(face_center_x, kinect_focal_length)
        x_coord, y_coord = dist * np.cos(angle_to_target), dist * np.sin(angle_to_target)
        point_s = PointStamped()
        point_s.point.x = -y_coord
        point_s.point.y = 0
        point_s.point.z = x_coord
        point_s.header.frame_id = "camera_rgb_optical_frame"
        point_s.header.stamp = time_stamp
        # pylint: disable=R0801 # Similar with ground_ring_detector
        try:
            point_world = self.tf_buf.transform(point_s, "map")
            pose = Pose()
            pose.position.x = point_world.point.x
            pose.position.y = point_world.point.y
            pose.position.z = point_world.point.z

        except TransformException as error:
            print(error)
            pose = None
        return pose

    def detect_text(self, rgb_image: np.ndarray, confidence_threshold=0.7) -> List[tuple]:
        """
        Detects text in an RGB image.

        Args:
            rgb_image (np.ndarray): The input RGB image
            confidence_threshold (float): The confidence threshold for the text detection

        Returns:
            List[str]: A list of the detected texts
        """

        # use easyocr to detect text
        rospy.loginfo(f"Detecting text from image: {rgb_image}")
        result = self.reader.readtext(rgb_image)
        # ([[79, 173], [125, 173], [125, 213], [79, 213]], 'W', 0.9848111271858215),

        # filter out the text with low confidence
        result = [x for x in result if x[2] > confidence_threshold]

        # return the text and the bounding box
        return result

    def get_indentity(self, face_region: np.ndarray) -> str:
        """
        Get the identity of the person in the image.

        Args:
            rgb_image (np.ndarray): The input RGB image
            confidence_threshold (float): The confidence threshold for the text detection

        Returns:
            str: The identity of the person in the image
        """

        # use dlib to exteact vector

    def is_poster_callback(self, request: IsPosterRequest) -> IsPosterResponse:
        """
        Callback for the is_poster service. This is used when the robot is close to face or poster to detect and
        determine if it is a poster or not. it will return true if it is a poster and false if it is not a poster. and the
        text of the poster. IT SHOULD BE CALLED WHEN THE ROBOT IS CLOSE TO THE POSTER OR FACE.

        Args:
            request (IsPosterRequest): The service request.

        Returns:
            IsPosterResponse: The service response.
        """

        with self.image_lock:
            rgb_image_msg = self.latest_rgb_image_msg
            depth_image_msg = self.latest_depth_image_msg
            if rgb_image_msg is None or depth_image_msg is None:
                return IsPosterResponse(False, "No image received", 0)

            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "32FC1")

            # save the image for debugging
            cv2.imwrite("./debug/rgb_image.jpg", rgb_image)

            # detect face with near range
            with mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.65
            ) as face_detection:
                rgb_converted_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                rgb_converted_image.flags.writeable = False
                detection_results = face_detection.process(rgb_converted_image)
                rgb_converted_image.flags.writeable = True
                rgb_converted_image = cv2.cvtColor(rgb_converted_image, cv2.COLOR_RGB2BGR)

                print(
                    "isPosterCallback: Face detection results: ",
                    detection_results.detections,
                )

                if detection_results.detections is None:
                    rospy.loginfo(f"Number of detections is None")

                if detection_results.detections:
                    for detection in detection_results.detections:
                        confidence_score = detection.score[0]
                        if confidence_score > 0.65:
                            print("Face detected with high confidence: ", confidence_score)

                            bounding_box = detection.location_data.relative_bounding_box
                            image_height, image_width, _ = rgb_converted_image.shape
                            img_x, img_y, width, height = (
                                int(bounding_box.xmin * image_width),
                                int(bounding_box.ymin * image_height),
                                int(bounding_box.width * image_width),
                                int(bounding_box.height * image_height),
                            )

                            x_1, y_1, x_2, y_2 = (
                                img_x,
                                img_y,
                                img_x + width,
                                img_y + height,
                            )
                            face_region = rgb_image[y_1:y_2, x_1:x_2]
                            face_distance = float(np.nanmean(depth_image[y_1:y_2, x_1:x_2]))

                            # get indentity based  on facial featurs using dlib

                            print("Distance to face", face_distance)
                            depth_timestamp = self.latest_depth_image_msg.header.stamp

                            # Show face with bounding box
                            cv2.rectangle(
                                rgb_converted_image,
                                (x_1, y_1),
                                (x_2, y_2),
                                (0, 255, 0),
                                2,
                            )

                            # Define enlarged bounding box dimensions
                            x1_new = max(0, x_1 - int(0.8 * width))
                            y1_new = max(0, y_1 - int(1.2 * height))
                            x2_new = min(image_width, x_2 + int(0.8 * width))
                            y2_new = min(image_height, y_2 + int(1.2 * height))

                            # Draw another bounding box that is 0.2 wider than the face and 0.4 taller than the face
                            cv2.rectangle(
                                rgb_converted_image,
                                (x1_new, y1_new),
                                (x2_new, y2_new),
                                (0, 255, 0),
                                2,
                            )

                            # Crop and detect text
                            text_region = rgb_image[y1_new:y2_new, x1_new:x2_new]

                            text = self.detect_text(text_region, confidence_threshold=0.7)

                            recognized_text = {}

                            for i in range(len(text)):
                                # ([[79, 173], [125, 173], [125, 213], [79, 213]], 'W', 0.9848111271858215),
                                # get the bounding box of the text
                                text_box = text[i][0]

                                # get the text
                                text_string = text[i][1]

                                recognized_text[text_string] = text_box

                                # draw the bounding box of the text on the image and display the text
                                x1, y1 = (
                                    x_1 - int(0.8 * width) + text_box[0][0],
                                    y_1 - int(1.2 * height) + text_box[0][1],
                                )
                                x2, y2 = (
                                    x_1 - int(0.8 * width) + text_box[2][0],
                                    y_1 - int(1.2 * height) + text_box[2][1],
                                )
                                cv2.rectangle(
                                    rgb_converted_image,
                                    (int(x1), int(y1)),
                                    (int(x2), int(y2)),
                                    (0, 255, 0),
                                    2,
                                )
                                cv2.putText(
                                    rgb_converted_image,
                                    text_string,
                                    (
                                        int(x_1 - int(0.8 * width) + text_box[0][0]),
                                        int(y_1 - int(1.2 * height) + text_box[0][1]),
                                    ),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )

                            # sort so the text is left to right top to bottom
                            recognized_text = dict(
                                sorted(
                                    recognized_text.items(),
                                    key=lambda item: item[1][0][0],
                                )
                            )

                            # to string
                            recognized_text = " ".join(list(recognized_text.keys()))

                            # cv2.imshow("face with text", rgb_converted_image)

                            cv2.imwrite("./debug/text_region.jpg", rgb_converted_image)

                            is_poster = len(recognized_text) > 1
                            print("Is poster: ", is_poster)
                            print("Recognized text: ", recognized_text)

                            # face_pose = self.get_pose(
                            #     (x_1, x_2, y_1, y_2), face_distance, depth_timestamp
                            # )

                            # based on group id and text, determine if the poster is the target poster
                            updated_face_group = self.detected_faces_tracker.update_poster_data(
                                is_poster=is_poster,
                                poster_text=recognized_text,
                                group_id=request.group_id,
                            )

                            if updated_face_group:
                                return IsPosterResponse(is_poster, recognized_text, 1)

                            else:
                                return IsPosterResponse(False, "group_id not found", 0)
                else:
                    return IsPosterResponse(False, "no face detected", 0)

        return IsPosterResponse(False, "detector err", 0)

    def detect_face(
        self, detection_range: int, rgb_image: np.ndarray, depth_image: np.ndarray
    ) -> bool:
        """
        Detects a face in an RGB image and estimates its position in the robot's coordinate frame.

        Args:
            detection_range (int): The model selection parameter for the face detection algorithm.
            rgb_image (np.ndarray): The input RGB image
            depth_image (np.ndarray): The depth image associated with the RGB image.

        Returns:
            bool: True if a real face was detected, False otherwise.
        """
        with mp.solutions.face_detection.FaceDetection(
            model_selection=detection_range, min_detection_confidence=0.50
        ) as face_detection:
            rgb_converted_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            rgb_converted_image.flags.writeable = False
            detection_results = face_detection.process(rgb_converted_image)
            rgb_converted_image.flags.writeable = True
            rgb_converted_image = cv2.cvtColor(rgb_converted_image, cv2.COLOR_RGB2BGR)

            if detection_results.detections:
                for detection in detection_results.detections:
                    confidence_score = detection.score[0]
                    print("Confidence score: ", confidence_score)
                    if confidence_score > 0.50:
                        print("Face detected with high confidence: ", confidence_score)

                        bounding_box = detection.location_data.relative_bounding_box
                        image_height, image_width, _ = rgb_converted_image.shape
                        img_x, img_y, width, height = (
                            int(bounding_box.xmin * image_width),
                            int(bounding_box.ymin * image_height),
                            int(bounding_box.width * image_width),
                            int(bounding_box.height * image_height),
                        )

                        x_1, y_1, x_2, y_2 = img_x, img_y, img_x + width, img_y + height
                        face_region = rgb_image[y_1:y_2, x_1:x_2]
                        face_distance = float(np.nanmean(depth_image[y_1:y_2, x_1:x_2]))
                        print("Distance to face", face_distance)
                        depth_timestamp = self.latest_depth_image_msg.header.stamp

                        try:
                            base_position_transform = self.tf_buf.lookup_transform(
                                "map", "base_link", depth_timestamp
                            )
                            robot_x = base_position_transform.transform.translation.x
                            robot_y = base_position_transform.transform.translation.y
                            robot_z = base_position_transform.transform.translation.z
                            robot_rotation_x = base_position_transform.transform.rotation.x
                            robot_rotation_y = base_position_transform.transform.rotation.y
                            robot_rotation_z = base_position_transform.transform.rotation.z
                            robot_rotation_w = base_position_transform.transform.rotation.w

                        except TransformException as error:
                            rospy.logerr(f"Error in base coords lookup: {error}")
                            return False

                        face_pose = self.get_pose(
                            (x_1, x_2, y_1, y_2), face_distance, depth_timestamp
                        )
                        if face_pose is not None:
                            face_distance_left = float(
                                np.nanmean(depth_image[y_1:y_2, x_1 : (x_1 + 1)])
                            )
                            pose_left = self.get_pose(
                                (x_1, x_1, y_1, y_1),
                                face_distance_left,
                                depth_timestamp,
                            )

                            face_distance_right = float(
                                np.nanmean(depth_image[y_1:y_2, (x_2 - 1) : x_2])
                            )
                            pose_right = self.get_pose(
                                (x_2, x_2, y_1, y_1),
                                face_distance_right,
                                depth_timestamp,
                            )

                            left_pose_x = pose_left.position.x
                            left_pose_y = pose_left.position.y
                            left_pose_z = pose_left.position.z

                            rigth_pose_x = pose_right.position.x
                            right_pose_y = pose_right.position.y
                            right_pose_z = pose_right.position.z

                            if (
                                (face_pose is not None)
                                and (pose_left is not None)
                                and (pose_right is not None)
                                and not (
                                    math.isnan(left_pose_x)
                                    or math.isnan(left_pose_y)
                                    or math.isnan(left_pose_z)
                                    or math.isnan(rigth_pose_x)
                                    or math.isnan(right_pose_y)
                                    or math.isnan(right_pose_z)
                                    or math.isnan(face_pose.position.x)
                                    or math.isnan(face_pose.position.y)
                                    or math.isnan(face_pose.position.z)
                                    or math.isnan(pose_left.position.x)
                                    or math.isnan(pose_left.position.y)
                                    or math.isnan(pose_left.position.z)
                                    or math.isnan(pose_right.position.x)
                                    or math.isnan(pose_right.position.y)
                                    or math.isnan(pose_right.position.z)
                                )
                            ):
                                detected_face = DetectedFace(
                                    face_region,
                                    face_distance,
                                    depth_timestamp,
                                    "unknown",
                                    face_pose,
                                    confidence_score,
                                    pose_left,
                                    pose_right,
                                    robot_x,
                                    robot_y,
                                    robot_z,
                                    robot_rotation_x,
                                    robot_rotation_y,
                                    robot_rotation_z,
                                    robot_rotation_w,
                                )
                                self.detected_faces_tracker.add_face(detected_face)
                                return True

        return False

    def find_faces(self) -> None:
        """
        Try to detect faces in the latest RGB and depth images.
        Also publishes the face detection results.
        """
        with self.image_lock:
            if self.latest_rgb_image_msg is None or self.latest_depth_image_msg is None:
                return

            print("I got a new image!")

            try:
                rgb_image = self.bridge.imgmsg_to_cv2(self.latest_rgb_image_msg, "bgr8")
            except CvBridgeError as bridge_error:
                rospy.logerr(bridge_error)

            try:
                depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image_msg, "32FC1")
            except CvBridgeError as bridge_error:
                rospy.logerr(bridge_error)

            self.dims = rgb_image.shape
            is_face_detected = False

            # Try to detect face at short range
            if not is_face_detected:
                is_face_detected = self.detect_face(0, rgb_image, depth_image)

            # Try to detect a face at long range
            if not is_face_detected:
                is_face_detected = self.detect_face(1, rgb_image, depth_image)

            # Show markers on every iteration
            self.show_face_markers()

    def show_face_markers(self) -> None:
        """
        Show face markers for the detected faces and publish the corresponding messages.

        This function extracts the coordinates, face groups, and IDs
        from the detected faces tracker, processes each face group,
        and then publishes the markers using the detected_faces_publisher.
        """
        face_group_data = self.detected_faces_tracker.get_greet_locations()
        coords, face_groups = [], []
        for _, (coord, group) in enumerate(face_group_data):
            coords.append(coord)
            face_groups.append(group)

        detected_faces_msg = DetectedFaces()

        # Process face groups and publish markers
        for _, (coord, face_group) in enumerate(zip(coords, face_groups)):
            self.unique_groups = len(face_groups)

            face_group_x, face_group_y = (
                face_group.avg_pose.position.x,
                face_group.avg_pose.position.y,
            )
            face_center_x, face_center_y = coord
            normal_x, normal_y = face_group.avg_face_normal

            # Calculate the quaternion for the face marker
            face_greet_orientation = self.quaternion_for_face_greet(
                face_center_x, face_center_y, face_group_x, face_group_y
            )

            # Create the message and append to the array
            msg = self.make_unique_face_coords_msg(
                face_group.group_id,
                face_center_x,
                face_center_y,
                face_greet_orientation,
                normal_x,
                normal_y,
                face_group_x,
                face_group_y,
            )
            detected_faces_msg.array.append(msg)

        # Publish detected faces and show markers
        self.detected_faces_publisher.publish(detected_faces_msg)
        self.show_markers_coords(coords)
        self.show_markers(face_groups)

    def quaternion_for_face_greet(
        self,
        face_center_x: float,
        face_center_y: float,
        face_group_x: float,
        face_group_y: float,
    ) -> Tuple[float, float, float, float]:
        """
        Calculate the quaternion for the face marker.

        Args:
            face_center_x (float): X coordinate of the face center
            face_center_y (float): Y coordinate of the face center
            face_group_x (float): X coordinate of the face group
            face_group_y (float): Y coordinate of the face group

        Returns:
            Tuple[float, float, float, float]: Quaternion for the face marker
        """
        vector_face_to_group = np.array([face_group_x, face_group_y, 0]) - np.array(
            [face_center_x, face_center_y, 0]
        )
        # in the direction of z axis
        vector_base = [1, 0, 0]
        # compute yaw - rotation around z axis
        yaw = np.arctan2(vector_face_to_group[1], vector_face_to_group[0]) - np.arctan2(
            vector_base[1], vector_base[0]
        )
        rospy.logdebug(f"Yaw: {str(yaw * 57.2957795)}")
        quat = quaternion_from_euler(0, 0, yaw)
        rospy.logdebug(f"Got quaternion: {str(quat)}")
        return quat

    def show_markers(self, grouped_faces):
        """
        Publishes markers for the detected faces.

        Args:
            grouped_faces (List[FaceGroup]): Face Groups to show markers for.
        """
        markers = MarkerArray()
        marker_num = 0

        for face_group in grouped_faces:
            avg_pose = face_group.avg_pose
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

            color = ColorRGBA()
            if face_group.is_poster == -1:
                color = ColorRGBA(1, 0.84, 0, 1)
            elif face_group.is_poster == 0:
                color = ColorRGBA(0, 1, 0, 1)
            else:
                color = ColorRGBA(1, 0, 0, 1)  # is poster

            marker.color = color
            markers.markers.append(marker)

        self.markers_pub.publish(markers)

    def show_markers_coords(self, coords: List[Tuple[float, float]]):
        """
        Publishes markers for the detected faces.

        Args:
            coords (List[Tuple[float, float]]): Coordinates to show markers for.
        """
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

    def make_unique_face_coords_msg(
        self,
        group_id: str,
        x_coord: float,
        y_coord: float,
        face_orientation: Quaternion,
        normal_x: float,
        normal_y: float,
        face_c_x: float,
        face_c_y: float,
    ) -> UniqueFaceCoords:
        """
        Create a UniqueFaceCoords message for a detected face.

        Args:
            group_id (int): The face's identifier.
            x (float): x-coordinate of the face's position.
            y (float): y-coordinate of the face's position.
            rr_x (float): x-component of the face's orientation quaternion.
            rr_y (float): y-component of the face's orientation quaternion.
            rr_z (float): z-component of the face's orientation quaternion.
            rr_w (float): w-component of the face's orientation quaternion.
            normal_x (float): x-component of the face's normal vector.
            normal_y (float): y-component of the face's normal vector.
            face_c_x (float): x-coordinate of the face's center.
            face_c_y (float): y-coordinate of the face's center.
        """
        msg = UniqueFaceCoords()
        msg.x_coord = x_coord
        msg.y_coord = y_coord
        msg.rr_x = face_orientation[0]
        msg.rr_y = face_orientation[1]
        msg.rr_z = face_orientation[2]
        msg.rr_w = face_orientation[3]
        msg.group_id = group_id
        msg.normal_x = normal_x
        msg.normal_y = normal_y
        msg.face_c_x = face_c_x
        msg.face_c_y = face_c_y
        return msg


def main():
    """
    Initialises the face_localizer node and calls every second the face_finder class to find_faces.
    """
    rospy.init_node("face_localizer", anonymous=True)
    face_finder = FaceLocalizer()
    rospy.Timer(rospy.Duration(0.3), lambda _: face_finder.find_faces())
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
