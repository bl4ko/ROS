"""
This class is responsible for managing the ring detector.
"""

import threading
import rospy

from combined.msg import DetectedFaces


class FaceManager:
    """
    This class is responsible for managing the ring detector.
    """

    def __init__(self):
        self.detected_faces_subscriber = rospy.Subscriber(
            "/detected_faces", DetectedFaces, self.faces_callback
        )
        self.detected_faces = []
        self.detected_faces_lock = threading.Lock()

    def detection_count(self) -> int:
        """
        Returns the number of detected faces.
        """
        return len(self.detected_faces)

    def faces_callback(self, msg: DetectedFaces) -> None:
        """
        Callback function for the faces subscriber. Stores
        detected faces in a thread-safe manner.

        Args:
            msg (DetectedFaces): The message containing the detected faces.
        """
        with self.detected_faces_lock:
            # if the array is smaller than before update only the existing faces

            # join the 2 arrays based on group_id
            # Create a dictionary of existing faces with group_id as key
            existing_faces_dict = {face.group_id: face for face in self.detected_faces}

            # Create a dictionary of new faces with group_id as key
            new_faces_dict = {face.group_id: face for face in msg.array}

            # Update existing_faces_dict with new_faces_dict. This will overwrite
            # existing entries with the same group_id and add new entries with new group_ids.
            existing_faces_dict.update(new_faces_dict)

            # Convert the updated dictionary back to a list
            self.detected_faces = list(existing_faces_dict.values())
