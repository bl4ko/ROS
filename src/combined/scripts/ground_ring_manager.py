"""
This class is responsible for managing the ring detector.
"""

import threading
from typing import List
import rospy

from combined.msg import DetectedRings, UniqueRingCoords


class GroundRingManager:
    """
    This class is responsible for managing the ring detector.
    """

    def __init__(self):
        self.detected_ground_rings_subscriber = rospy.Subscriber(
            "/detected_ground_ring_coords", DetectedRings, self.ground_ring_callback
        )

        self.detected_ground_rings: List[UniqueRingCoords] = []
        self.detected_ground_rings_lock = threading.Lock()

    def detection_count(self) -> int:
        """
        Returns the number of detected rings.
        """
        return len(self.detected_ground_rings)

    def ground_ring_callback(self, msg: DetectedRings):
        """
        Callback function for the ring subscriber. Stores
        detected rings in a thread-safe manner.

        Args:
            msg (DetectedRings): The message containing the detected rings.
        """
        with self.detected_ground_rings_lock:
            self.detected_ground_rings = msg.array
