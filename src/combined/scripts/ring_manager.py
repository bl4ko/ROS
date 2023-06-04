"""
This class is responsible for managing the ring detector.
"""

import threading
from typing import List, Tuple
import rospy

from utils import euclidean_distance
from combined.msg import DetectedRings, UniqueRingCoords


class RingManager:
    """
    This class is responsible for managing the ring detector.
    """

    def __init__(self):
        self.detected_rings_subscriber = rospy.Subscriber(
            "/detected_ring_coords", DetectedRings, self.ring_callback
        )

        self.detected_rings_lock = threading.Lock()
        self.detected_rings: List[UniqueRingCoords] = []

    def detection_count(self) -> int:
        """
        Returns the number of detected rings.
        """
        return len(self.detected_rings)

    def ring_callback(self, msg: DetectedRings) -> None:
        """
        Callback function for the ring subscriber. Stores
        detected rings in a thread-safe manner.

        Args:
            msg (DetectedRings): The message containing the detected rings.
        """
        with self.detected_rings_lock:
            self.detected_rings = msg.array

    def get_closest_ring(self, robot_x: float, robot_y: float) -> Tuple[UniqueRingCoords, float]:
        """
        Get closest ring to the robot position (x, y).

        Args:
            robot_x (float): The x coordinate of the robot.
            robot_y (float): The y coordinate of the robot.

        Returns:
            Tuple[UniqueRingCoords, float]: The closest ring and the distance to it.
        """
        with self.detected_rings_lock:
            closest_ring = None
            closest_distance = float("inf")
            for ring in self.detected_rings:
                distance = euclidean_distance(
                    robot_x,
                    robot_y,
                    ring.ring_pose.position.x,
                    ring.ring_pose.position.y,
                )
                if distance < closest_distance:
                    closest_distance = distance
                    closest_ring = ring

            return closest_ring, closest_distance
