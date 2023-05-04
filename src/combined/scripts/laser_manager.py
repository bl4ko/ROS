#!/usr/bin/python3

"""
Module for managing the laser scan data.
"""

import rospy
from sensor_msgs.msg import LaserScan
from laser_geometry import LaserProjection


class LaserManager:
    """
    Class for managing the laser scan data.
    """

    def __init__(self):
        # distance to closest obstacle
        self.distance_to_obstacle = 100
        self.too_close = False
        self.laser_projection = LaserProjection()
        # self.pcPub = rospy.Publisher("/laserPointCloud", pc2, queue_size=1)
        self.laser_subscriber = rospy.Subscriber("/scan", LaserScan, self.laser_callback)

        rospy.loginfo("Laser init complete.")

    def laser_callback(self, data: LaserScan) -> None:
        """
        Callback for laser scan data.

        Args:
            data (LaserScan): Laser scan data.
        """
        # cloud_out = self.laserProj.projectLaser(data)
        # self.pcPub.publish(cloud_out)

        # print(str(self.too_close))

        # get distance to closest obstacle
        self.distance_to_obstacle = min(data.ranges)
        # self.distance_to_obstacle = data.ranges[360]
        # print(str(self.distance_to_obstacle))

        # rospy.loginfo("Distance to closest obstacle: " + str(self.distance_to_obstacle))

        if self.distance_to_obstacle > 0.2:
            self.too_close = False

        else:
            # rospy.loginfo("Distance to closest obstacle: " + str(self.distance_to_obstacle))
            self.too_close = True

        # print(str(self.too_close))

    def is_too_close_to_obstacle(self):
        """
        Returns if the robot is too close to an obstacle.

        Returns:
            bool: True if the robot is too close to an obstacle, False otherwise.
        """
        return self.too_close


if __name__ == "__main__":
    rospy.init_node("laser_manager")
    lm = LaserManager()

    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        r.sleep()
