#!/usr/bin/python3

import rospy
import time
from sensor_msgs.msg import PointCloud2 as pc2
from sensor_msgs.msg import LaserScan
from laser_geometry import LaserProjection

class LaserManager():
    def __init__(self):
        # distance to closest obstacle
        self.distance_to_obstacle = 100
        self.too_close = False

        self.laserProj = LaserProjection()
        #self.pcPub = rospy.Publisher("/laserPointCloud", pc2, queue_size=1)
        self.laserSub = rospy.Subscriber("/scan", LaserScan, self.laserCallback)

        rospy.loginfo("Laser init complete.")

    def laserCallback(self, data):
        #cloud_out = self.laserProj.projectLaser(data)
        #self.pcPub.publish(cloud_out)

        #print(str(self.too_close))

        # get distance to closest obstacle
        self.distance_to_obstacle = min(data.ranges)
        #self.distance_to_obstacle = data.ranges[360]
        #print(str(self.distance_to_obstacle))

        #rospy.loginfo("Distance to closest obstacle: " + str(self.distance_to_obstacle))


        
        if self.distance_to_obstacle > 0.2:
            self.too_close = False
            

        else:
            #rospy.loginfo("Distance to closest obstacle: " + str(self.distance_to_obstacle))
            self.too_close = True

        #print(str(self.too_close))


    def is_too_close_to_obstacle(self):
        return self.too_close
        


if __name__ == "__main__":
    rospy.init_node("laser_manager")
    lm = LaserManager()
    
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        r.sleep()
