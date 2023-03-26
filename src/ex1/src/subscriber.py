#!/usr/bin/env python

import rospy
from ex1.msg import CustomMessage

def callback(data):
    rospy.loginfo('Subscriber is getting: %s[id=%d]', data.message, data.sequence_id)

def subscriber():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('custom_listener', anonymous=True)
    rospy.Subscriber('custom_topic', CustomMessage, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    subscriber()
