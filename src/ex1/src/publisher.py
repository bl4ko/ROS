#!/usr/bin/env python

import rospy
from ex1.msg import CustomMessage


def publisher():
    pub = rospy.Publisher("custom_topic", CustomMessage, queue_size=10)
    rospy.init_node("custom_publisher", anonymous=True)
    msg = CustomMessage(message="Hello World", sequence_id=0)

    rate = rospy.Rate(1)  # 1hz

    while not rospy.is_shutdown():
        rospy.loginfo("Publisher is sending: %s[id=%d]", msg.message, msg.sequence_id)
        pub.publish(msg)
        msg.sequence_id += 1
        rate.sleep()


if __name__ == "__main__":
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
