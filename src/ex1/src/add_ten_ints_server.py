#!/usr/bin/env python

from __future__ import print_function

from ex1.srv import AddTenInts,AddTenIntsResponse
import rospy

def handle_add_ten_ints(req):
    print('Service received: ')
    print(*req.data)
    return AddTenIntsResponse(sum=sum(req.data))

def add_two_ints_server():
    rospy.init_node('add_ten_ints_server')
    # Declare a service named add_ten_ints, all requests are passed
    # to handle_add_ten_ints function.
    s = rospy.Service('add_ten_ints', AddTenInts, handle_add_ten_ints)
    print("Ready to add ten ints.")
    rospy.spin()

if __name__ == "__main__":
    add_two_ints_server()