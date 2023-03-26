#!/usr/bin/env python

from __future__ import print_function

import sys
import random
import rospy
from ex1.srv import AddTenInts

def add_ten_ints_client(*args):
    rospy.wait_for_service('add_ten_ints')
    try:
        add_ten_ints = rospy.ServiceProxy('add_ten_ints', AddTenInts)
        resp1 = add_ten_ints(*args)
        return resp1.sum
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def usage():
    return "%s [x y]"%sys.argv[0]

if __name__ == "__main__":
    for i in range(10):
        array = [random.randint(0, 100) for _ in range(10)]
        res = add_ten_ints_client(array)
        print('Array: %s, Sum: %s' % (array, res))