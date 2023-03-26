
import rospy
from exercise2.srv import TurtleMove, TurtleMoveResponse
from geometry_msgs.msg import Twist
import random
from geometry_msgs.msg import Twist


def no_move(step):
    twist = Twist()
    twist.linear.x = 0.0
    twist.angular.z = 0.0
    return twist

def rectangle_movement(step):
    twist = Twist()
    twist.linear.x = 1.0
    step = step % 20
    if step % 5 == 0:
        twist.linear.x = 0
        twist.angular.z = 1.57   #(90 / 360) * 2 * 3.14
    return twist 

def circle_movement(step):
    twist = Twist()
    twist.linear.x = 1.0
    twist.angular.z = 0.4
    return twist


def triangle_movement(step):
    twist = Twist()
    twist.linear.x = 1.0
    step = step % 20
    if step % 3 == 0:
        twist.linear.x = 0
        twist.angular.z = 2.093333333
    return twist 
    

def random_movement(step):
    twist = Twist()
    # Set linear and angular velocities to random values
    twist.linear.x = random.uniform(-1.0, 1.0)
    twist.angular.z = random.uniform(-1.57, 1.57)
    return twist

def handle_request(req):
    #do something
    type_of_move = req.move_type 
    duration = req.duration
    #"circle", "rectangle", "triangle" or "random"
    step = 0.0
    r = rospy.Rate(1)
    twist = no_move(step)
    while duration > 0:
        rospy.loginfo("Moving for " + str(duration) + " more seconds")
        rospy.loginfo("Moving in a " + type_of_move + " pattern")
        if type_of_move == "circle":
            twist = circle_movement(step)
        elif type_of_move == "rectangle":
            twist = rectangle_movement(step)
        elif type_of_move == "triangle":
            twist = triangle_movement(step)
        elif type_of_move == "random":
            twist = random_movement(step)
        else:
            twist = no_move(step)
        pub.publish(twist)
        step = step + 1.0
        duration = duration - 1
        r.sleep()
    return TurtleMoveResponse("Done + " + type_of_move)


if __name__ == "__main__":
    rospy.init_node('move')
    service = rospy.Service('move', TurtleMove, handle_request)
    pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size = 1000)
    rospy.spin()