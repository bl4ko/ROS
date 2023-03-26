cmake_minimum_required(VERSION 3.0.2)
project(exercise2)

find_package(catkin REQUIRED COMPONENTS rospy roscpp geometry_msgs message_generation std_msgs)

add_service_files(
    FILES
    TurtleMove.srv
)

generate_messages(
    DEPENDENCIES
    std_msgs
)

catkin_package(
   CATKIN_DEPENDS message_runtime rospy std_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS src/turtle_move.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

