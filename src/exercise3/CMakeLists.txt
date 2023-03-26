cmake_minimum_required(VERSION 2.8.3)
project(exercise3)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	cv_bridge
	sensor_msgs
	nav_msgs
	std_msgs
	geometry_msgs
    message_generation
	tf2
	tf2_geometry_msgs
)

find_package(OpenCV)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
add_executable(map_goals src/map_goals.cpp)
target_link_libraries(map_goals ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

catkin_package(
   CATKIN_DEPENDS message_runtime rospy std_msgs
)

catkin_install_python(PROGRAMS src/goals.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


