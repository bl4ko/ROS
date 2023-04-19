cmake_minimum_required(VERSION 2.8.3)
project(course_project)

## Find system dependencies
find_package(Boost REQUIRED COMPONENTS system filesystem thread)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	rospy
	cv_bridge
	sensor_msgs
	visualization_msgs
	tf2_ros
	nav_msgs
	std_msgs
	geometry_msgs
	message_generation

	dynamic_reconfigure
	genmsg
	nodelet
	nodelet_topic_tools
	pcl_conversions
	pcl_msgs
	pluginlib
	rosbag
	rosconsole
	roslib
	tf2_geometry_msgs
)

find_package( OpenCV REQUIRED )

## Add include directories
include_directories(
	${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

## Add link directories
link_directories(
  ${Boost_LIBRARY_DIRS}
  ${Eigen_LIBRARY_DIRS}
  ${PCL_LIBRARY_DIRS}
  ${catkin_LIBRARY_DIRS}
)

link_libraries(${Boost_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_message_files(DIRECTORY msg
  FILES
  UniqueFaceCoords.msg
  MyStatus.msg
  RingGreetInstructions.msg
  CylinderGreetInstructions.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  ColorSrv.srv
)

generate_messages(DEPENDENCIES std_msgs 
geometry_msgs sensor_msgs)

## Declare the catkin package
catkin_package(
  LIBRARIES
    pcl_ros_filters
    pcl_ros_io
    pcl_ros_tf
  CATKIN_DEPENDS
    dynamic_reconfigure
    pcl_conversions
    pcl_msgs
    roscpp
    sensor_msgs
    std_msgs
    geometry_msgs
	  message_runtime
  DEPENDS
    Boost
    Eigen
    PCL
)

include_directories(include ${catkin_INCLUDE_DIRS})

#include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
#add_executable(map_goals src/map_goals.cpp)
#target_link_libraries(map_goals ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable (cylinder_detector src/cylinder_detector.cpp)

#catkin_install_python(PROGRAMS scripts/detect_rings DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

add_dependencies(cylinder_detector ${${PROJECT_NAME}_EXPORTED_TARGETS})

#add_dependencies(ColorSrv ${${PROJECT_NAME}_EXPORTED_TARGETS})
