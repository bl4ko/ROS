cmake_minimum_required(VERSION 2.8.3)
project(combined)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	cv_bridge
	sensor_msgs
	visualization_msgs
	tf2_ros
	nav_msgs
	std_msgs
	geometry_msgs
	message_generation
)

add_message_files(DIRECTORY msg
  FILES
  UniqueFaceCoords.msg
  DetectedFaces.msg
  UniqueRingCoords.msg
  DetectedRings.msg
  CylinderGreetInstructions.msg
)

generate_messages(DEPENDENCIES std_msgs geometry_msgs sensor_msgs)

find_package( OpenCV REQUIRED )

find_package( PCL REQUIRED )

catkin_package()

add_executable(cylinder_detector src/cylinder_detector.cpp)

target_link_libraries(cylinder_detector ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

include_directories(
    ${catkin_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)
#add_executable(map_goals src/map_goals.cpp)
#target_link_libraries(map_goals ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
#catkin_install_python(PROGRAMS scripts/detect_rings DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
