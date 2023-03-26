cmake_minimum_required(VERSION 3.0.2)
project(ex1)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

add_message_files(
  FILES
  Num.msg
  CustomMessage.msg
)

add_service_files(
  FILES
  AddTenInts.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS src/publisher.py src/subscriber.py src/add_ten_ints_client.py src/add_ten_ints_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

