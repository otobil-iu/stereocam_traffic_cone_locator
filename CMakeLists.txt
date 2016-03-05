cmake_minimum_required(VERSION 2.8.3)
project(stereocam_traffic_cone_locator)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
)

find_package(PCL REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR} 
  ${PROJECT_SOURCE_DIR}/include 
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_executable(image_pair_to_disparity src/image_pair_to_disparity.cpp)
target_link_libraries(image_pair_to_disparity ${PCL_LIBRARIES} ${Boost_LIBRARIES} 
  ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

add_executable(traffic_cone_locate src/traffic_cone_locate.cpp
  src/cone_detector.cpp src/entrance_cone_detector.cpp src/util.cpp)
target_link_libraries(traffic_cone_locate ${PCL_LIBRARIES} ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
