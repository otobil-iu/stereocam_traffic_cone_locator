# Stereo Cam Based Traffic Cone Detection & Localization for Building Entry

This package was developed for TechX Challenge 2013, when the author was 
a member in T-Mobile team, Nanyang Technological University, Singapore.  
Autonomous robots were required to enter building, where a pair of traffic 
cones were placed. This package was based on ROS and stereo camera (Bumblebee2).

## Installation
1. Install ROS (tested Ubuntu 14.04 + ROS indigo).
2. The package uses OpenCV, PCL (ROS store support by using 'rosdep').


## Usage
A ros bag 'cone.bag' is included in the 'data' folder, to test the package:
(1) Download and copy the package to your ROS workspace.
(2) $ cd /your/ros/workspace/
(3) $ catkin_make  && rospack profile
(4) $ roslaunch stereocam_traffic_cone_locator traffic_cone_locate.launch


## History

The package was developed in Ubuntu 12.04 + ROS fuerte using rosbuild. Recently it was
converted to Ubuntu 14.04 + indigo using catkin package.

## License

Please feel free to use the package for non-conmercial purpose at your own
risks.
