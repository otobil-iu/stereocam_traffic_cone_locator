/*
 *  File: entrance_cone_detector.h
 *  Declaration of entrance cone detector based on stereo cam (xb2) for TechX.
 *
 *  Created by Zhou Lubing.
 *  developed for TechX Challenge 2013.
 *  T-Mobile team, Nanyang Technological University
 *
 */

#ifndef ENTRANCE_CONE_HPP
#define ENTRANCE_CONE_HPP

#include "cone_detector.h"
#include <geometry_msgs/Twist.h>

class EntranceConeDetector : public ConeDetector {
  public:
    EntranceConeDetector(ros::NodeHandle& _nh, ConeType cone_type, 
        string proj_dir);
    ~EntranceConeDetector(){};
    vector<Vec3f> locate(Mat& img, Mat& xyz, float scale, Mat& disp_image);
};

#endif //ENTRANCE_CONE_HPP
