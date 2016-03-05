/*
 *  File: cone_detector.h
 *  Declaration of abstracted class for cone detection.
 *
 *  Created by Zhou Lubing.
 *  developed for TechX Challenge 2013.
 *  T-Mobile team, Nanyang Technological University
 *
 */

#ifndef CONE_DETECTOR_HPP
#define CONE_DETECTOR_HPP

#include "util.h"
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

#include <stdio.h>

#define DEBUG 1
#define AREA_FILTER 1
#define COLOR_FILTER 1
#define ENTRANCE_CONE_DEBUG 0

enum ConeType { ENTRANCE, STAIRCASE };

class ConeDetector {
  public:
    ConeDetector(ros::NodeHandle& _nh, ConeType cone_type, string projDir);
    ~ConeDetector(){};

    virtual vector<Vec3f> locate(Mat& img, Mat& xyz, float scale, 
        Mat& img_display) = 0;

    void resetColor(const Mat& color_template);
    void saveConfig() const;
    void setFocalLenImageCenter(double foc_len, double c_x, double c_y);
    void publish(vector<Vec3f>& cone_location) const;

  protected:
    ros::Publisher pub;
    ros::NodeHandle nh;

    ConeType type;
    int color_type;
    double focal_len, x_origin, y_origin;
    std::string proj_dir, config_path, msg_name, clsf_path, config_file;
    Mat colors, band;
    CascadeClassifier clsf;

    Mat inColorRanges(const Mat& src) const;
    Mat colorDistance(const Mat& src) const;
};

#endif //CONE_DETECTOR_HPP
