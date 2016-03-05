/*
 *  File: cone_detector.cpp
 *  Parent class for cone detector, two different types of cone makers in 
 *  TechX: door entrance and staircase.
 *
 *  Created by Zhou Lubing.
 *  developed for TechX Challenge 2013.
 *  T-Mobile team, Nanyang Technological University
 *
 */

#include "cone_detector.h"

ConeDetector::ConeDetector(ros::NodeHandle& _nh, ConeType cone_type, 
    string projDir) {
  proj_dir = projDir;
  type = cone_type;
  nh = _nh;

  if (type == ENTRANCE)
    config_path = "/rst/config/entrance_cone_config.xml";
  else if ( type == STAIRCASE )
    config_path = "/rst/config/staircase_cone_config.xml";

  config_file = proj_dir + config_path;
  FileStorage fs;
  if (!fs.open(config_file, FileStorage::READ)) {
    fprintf(stderr, " Error:  config file: %s can not be opened!\n", 
        config_file.c_str() );
    exit(0);
  }

  clsf_path = (string)fs["clsfPath"];
  string clsf_dir = proj_dir + clsf_path;

  if (!clsf.load(clsf_dir)) {
    fprintf(stderr, "Error: classifier: %s can not be loaded\n", 
        clsf_dir.c_str());
    exit(0);
  }

  msg_name = (string)fs["msgName"];
  fs["colors"] >> colors;
  fs["band"] >> band;
  fs.release();

  colors = Mat_<int>(colors);
  band = Mat_<int>(band);
  pub = nh.advertise<geometry_msgs::Twist>(msg_name, 1);
}

void ConeDetector::setFocalLenImageCenter(double foc_len, double c_x, 
    double c_y) {
  this->focal_len = foc_len;
  this->x_origin = c_x;
  this->y_origin = c_y;
}

void ConeDetector::publish(vector<Vec3f>& cone_location) const {
  int sz = (int)cone_location.size();
  if(sz > 2) sz = 2;
  geometry_msgs::Twist cone_msg;

  cone_msg.linear.x = 0;
  cone_msg.linear.y = 0;
  cone_msg.linear.z = 0;
  cone_msg.angular.x = 0;
  cone_msg.angular.y = 0;
  cone_msg.angular.z = 0;

  if (sz == 1) {
    Vec3f v = cone_location[0];
    cone_msg.linear.x = v[0];
    cone_msg.linear.y = v[1];
    cone_msg.linear.z = 1;
  } else if (sz == 2) {
    Vec3f v0, v1;
    v0 = cone_location[0];
    v1 = cone_location[1];
    if (v0[0] > v1[0]) {
      v0 = cone_location[1];
      v1 = cone_location[0];
    }
    cone_msg.linear.x = v0[0];
    cone_msg.linear.y = v0[1];
    cone_msg.linear.z = 1;

    cone_msg.angular.x = v1[0];
    cone_msg.angular.y = v1[1];
    cone_msg.angular.z = 1;
  }
  pub.publish(cone_msg);
}

Mat ConeDetector::inColorRanges(const Mat& src) const {
  CV_Assert(colors.rows > 0 && colors.cols == 3);
  CV_Assert(band.rows > 0 && band.cols == 3);

  Mat dst(src.size(), CV_8U, Scalar(255));
  Mat bin_img(src.size(), CV_8U, Scalar(0));

  for (int k = 0; k < colors.rows; k++) {
    Scalar color_center, bands;
    bands[0] = band.at<int>(k,0);
    bands[1] = band.at<int>(k,1);
    bands[2] = band.at<int>(k,2);
    color_center[0] = colors.at<int>(k, 0);
    color_center[1] = colors.at<int>(k, 1);
    color_center[2] = colors.at<int>(k, 2);
    Scalar lowb = color_center - bands;
    Scalar upb = color_center + bands;

    Mat tmp_img;
    inRange(src, lowb, upb, tmp_img);
    bin_img = bin_img | tmp_img;
  }

  dst = dst & bin_img;
  return dst;
}

Mat ConeDetector::colorDistance(const Mat& src) const {
  CV_Assert( colors.rows > 0 && colors.cols == 3);
  CV_Assert( band.rows > 0 && band.cols == 3);

  int ncolors = colors.rows;
  Mat dist(src.size(), CV_32F, Scalar(10000));

  for (int k = 0; k < ncolors; k++) {
    Scalar color_center;
    color_center[0] = colors.at<int>(k, 0);
    color_center[1] = colors.at<int>(k, 1);
    color_center[2] = colors.at<int>(k, 2);

    Mat diff_img, planes[3];
    absdiff(src, color_center, diff_img);
    Mat diff_sum(src.size(), CV_32F, Scalar(0));
    cv::split(diff_img, planes);

    for (int c = 0; c < 3; c++) {
      planes[c] = Mat_<float>(planes[c]);
      diff_sum += planes[c];
    }
    cv::min(dist, diff_sum, dist);
  }
  return dist;
}

void ConeDetector::resetColor(const Mat& color_template) {
  CV_Assert(color_template.channels() == 3);

  Mat changed_color;
  changed_color = colorConvert(color_template, color_type);

  Mat planes[3];
  cv::split(changed_color, planes);
  Scalar avg[3], sdv[3];

  for (int k = 0; k < 3; k++)
    cv::meanStdDev(planes[k], avg[k], sdv[k]);

  int c0, c1, c2;
  c0 = cvRound(avg[0][0]);
  c1 = cvRound(avg[1][0]);
  c2 = cvRound(avg[2][0]);

  int d0, d1, d2;
  float coef = 3.0;
  d0 = cvRound(coef * sdv[0][0]);
  d1 = cvRound(coef * sdv[1][0]);
  d2 = cvRound(coef * sdv[2][0]);
  d0 = std::min(d0, 50);
  d1 = std::min(d1, 40);
  d2 = std::min(d2, 40);

  int val = band.at<int>(0, 0);
  if (colors.rows == 1 && val == 0) {
    colors.at<int>(0, 0) = c0;
    colors.at<int>(0, 1) = c1;
    colors.at<int>(0, 2) = c2;
    band.at<int>(0, 0) = d0;
    band.at<int>(0, 1) = d1;
    band.at<int>(0, 2) = d2;
  } else if (colors.rows >= 1) {
    Mat tmp_color = (Mat_<int>(1,3) << c0, c1, c2);
    colors.push_back( tmp_color );
    Mat tmp_band = (Mat_<int>(1,3) << d0, d1, d2);
    band.push_back(tmp_band);
  }
}

void ConeDetector::saveConfig() const {
  FileStorage fs;
  if (!fs.open(config_file, FileStorage::WRITE)) {
    fprintf(stderr, " Error:  cannot write to config file: %s!\n", 
        config_file.c_str() );
    exit(0);
  }

  fs << "clsfPath" << clsf_path;
  fs << "msgName" << msg_name;
  fs << "colors" << colors;
  fs << "band" << band;
  fs.release();
}
