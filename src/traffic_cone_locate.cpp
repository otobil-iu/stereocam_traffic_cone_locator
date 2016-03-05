/*
 *  File: traffic_cone_locate.cpp
 *  Main function to detect and locate traffic cones for door entrance.
 *
 *  Created by Zhou Lubing.
 *  developed for TechX Challenge 2013.
 *  T-Mobile team, Nanyang Technological University
 *
 */

#include "entrance_cone_detector.h"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include "std_msgs/Float64MultiArray.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <boost/thread.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>

using namespace std;
using namespace cv;

#define ENTRANCE_CONE 1

string proj_dir;
EntranceConeDetector* enter_cone;
cv::Mat img_display, color_set_img, bgr_roi, disp_roi, xyz_roi, Q;
cv_bridge::CvImagePtr cv_ptr_left, cv_ptr_disp;
Point origin;
Rect roi, box;
double center_u, center_v;
float base_line, focal_length;

bool selecting_flag = false, reset_color_flag = false;
bool cam_info_sub_flag = false;

Mat disparity2xyz(Point2f camCenter, float baseLine, float focalLength, 
    Mat & disparity);
void camInfoCallback(const sensor_msgs::CameraInfoConstPtr& l_info_msg,
    const sensor_msgs::CameraInfoConstPtr& r_info_msg);
void imageCallback (const sensor_msgs::ImageConstPtr& left_img_msg, 
    const sensor_msgs::ImageConstPtr& disp_img_msg);
void onMouse(int event,int x,int y,int,void*);
void inputKeyHandler(char ch);

int main(int argc, char** argv) {
  proj_dir = ros::package::getPath("stereocam_traffic_cone_locator");
  namedWindow("traffic_cone_locate", CV_WINDOW_NORMAL);
  setMouseCallback("traffic_cone_locate", onMouse,0);
  ros::init(argc, argv, "traffic_cone_locate");
  ros::NodeHandle nh;
  enter_cone = new EntranceConeDetector(nh, ENTRANCE, proj_dir);

  // Subscribe CameraInfo
  message_filters::Subscriber<sensor_msgs::CameraInfo> left_info_sub, 
    right_info_sub;

  left_info_sub.subscribe(nh, "/left_rectified/camera_info", 1);
  right_info_sub.subscribe(nh, "/right_rectified/camera_info", 1);

  typedef message_filters::sync_policies::ExactTime<sensor_msgs::CameraInfo, 
          sensor_msgs::CameraInfo> MyCamInfoSyncPolicy;
  message_filters::Synchronizer<MyCamInfoSyncPolicy> 
    camInfoSync(MyCamInfoSyncPolicy(1), left_info_sub, right_info_sub);

  camInfoSync.registerCallback(boost::bind(camInfoCallback,  _1, _2));

  // Subscribe left image and corresponding disparity image.
  message_filters::Subscriber<sensor_msgs::Image> image1_sub(nh, 
      "/left_rectified/rgb_rectified", 1);
  message_filters::Subscriber<sensor_msgs::Image> image2_sub(nh, 
      "/camera/disparity", 1);

  typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, 
          sensor_msgs::Image> MyImgSyncPolicy;
  message_filters::Synchronizer<MyImgSyncPolicy> sync(MyImgSyncPolicy(10), 
      image1_sub, image2_sub);
  sync.registerCallback(boost::bind(&imageCallback, _1, _2));

  while (ros::ok()) {
    char ch = (char)waitKey(5);
    inputKeyHandler(ch);
    if (reset_color_flag && !color_set_img.empty()) {
      if (!selecting_flag)
        rectangle(color_set_img, box, Scalar(255,0,0), 1, 8, 0);
      imshow("traffic_cone_locate", color_set_img);
    }
    ros::spinOnce();
  }
  return 0;
}


Mat disparity2xyz(Point2f camCenter, float baseLine, float focalLength, 
    Mat & disparity) {
  Mat xyz_map(disparity.rows, disparity.cols, CV_32FC3, Scalar(0, 0, 0));
  float const_val = focalLength * baseLine;
  for (int r = 0; r < disparity.rows; r++) {
    for (int c = 0; c < disparity.cols; c++) {
      Point pt(c, r);
      float d = disparity.at<float>(pt);
      if (d <= 1) continue;
      float tmpZ = const_val / d;
      if (tmpZ > 10.0) continue;
      Point2f sc = (camCenter - Point2f(c, r)) * (tmpZ / focalLength);
      Vec3f xyz_pixel(sc.x, sc.y, tmpZ);
      xyz_map.at<Vec3f>(pt) = Vec3f(sc.x, sc.y, tmpZ);
    }
  }
  return xyz_map;
}

void camInfoCallback(const sensor_msgs::CameraInfoConstPtr& l_info_msg,
    const sensor_msgs::CameraInfoConstPtr& r_info_msg) {
  if (cam_info_sub_flag) return;
  base_line = -r_info_msg->P[3] / r_info_msg->P[0];
  focal_length = (float)l_info_msg->P[0];
  roi.x = l_info_msg->roi.x_offset;
  roi.y = l_info_msg->roi.y_offset;
  roi.width = l_info_msg->roi.width;
  roi.height = l_info_msg->roi.height;
  center_u = l_info_msg->P[2] - roi.x;
  center_v = l_info_msg->P[6] - roi.y;

#if (ENTRANCE_CONE)
  enter_cone->setFocalLenImageCenter(focal_length, center_u, center_v);
#endif
  cam_info_sub_flag = true;
}

//callback to subscribe left rbg image and disparity image
void imageCallback(const sensor_msgs::ImageConstPtr& left_img_msg, 
    const sensor_msgs::ImageConstPtr& disp_img_msg) {
  double t = (double)getTickCount();
  if (!cam_info_sub_flag) return;
  if (reset_color_flag) return;
  Mat img;
  try {
    cv_ptr_left = cv_bridge::toCvCopy(left_img_msg, "bgr8");
    cv_ptr_disp= cv_bridge::toCvCopy(disp_img_msg, "32FC1");
    img = cv_ptr_left->image;
    disp_roi = cv_ptr_disp->image;
  } 
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  bgr_roi = img(roi);
  cv::Point2f camCenter(center_u, center_v);
  xyz_roi =  disparity2xyz(camCenter, base_line, focal_length, disp_roi);
  bgr_roi.copyTo(img_display);

  vector< Vec3f > cones;
  cones = enter_cone->locate(bgr_roi, xyz_roi, 0.6, img_display);
  enter_cone->publish(cones);

  imshow("traffic_cone_locate", img_display);
  t = getTickCount() - t;
  //cout<< "time= " << t * 1000.0 / getTickFrequency() << "ms" << endl;
}

void onMouse(int event,int x,int y,int,void*) {
  if (selecting_flag) {
    box.x = MIN(origin.x, x);
    box.y = MIN(origin.y, y);
    box.width = abs(x - origin.x);
    box.height = abs(y - origin.y);
    box &= Rect(0, 0, bgr_roi.cols, bgr_roi.rows);
  } if (event == CV_EVENT_LBUTTONDOWN) {
    selecting_flag = true;
    origin = Point(x,y);
    box = Rect(x, y, 0, 0);
  } else if(event == CV_EVENT_LBUTTONUP) {
    selecting_flag = false;
  }
}

void inputKeyHandler(char ch) {
  if (ch > 0) {
    switch (ch) {
      case 'q':
        exit(1);
        break;
      case 's':
        // start set and select rect to collect color sample.
        cout << "Start setting, pls select the rect"<<endl;
        color_set_img = bgr_roi.clone();
        reset_color_flag = true;
        break;
      case 'e'://confirm the reset operation for entrance cone 
        cout << "Save colors for entrance cone, release resetting hold!"<<endl;
        reset_color_flag = false;
        enter_cone->resetColor(bgr_roi(box));
        enter_cone->saveConfig();
        box = Rect(-1, -1, 0, 0);
        break;
      case 'c':
        cout << "Cancel color setting"<<endl;
        reset_color_flag = false;
        box = Rect(-1, -1, 0, 0);
        break;
      default:
        ;
    }
  }
}

