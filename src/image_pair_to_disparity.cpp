/*
 *  File: image_pair_to_disparity.cpp
 *  Use BM or SGBM method to generate disparity image.
 *
 *  Created by Zhou Lubing.
 *  developed for TechX Challenge 2013.
 *  T-Mobile team, Nanyang Technological University
 *
 */

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Float64MultiArray.h>

#include <pcl/ros/conversions.h>
#include <pcl/io/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>

using namespace std;
using namespace cv;

#define DEBUG 0
#define USE_SGBM 0

long int pub_counter = 0;
typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, 
        sensor_msgs::Image> MyImageSyncPolicy;
typedef message_filters::sync_policies::ExactTime<sensor_msgs::CameraInfo, 
        sensor_msgs::CameraInfo> MyCamInfoSyncPolicy;

cv_bridge::CvImagePtr cv_ptr_left, cv_ptr_right;
cv::Mat left_img, right_img;
cv::Mat Q, disp, dispReal;

float base_line_, focal_length_; //camera baseline, focal length
double cu_, cv_; // Image center
Rect roi_;

#if (DEBUG)
Mat disp8;
#endif 

Size img_size;
StereoSGBM sgbm;
StereoBM bm;
cv_bridge::CvImage m_cvi;

image_transport::Publisher m_disparity_pub;
sensor_msgs::Image m_left_msg;
sensor_msgs::Image m_right_msg;
sensor_msgs::Image m_disparity_msg;
double start=0.; 
int frame_counter = 0;

inline double getSecOfNow() {
  timeval tmpCurTime;
  gettimeofday(&tmpCurTime, NULL);
  return ((double)tmpCurTime.tv_usec / 1000000.) + ((double)tmpCurTime.tv_sec);
}

void qMatCallback(const std_msgs::Float64MultiArray::ConstPtr& msg) {
  if (!Q.empty()) return;
  Q = (Mat_<double>(4,4) << msg->data[0], msg->data[1], msg->data[2], msg->data[3],
      msg->data[4], msg->data[5], msg->data[6], msg->data[7],
      msg->data[8], msg->data[9], msg->data[10], msg->data[11],
      msg->data[12], msg->data[13], msg->data[14], msg->data[15]);
  std::cout << Q << std::endl;
  //q_sub.shutdown();
}

void camInfoCallback(const sensor_msgs::CameraInfoConstPtr& l_info_msg,
    const sensor_msgs::CameraInfoConstPtr& r_info_msg) {
  base_line_ = -r_info_msg->P[3] / r_info_msg->P[0];
  focal_length_ = (float)l_info_msg->P[0];
  cu_ = l_info_msg->P[2];
  cv_ = l_info_msg->P[6];
  roi_.x = l_info_msg->roi.x_offset;
  roi_.y = l_info_msg->roi.y_offset;
  roi_.width = l_info_msg->roi.width;
  roi_.height = l_info_msg->roi.height;
  //left_info_sub.shutdown();
  //right_info_sub.shutdown();
}

void imageCallback (const sensor_msgs::ImageConstPtr& left_img_msg, 
    const sensor_msgs::ImageConstPtr& right_img_msg) {
  try {
    cv_ptr_left = cv_bridge::toCvCopy(left_img_msg, "bgr8");
    cv_ptr_right = cv_bridge::toCvCopy(right_img_msg, "bgr8");
    left_img = cv_ptr_left->image;
    //left_sync_img = cv_ptr_left->image;
    right_img = cv_ptr_right->image;
  }
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  Mat left_img_roi = left_img(roi_);
  Mat right_img_roi = right_img(roi_);
  if (0 == frame_counter) {
    start = getSecOfNow();
  }

  if (left_img_roi.rows < 10 || left_img_roi.cols < 10 || 
      right_img_roi.rows < 10 || right_img_roi.cols < 10) return;

#if (USE_SGBM)
  sgbm(left_img_roi, right_img_roi, disp);
  disp.convertTo(dispReal, CV_32F, 1.0/16.0);
#else  
  Mat left_img_roi_grey, right_img_roi_grey;
  cvtColor(left_img_roi, left_img_roi_grey, CV_RGB2GRAY);
  cvtColor(right_img_roi, right_img_roi_grey, CV_RGB2GRAY);
  bm(left_img_roi_grey, right_img_roi_grey, disp, CV_32F);
#endif 

#if (DEBUG)
  disp.convertTo(disp8, CV_8U);
  imshow("left", left_img_roi);
  imshow("right", right_img_roi);
  imshow("disparity", disp8);
  char key_pressed = (char)cv::waitKey(10);
#endif 

  //publish disparity image
  ros::Time time = left_img_msg->header.stamp;
  m_cvi.header.stamp = time;
  m_cvi.header.frame_id = "/camera";
  m_cvi.encoding = "32FC1";
#if(USE_SGBM)
  m_cvi.image = dispReal;
#else
  m_cvi.image = disp;
#endif

  m_cvi.toImageMsg(m_disparity_msg);
  m_disparity_pub.publish(m_disparity_msg);
  frame_counter++;

  if (frame_counter == 20) {
    double time_elapsed = getSecOfNow() - start;
    printf("Publisheing messages in %lf hz\n", frame_counter / time_elapsed);
    frame_counter = 0;
  }
}

int main(int argc, char **argv) {
  img_size.width = 640;
  img_size.height = 480;
  int num_disparities = ((img_size.width/8) + 15) & -16; 

#if (USE_SGBM)
  int cn = left_img.channels();
  sgbm.preFilterCap = 63;
  sgbm.SADWindowSize = 3;
  sgbm.P1 = 8 * cn * sgbm.SADWindowSize * sgbm.SADWindowSize;
  sgbm.P2 = 32 * cn * sgbm.SADWindowSize * sgbm.SADWindowSize;
  sgbm.minDisparity = 0;
  sgbm.numberOfDisparities = num_disparities;
  sgbm.uniquenessRatio = 10;
  sgbm.speckleWindowSize =100;
  sgbm.speckleRange = 32;
  sgbm.disp12MaxDiff = 1;
  sgbm.fullDP = false;  
#else // USE_SGBM
  bm.state->preFilterCap = 31;
  bm.state->minDisparity = 0;
  bm.state->numberOfDisparities = num_disparities;
  bm.state->textureThreshold = 10;
  bm.state->uniquenessRatio = 15;
  bm.state->speckleWindowSize = 100;
  bm.state->speckleRange = 32;
  bm.state->disp12MaxDiff = 1;
#endif

#if(DEBUG)
  namedWindow("disparity");
  namedWindow("left");
#endif

  ros::init(argc, argv, "image_pair_to_disparity");
  ros::NodeHandle nh;

  message_filters::Subscriber<sensor_msgs::CameraInfo> left_info_sub, 
    right_info_sub;

  left_info_sub.subscribe(nh, "/left_rectified/camera_info", 1);
  right_info_sub.subscribe(nh, "/right_rectified/camera_info", 1);
  image_transport::ImageTransport m_it(nh);
  m_disparity_pub = m_it.advertise("/camera/disparity", 1);

  message_filters::Subscriber<sensor_msgs::Image> left_image_sub(nh, 
      "/left_rectified/rgb_rectified", 1);
  message_filters::Subscriber<sensor_msgs::Image> right_image_sub(nh, 
      "/right_rectified/rgb_rectified", 1);
  message_filters::Synchronizer<MyImageSyncPolicy> 
    imageSync(MyImageSyncPolicy(1), left_image_sub, right_image_sub);
  message_filters::Synchronizer<MyCamInfoSyncPolicy> 
    camInfoSync(MyCamInfoSyncPolicy(1), left_info_sub, right_info_sub);

  imageSync.registerCallback(boost::bind(imageCallback,  _1, _2));
  camInfoSync.registerCallback(boost::bind(camInfoCallback,  _1, _2));
  ros::spin();
}
