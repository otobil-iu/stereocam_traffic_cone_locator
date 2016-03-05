/*
 *  File: entrance_cone_detector.cpp
 *  Stereo Cam based detector to detect pair of traffic cones placed at door 
 *  entrance in TechX 2013.
 *
 *  Created by Zhou Lubing in 2013.
 *  developed for TechX Challenge 2013.
 *  T-Mobile team, Nanyang Technological University
 *
 */


#include <iostream>
#include "entrance_cone_detector.h"
using namespace std;

EntranceConeDetector::EntranceConeDetector(ros::NodeHandle &_nh, ConeType cone_type, 
    string proj_dir) : ConeDetector(_nh, cone_type, proj_dir) {
  color_type = CV_BGR2Igr; //CV_BGR2HSV
}

vector<Vec3f> EntranceConeDetector::locate(Mat& img, Mat& xyz, float scale, 
    Mat& img_display) {
  vector<Vec3f> targets;
  float color_dist_thresh = 16.0;
  int wd = img.cols, ht = img.rows;

  // Detection is applied in a downsampled low-res image for speed concern.
  Size sml_sz(cvRound(wd * scale), cvRound(ht * scale));
  Mat big_gray, xyz_parts[3];
  Mat sml_bgr_img, sml_color_img, sml_color_dist, sml_dist_bin, sml_y, sml_z;
  cv::cvtColor(img, big_gray, CV_BGR2GRAY);
  split(xyz, xyz_parts);
  resize(xyz_parts[1], sml_y, sml_sz, 0, 0, INTER_NEAREST);
  resize(xyz_parts[2], sml_z, sml_sz, 0, 0, INTER_NEAREST);
  resize(img, sml_bgr_img, sml_sz);

  // Convert color to Igr space.
  sml_color_img = colorConvert(sml_bgr_img, color_type);
  // Compute color similarity or saliency map.
  sml_color_dist = colorDistance(sml_color_img);
  sml_dist_bin = (sml_color_dist < color_dist_thresh);

  dilate(sml_dist_bin, sml_dist_bin, Mat(), Point(-1,-1), 3);
  erode(sml_dist_bin, sml_dist_bin, Mat(), Point(-1,-1), 3);

  // Cone distance in z-direction in range (1.2m~10m), height < 2m.
  Mat yz_mask = (sml_z > 1.2) & (sml_z < 10) & (sml_y < 2);

#if (ENTRANCE_CONE_DEBUG)
  imshow("entrance_binary", sml_dist_bin);
#endif

  vector<vector<Point> > contours, contours1;
  vector<Rect> fnl_rects;

  // Perform shape, geometry analysis, boosted classifier to filter out false 
  // alarms step by step.  Candidate color, area, upright pose, height, 
  // ratio of height over width, classifier, 2D/3D fusion.
  findContours(sml_dist_bin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  for (unsigned int k = 0; k < contours.size(); k++) {
    double area = cv::contourArea(Mat(contours[k]));
    if (area < 50 * scale * scale || area > 20000 * scale * scale) continue;

    RotatedRect rRect = minAreaRect(Mat(contours[k]));
    float rRectWd = rRect.size.width;
    float rRectHt = rRect.size.height;
    float ang, ratio;
    if (rRectWd > rRectHt) {
      ang = 90 - std::abs(rRect.angle);
      ratio = rRectWd / rRectHt;
    } else {
      ang = std::abs(rRect.angle);
      ratio = rRectHt / rRectWd;
    }

    Point2f vtx[4];
    rRect.points(vtx);
    for (int i = 0; i < 4; i++) {
      vtx[i].x = cvRound(vtx[i].x / scale);
      vtx[i].y = cvRound(vtx[i].y / scale);
    }

#if (ENTRANCE_CONE_DEBUG)
    for (int i = 0; i < 4; i++)
      line(img_display, vtx[i], vtx[(i+1)%4], Scalar(0, 0, 255), 1, CV_AA);
#endif

    if (ang > 30 || ratio < 1.2) continue;
    Rect brect = rRect.boundingRect();
    boundRect(sml_sz, brect);
    Mat local_z = sml_z(brect);
    Mat mask = sml_dist_bin(brect) & yz_mask(brect);
    float dep = depthCalculate(local_z, mask, 0.5);

    Rect big_brect;
    big_brect.x = cvRound(brect.x / scale);
    big_brect.y = cvRound(brect.y / scale);
    big_brect.width = cvRound(brect.width / scale);
    big_brect.height = cvRound(brect.height / scale);
    // Computer physical area of core based on bounding box area.
    double obj_area = big_brect.area() * dep * dep / (focal_len * focal_len);

#if (ENTRANCE_CONE_DEBUG)
    rectangle(img_display, big_brect, Scalar(128,128,0), 2);
#endif

    //if (obj_area > 0.60 || obj_area < 0.07 || y_pos > 1.5)
    if (obj_area > 1.2 || obj_area < 0.07) continue;

    // Cone classifier is used in extended region around the candidated regions.
    double extendFactorX, extendFactorY;
    Size init_size;
    bool enlarge_gray = false;
    if (dep < 2.3) {
      extendFactorX = 2.5;
      extendFactorY = 1.5;
      init_size = Size(40, 80);
    } else if (dep < 5) {
      extendFactorX = 2.5;
      extendFactorY = 2;
      init_size = Size(25, 50);
    } else {
      extendFactorX = 3;
      extendFactorY = 2;
      init_size = Size(20, 30);
      enlarge_gray = true;
    }

    int rect_width = cvRound(big_brect.width * extendFactorX);
    int rect_height = cvRound(big_brect.height * extendFactorY);
    int rect_cx = big_brect.x + big_brect.width/2;
    int rect_cy = big_brect.y + big_brect.height/2;

    Rect rr(rect_cx - rect_width/2, rect_cy - rect_height/2, rect_width, 
        rect_height);
    boundRect(img_display.size(), rr);

#if (ENTRANCE_CONE_DEBUG)
    rectangle(img_display, rr, Scalar(128,128,128), 2);
    rectangle(img_display, big_brect, Scalar(255,255,0), 2);
#endif

    Mat tmp_gray = big_gray(rr);
    Mat tmp_gray1;
    if (!enlarge_gray) 
      tmp_gray1 = tmp_gray;
    else
      cv::resize(tmp_gray, tmp_gray1, Size(2*tmp_gray.cols, 2*tmp_gray.rows));

    // Use LBP based boosting classifier to filter out false alarms.
    vector<Rect> rects;
    clsf.detectMultiScale(tmp_gray1, rects, 1.04, 2, 0 | CV_HAAR_SCALE_IMAGE, 
        init_size);

    // Based on classifier results, use more accurate geometrical info: 
    // area, height to verify the detection.
    for (unsigned int i = 0; i < rects.size(); i++) {
      Rect rc = rects[i];
      if (enlarge_gray) 
        rc = Rect(rc.x/2, rc.y/2, rc.width/2, rc.height/2);

      rc.x += rr.x;
      rc.y += rr.y;
      int cx = rc.x + rc.width/2;
      int cy = rc.y + rc.height/2;

      // Once classifier passed, verify the region again by geometrical info.
      if (cx > big_brect.x && cx < big_brect.x + big_brect.width 
          && cy > big_brect.y && cy < big_brect.y + big_brect.height) {
        Rect rc_sml;
        rc_sml.x = cvRound(rc.x * scale);
        rc_sml.y = cvRound(rc.y * scale);
        rc_sml.width = cvRound(rc.width * scale);
        rc_sml.height = cvRound(rc.height * scale);
        boundRect(sml_sz, rc_sml);

        Mat tmp_z = sml_z(rc_sml);
        Mat tmp_mask = sml_dist_bin(rc_sml) & yz_mask(rc_sml);

        float z_val = depthCalculate(tmp_z, tmp_mask, 0.4);
        float y_center = (rc_sml.y + rc_sml.height/2.0) / scale;
        float y_val = (y_origin - y_center) * z_val / focal_len; //up - positive
        float x_center = (rc_sml.x + rc_sml.width/2.0) / scale;
        float x_val = (x_center - x_origin) * z_val / focal_len; //up - positive

        Vec3f vf(x_val, z_val, y_val);
        if (z_val > 10 || z_val < 1 || y_val > 1.0) continue;

        double cone_area = rc.area() * z_val * z_val / (focal_len * focal_len);
        if (cone_area > 0.6 || cone_area < 0.06) continue;
        fnl_rects.push_back(rc);
        targets.push_back(vf);
      }
    }
  }


  // Based on resulting bounding boxes, filter out outliers by the constraints
  // of the pair of cones: distance around 1m.
  int num = (int)targets.size();
  vector<int> keep_flag(num, 1);
  for (int k = 0; k < num; k++) {
    if (keep_flag[k] == 0) continue;
    float x0 = targets[k][0];
    float y0 = targets[k][1];

    for (int n = 0; n < num; n++) {
      if (n == k || keep_flag[n] == 0) continue;
      float x1 = targets[n][0];
      float y1 = targets[n][1];
      float dist = (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1);
      dist = std::sqrt(dist);

      if (dist < 0.5) {
        Rect r0 = fnl_rects[k];
        Rect r1 = fnl_rects[n];
        int c0y = r0.y + r0.height/2;
        int c1y = r1.y + r1.height/2;

        if (c0y > c1y) keep_flag[n] = 0;
        else keep_flag[k] = 0;
      }
    }
  }

  vector<Vec3f> tmp_targets;
  vector<Rect> tmp_rects;

  for (int k = 0; k < num; k++) {
    if (1 == keep_flag[k]) {
      tmp_targets.push_back(targets[k]);
      tmp_rects.push_back(fnl_rects[k]);
      rectangle(img_display, fnl_rects[k], Scalar(0,0,255), 2);
    }
  }

  targets.clear();
  fnl_rects.clear();
  targets = tmp_targets;
  fnl_rects = tmp_rects;

  // Handle the case when only one cone is detected at door entrance: if one
  // cone was found, another one could be found nearby (within 1-2m), weaker
  // criterion can be used to get the 2nd one.
  if (1 == targets.size()) {
    Rect rect1 = fnl_rects[0];
    Vec3f v = targets[0];

    // The 2nd cone is supposed to at a maximal distance of 2.2m from 1st one.
    float x_dist = 2.2; 
    float dep = v[1];
    int x1 = rect1.x + rect1.width/2; // x position of cone 1
    int y1 = rect1.y + rect1.height/2; // y position of cone 1

    int x2_up = x_dist * focal_len / dep + x1;
    int x2_low = x1 - x_dist * focal_len / dep;
    int y2_low = y1 - rect1.height;
    int y2_up = y1 + rect1.height;

    Rect search_rois[2];
    // Searching ROI on the left side of 1st cone
    search_rois[0] = Rect(x2_low, y2_low, rect1.x - x2_low, y2_up - y2_low);
    boundRect(img.size(), search_rois[0]);

    // Searching ROI on the right side of 1st cone
    search_rois[1] = Rect(rect1.x + rect1.width, y2_low, 
        x2_up - rect1.x - rect1.width, y2_up - y2_low);
    boundRect(img.size(), search_rois[1]);

    vector<Rect> cone2_rects;
    for (int i = 0; i < 2; i++) {
      bool enlarge_gray = false;
      Size init_sz;
      if (dep < 2.3) init_sz = Size(40, 80);
      else if (dep < 5) init_sz = Size(25, 50);
      else if (dep < 10) {
        enlarge_gray = true;
        init_sz = Size(20, 30);
      } else { continue; }

      Mat tmp_gray = big_gray(search_rois[i]);
      Mat tmp_gray1;
      if (!enlarge_gray) { tmp_gray1 = tmp_gray; }
      else {
        cv::resize(tmp_gray, tmp_gray1, Size(2 * tmp_gray.cols, 
              2 * tmp_gray.rows));
      }

      vector<Rect> rcs;
      clsf.detectMultiScale(tmp_gray1, rcs, 1.04, 1, 0 | CV_HAAR_SCALE_IMAGE, 
          init_sz);

      float area_diff = 1000000;
      Rect used_rc;
      bool valid_rc = false;
      for (unsigned int k = 0; k < rcs.size(); k++) {
        Rect rc = rcs[k];
        if (enlarge_gray) {
          rc.x /= 2;
          rc.y /= 2;
          rc.width /= 2;
          rc.height/= 2;
        }
        rc.x += search_rois[i].x;
        rc.y += search_rois[i].y;
        boundRect(img.size(), rc);

#if (ENTRANCE_CONE_DEBUG)
        rectangle(img_display, rc, Scalar(255,255,255), 2);
#endif

        float tmp_area_diff = std::abs(rc.area() - rect1.area());
        float rect_ratio = rc.area() / (float)rect1.area();
        if (rect_ratio > 0.4 && rect_ratio < 1.8)
          valid_rc = true;
        if (tmp_area_diff < area_diff) {
          area_diff = tmp_area_diff;
          used_rc = rc;
        }
      }

      if (rcs.size() >= 1 && valid_rc) cone2_rects.push_back(used_rc);

#if (ENTRANCE_CONE_DEBUG)
      rectangle(img_display, search_rois[i], Scalar(0,255,0), 2);
#endif
    }

    // calculate accurate 2D position in ground for final traffic cones.
    if (cone2_rects.size() >= 1) {
      Rect rc_sml = cone2_rects[0];
      rc_sml.x = cvRound(rc_sml.x * scale);
      rc_sml.y = cvRound(rc_sml.y * scale);
      rc_sml.width = cvRound(rc_sml.width * scale);
      rc_sml.height = cvRound(rc_sml.height * scale);
      boundRect(sml_sz, rc_sml);

      Mat tmp_z = sml_z(rc_sml);
      Mat tmp_mask = sml_dist_bin(rc_sml) & yz_mask(rc_sml);

      int color_pixels = cv::countNonZero( sml_dist_bin );
      float color_ratio = color_pixels / (float)rc_sml.area();

      float z_val = depthCalculate(tmp_z, tmp_mask, 0.4);
      float y_center = (rc_sml.y + rc_sml.height/2.0) / scale;
      float y_val = (y_origin - y_center) * z_val / focal_len; //up - positive
      float x_center = (rc_sml.x + rc_sml.width/2.0) / scale;
      float x_val = (x_center - x_origin) * z_val / focal_len; //up - positive
      Vec3f vf(x_val, z_val, y_val);
      if ((std::abs(v[2] - vf[2]) < 0.7) && color_ratio > 0.2){
        targets.push_back(vf);
        rectangle(img_display, cone2_rects[0], Scalar(0,0,255), 2);
      }
    }
  }

  return targets;
}
