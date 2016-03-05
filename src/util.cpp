/*
 *  File: util.cpp
 *  Definition of utility functions.
 *
 *  Created by Zhou Lubing.
 *  developed for TechX Challenge 2013.
 *  T-Mobile team, Nanyang Technological University
 *
 */

#include "util.h"

void drawCross(Mat& img, Point pt, Scalar color, int r) {
  int x0 = pt.x;
  int y0 = pt.y;
  Point tl(x0-r, y0-r);
  Point tr(x0+r, y0-r);
  Point bl(x0-r, y0+r);
  Point br(x0+r, y0+r);
  cv::line(img, tl, br, color);
  cv::line(img, bl, tr, color);
}

Mat colorConvert(const Mat& colorimg, int type) {
  CV_Assert(colorimg.channels() == 3);
  Mat img, planes[3], sumimg, bgrimg, grayimg, hsv;

  switch (type) {
    case CV_BGR2bgr:
      bgrimg = Mat_<Vec3f>(colorimg);
      split(bgrimg, planes);
      sumimg = planes[0] + planes[1] + planes[2];
      for (int k = 0; k < 3; k++)
        planes[k] = 255.0 * planes[k] / sumimg;
      merge(planes, 3, img);
      break;
    case CV_BGR2ISV:
      cvtColor( colorimg, hsv, CV_BGR2HSV);
      cvtColor( colorimg, grayimg, CV_BGR2GRAY);
      split(hsv, planes);
      planes[0] = grayimg;
      cv::merge(planes, 3, img);
      break;
    case CV_BGR2Igr:
      bgrimg = Mat_<Vec3f>(colorimg);
      split(bgrimg, planes);
      sumimg = planes[0] + planes[1] + planes[2];
      for (int k = 1; k < 3; k++) {
        planes[k] = 255.0 * planes[k] / sumimg;
        planes[k] = (Mat_<uchar>(planes[k]));
      }

      cvtColor(colorimg, grayimg, CV_BGR2GRAY);
      planes[0] = grayimg;
      merge(planes, 3, img);
      break;
    default:
      cvtColor(colorimg, img, type); 
  }
  return img;
}

float coneDepthCalc(const Mat& src, const Rect& rc, const Mat& dispValid) {
  Mat dep_src, valid_src;
  dep_src = src(rc);
  valid_src = dispValid(rc);

  Mat dep_inrange =  (dep_src < 0) & (dep_src > -5);
  int num_inrange = cv::countNonZero(dep_inrange);
  int num_valids =  cv::countNonZero(valid_src > 0);
  float inrange_ratio;
  if (num_valids <  10)
    inrange_ratio = 0;
  else
    inrange_ratio = (float)num_inrange / num_valids;

  float dep = -10000;
  if (inrange_ratio > 0.3) {
    Mat rangeCounter(1, 18, CV_32S, Scalar(0));
    float interv = 0.5; // 0.5m distance interval for histogram.

    for (int r = 0; r < rc.height; r++) {
      for (int c = 0; c < rc.width; c++) {
        if ((int)dep_inrange.at<uchar>(r,c) > 0) {
          float zval = -dep_src.at<float>(r,c);
          int idx = cvFloor(zval / interv);
          if (idx >= 0 && idx < 18)
            rangeCounter.at<int>(0, idx) += 1;
        }
      }
    }

    for (int i = 0; i < rangeCounter.cols - 1; i++)
      rangeCounter.at<int>(0, i) += rangeCounter.at<int>(0, i + 1);

    Point pt;
    cv::minMaxLoc(rangeCounter, NULL, NULL, NULL, &pt);

    int npixels = 0;
    dep = 0.0;
    for (int r = 0; r < rc.height; r++) {
      for (int c = 0; c < rc.width; c++) {
        if ((int)dep_inrange.at<uchar>(r,c) > 0) {
          float zval = -dep_src.at<float>(r,c);
          int idx = cvFloor(zval/interv);
          if (idx == pt.x || idx == (pt.x+1)) {
            dep -= zval;
            npixels++;
          }
        }
      }
    }

    if (npixels > 6) dep /= (float)npixels;
    else dep = -10000;
  }

  return dep;
}

Vec3f coneCordCalculate(const Mat& xyz, const Rect& rc, const Mat& dispValid) {
  Mat dep_src, valid_src;
  dep_src = xyz(rc);
  valid_src = dispValid(rc);

  Mat planes[3];
  split(dep_src, planes);
  Mat z = planes[2];

  Mat triangle_mat = triangleMat(Size(rc.width, rc.height));
  Mat dep_inrange =  (z < 0) & (z > -5) & triangle_mat ;

  int num_valids =  cv::countNonZero((valid_src > 0) & triangle_mat);
  float inrange_ratio;
  if (num_valids <  10) {
    inrange_ratio = 0;
  } else {
    int num_inrange = cv::countNonZero(dep_inrange);
    inrange_ratio = (float)num_inrange / num_valids;
  }

  if (inrange_ratio > 0.3) {
    float interv = 0.4; //0.5 meter each bin
    int nbins = cvFloor(5.0/interv);
    Mat rangeCounter(1, nbins, CV_32S, Scalar(0));

    for (int r = 0; r < rc.height; r++) {
      for (int c = 0; c < rc.width; c++) {
        if ((int)dep_inrange.at<uchar>(r,c) > 0) {
          float zval = -z.at<float>(r,c);
          int idx = cvFloor(zval / interv);
          if(idx >= 0 && idx < nbins)
            rangeCounter.at<int>(0, idx) += 1;
        }
      }
    }

    for (int i = 0; i < nbins-1; i++)
      rangeCounter.at<int>(0, i) += rangeCounter.at<int>(0, i + 1);
    Point pt;
    cv::minMaxLoc(rangeCounter, NULL, NULL, NULL, &pt);
    float maxz = -interv * pt.x; 
    float minz = -interv * (pt.x + 2);
    Mat usedPixels = (z < maxz) & (z > minz) & triangle_mat;

    //imshow("usedpixels", usedPixels);

    Scalar s = cv::mean(dep_src, usedPixels);
    Vec3f v(s[0], s[1], s[2]);
    return v;
  } else {
    return Vec3f(0,0,0);
  }
}

Mat triangleMat(Size sz) {
  int w = sz.width;
  int h = sz.height;

  Mat dst( sz, CV_8U, Scalar(0) );
  float x0, y1, x2, y2;
  x0 = (w - 1) / 2.0;
  y1 = h - 1;
  y2 = h - 1;
  x2 = w - 1;

  for (int x = 0; x < w; x++) {
    for (int y = 0; y < h; y++) {
      float s1 = y1 * x - x0 * y1 + x0 * y;
      float s2 = y * (x2 - x0) - y2 * (x - x0);
      if (s1 > 0 && s2 > 0)
        dst.at<uchar>(y, x) = (uchar)255;
    }
  }
  return dst;
}

float depthCalculate(const Mat& z_img, Mat& mask, float tolerance) {
  int num_valids = cv::countNonZero(mask);
  float z_dist = -10000;
  if (num_valids > 10) {
    Scalar s = cv::mean(z_img, mask);
    mask = mask & (z_img > s[0] - tolerance) & (z_img < s[0] + tolerance);
    s = cv::mean(z_img, mask);
    z_dist = s[0];
  }
  return z_dist;
}

void boundRect(Size sz, Rect& rect) {
  Rect whole_roi(0, 0, sz.width, sz.height);
  rect = whole_roi & rect;
}
