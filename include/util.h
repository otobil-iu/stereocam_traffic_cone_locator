/*
 *  File: util.h
 *  Declaration utility functions.
 *
 *  Created by Zhou Lubing.
 *  developed for TechX Challenge 2013.
 *  T-Mobile team, Nanyang Technological University
 *
 */

#ifndef UTIL_H_H
#define UTIL_H_H

#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

#define CV_BGR2bgr 100
#define CV_BGR2ISV 101
#define CV_BGR2Igr 102

void drawCross(Mat& img, Point pt, Scalar color, int r);
Mat colorConvert(const Mat& colorimg, int type);
float coneDepthCalc(const Mat& depth_src, const Rect& rc, const Mat& dispValid);
Mat triangleMat(Size sz);
Vec3f coneCordCalculate(const Mat& xyz, const Rect& rc, const Mat& dispValid);
void boundRect(Size sz, Rect& rect);
float depthCalculate(const Mat& z_img, Mat& mask, float tolerance);

#endif //UTIL_H_H
