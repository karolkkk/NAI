#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "PossibleChar.h"

using namespace cv;
using namespace std;


Mat src; Mat src_gray; Mat binary_img;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

void thresh_callback(int, void*);


int main(int argc, char** argv)
{
	/// Load source image and convert it to gray
	src = imread("./image/plate1.png", 1);
	resize(src, src, Size(500, 250));
	
	/// Convert image to gray and blur it
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));


	thresh_callback(0, 0); 

	



	waitKey(0);
	return(0);
}

void thresh_callback(int, void*)
{
	Mat canny_output, imgE, img_sobel, binary_img0;
	vector<vector<Point> > contours3;
	vector<Vec4i> hierarchy2;



	threshold(src_gray, binary_img0, 60, 255, THRESH_BINARY);

	Mat kernel = (Mat_<uchar>(3, 3) <<
		1, 1, 1, 1, 1,
		1, 1, 1, 1);

	erode(binary_img0, imgE, kernel);
	dilate(imgE, img_sobel, kernel);


	/// Detect edges using canny
	Canny(binary_img0, canny_output, thresh, thresh * 2, 3);
	/// Find contours
	findContours(canny_output, contours3, hierarchy2, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));




	vector < vector<Point>> contour;
	findContours(binary_img0, contour, 0, 2);


	bitwise_not(binary_img0, binary_img);

	cv::Mat img2 = binary_img.clone();


	std::vector<cv::Point> points;
	cv::Mat_<uchar>::iterator it = binary_img.begin<uchar>();
	cv::Mat_<uchar>::iterator end = binary_img.end<uchar>();
	for (; it != end; ++it)
		if (*it)
			points.push_back(it.pos());

	cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));

	double angle = box.angle;
	if (angle < -45.)
		angle += 90.;

	cv::Point2f vertices[4];
	box.points(vertices);
	for (int i = 0; i < 4; ++i)
		cv::line(binary_img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 1, CV_AA);



	cv::Mat rot_mat = cv::getRotationMatrix2D(box.center, angle, 1);

	cv::Mat rotated;
	cv::warpAffine(img2, rotated, rot_mat, binary_img.size(), cv::INTER_CUBIC);



	cv::Size box_size = box.size;
	if (box.angle < -45.)
		std::swap(box_size.width, box_size.height);
	cv::Mat cropped;

	cv::getRectSubPix(rotated, box_size, box.center, cropped);
	cv::imshow("Cropped", cropped);
	imwrite("example5.jpg", cropped);

	Mat cropped2 = cropped.clone();
	cvtColor(cropped2, cropped2, CV_GRAY2RGB);



	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Find contours
	cv::findContours(cropped, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0));



	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());


	//Get poly contours
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
	}


	//Get only important contours, merge contours that are within another
	vector<vector<Point> > validContours;
	for (int i = 0; i < contours_poly.size(); i++) {

		
			Rect r = boundingRect(Mat(contours_poly[i]));
			if (r.area() < 100)continue;
			bool inside = false;
			for (int j = 0; j < contours_poly.size(); j++) {
				if (j == i)continue;

				Rect r2 = boundingRect(Mat(contours_poly[j]));
				if (r2.area() < 100 || r2.area() < r.area())continue;
				if (r.x > r2.x&&r.x + r.width<r2.x + r2.width&&
					r.y>r2.y&&r.y + r.height < r2.y + r2.height) {

					inside = true;
				}
			}
			if (inside)continue;
			validContours.push_back(contours_poly[i]);
		}



		//Get bounding rects
		for (int i = 0; i < validContours.size(); i++) {
			boundRect[i] = boundingRect(Mat(validContours[i]));
		}

		

 
		Scalar color = Scalar(0, 0, 255);
		for (int i = 0; i < validContours.size(); i++)
		{

			if (validContours[i].size() > 6) {


				Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				char ch = waitKey(0);
				if (boundRect[i].area() < 100)continue;
				drawContours(cropped2, validContours, i, color, FILLED, 8, vector<Vec4i>(), 0, Point());


				imshow("Contours2", cropped2);
			}

		
		}
	}

 