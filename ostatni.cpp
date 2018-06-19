#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>



using namespace cv;
using namespace std;





Mat src; Mat src_gray; Mat binary_img;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);
Ptr<ml::KNearest> kNearest = ml::KNearest::create();

/// Function header
void thresh_callback(int, void*);
bool loadKNNDataAndTrainKNN(void);
void extractContours(Mat& image, vector< vector<Point> > contours_poly);

class comparator {
public:
	bool operator()(vector<Point> c1, vector<Point>c2) {

		return boundingRect(Mat(c1)).x<boundingRect(Mat(c2)).x;

	}

};



void extractContours(Mat& image, vector< vector<Point> > contours_poly) {



	
	sort(contours_poly.begin(), contours_poly.end(), comparator());


	for (int i = 0; i < contours_poly.size(); i++) {

		Rect r = boundingRect(Mat(contours_poly[i]));


		Mat mask = Mat::zeros(image.size(), CV_8UC1);
		
		drawContours(mask, contours_poly, i, Scalar(255), CV_FILLED);

		
		if (i + 1 < contours_poly.size()) {
			Rect r2 = boundingRect(Mat(contours_poly[i + 1]));
			if (abs(r2.x - r.x) < 20) {
				
				drawContours(mask, contours_poly, i + 1, Scalar(255), CV_FILLED);
				i++;
				int minX = min(r.x, r2.x);
				int minY = min(r.y, r2.y);
				int maxX = max(r.x + r.width, r2.x + r2.width);
				int maxY = max(r.y + r.height, r2.y + r2.height);
				r = Rect(minX, minY, maxX - minX, maxY - minY);

			}
		}
		
		Mat extractPic;
		
		image.copyTo(extractPic, mask);
		Mat resizedPic = extractPic(r);

		

		
		imshow("image" + i, resizedPic);


		
		stringstream searchMask;
		searchMask << i-1 << ".jpg";
		imwrite(searchMask.str(), resizedPic);

	}
}
int main(int argc, char** argv)
{

	bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();           // attempt KNN training

	if (blnKNNTrainingSuccessful == false) {                            // if KNN training was not successful
																		
		cout << endl << endl << "error: error: KNN traning was not successful" << endl << endl;
		return(0);                                                      
	}

	
	src = imread("image5.png", 1);

	
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

	/// Draw contours
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (int i = 0; i < contours3.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours3, i, color, 2, 8, hierarchy2, 0, Point());
	}


	vector<vector<Point>> contours_poly2(contours3.size());
	

	bitwise_not(binary_img0, binary_img);

	Mat img2 = binary_img.clone();


	vector<Point> points;
	Mat_<uchar>::iterator it = binary_img.begin<uchar>();
	Mat_<uchar>::iterator end = binary_img.end<uchar>();
	for (; it != end; ++it)
		if (*it)
			points.push_back(it.pos());

	RotatedRect box = minAreaRect(Mat(points));

	double angle = box.angle;
	if (angle < -45.)
		angle += 90.;

	Point2f vertices[4];
	box.points(vertices);
	for (int i = 0; i < 4; ++i)
		line(binary_img, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0), 1, CV_AA);



	Mat rot_mat = getRotationMatrix2D(box.center, angle, 1);

	Mat rotated;
	warpAffine(img2, rotated, rot_mat, binary_img.size(), INTER_CUBIC);



	Size box_size = box.size;
	if (box.angle < -45.)
		swap(box_size.width, box_size.height);
	Mat cropped;

	getRectSubPix(rotated, box_size, box.center, cropped);
	imshow("Cropped", cropped);
	imwrite("example5.jpg", cropped);

	Mat cropped2 = cropped.clone();
	cvtColor(cropped2, cropped2, CV_GRAY2RGB);

	Mat cropped3 = cropped.clone();
	cvtColor(cropped3, cropped3, CV_GRAY2RGB);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(cropped, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0));



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


	vector<vector<Point> > validContours;
	for (int i = 0; i<contours_poly.size(); i++) {

		Rect r = boundingRect(Mat(contours_poly[i]));
		if (r.area()<100)continue;
		bool inside = false;
		for (int j = 0; j<contours_poly.size(); j++) {
			if (j == i)continue;

			Rect r2 = boundingRect(Mat(contours_poly[j]));
			if (r2.area()<100 || r2.area()<r.area())continue;
			if (r.x>r2.x&&r.x + r.width<r2.x + r2.width&&
				r.y>r2.y&&r.y + r.height<r2.y + r2.height) {

				inside = true;
			}
		}
		if (inside)continue;
		validContours.push_back(contours_poly[i]);
	}


	//Get bounding rects
	for (int i = 0; i<validContours.size(); i++) {
		boundRect[i] = boundingRect(Mat(validContours[i]));
	}


	//Display
	Scalar color = Scalar(255,0 , 0);
	for (int i = 0; i< validContours.size(); i++)
	{
		if (boundRect[i].area()<100)continue;
		drawContours(cropped2, validContours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(cropped2, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);

	}

	imshow("a", cropped2);

	extractContours(cropped3, validContours);




	//string strChars;               // this will be the return value, the chars in the lic plate

	
	Mat imgThreshColor;

	// posortuj chary od lewej do prawej
	sort(contours_poly.begin(), contours_poly.end(), comparator());

	
	int i = 0;
	for (auto &currentChar : contours_poly) {
		 


		stringstream searchMask;
		searchMask << i << ".jpg";
		Mat img = imread(searchMask.str());
		
		Mat imgROIResized;
		resize(img, imgROIResized, Size(20, 30));

		imshow("as", imgROIResized);

		waitKey(1000);

		Mat matROIFloat;

		imgROIResized.convertTo(matROIFloat, CV_32FC1);         // convert Mat to float, necessary for call to findNearest

		Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);       // flatten Matrix into one row

		Mat matCurrentChar(0, 0, CV_32F);                   // declare Mat to read current char into, this is necessary b/c findNearest requires a Mat


		//kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     

																		


		i++;
	}


}

bool loadKNNDataAndTrainKNN(void) {

	// zczytaj treningowe classifications ///////////////////////////////////////////////////

	Mat matClassificationInts;              // we will read the classification numbers into this variable as though it is a vector

	FileStorage fsClassifications("classifications.xml", FileStorage::READ);        // otworrz plik classifications 

	if (fsClassifications.isOpened() == false) {                                                        // jak sie nie otworzy
		
	cout << "error, unable to open training classifications file, exiting program\n\n";        
		return(false);                                                                                 
	}

	fsClassifications["classifications"] >> matClassificationInts;          // read classifications section into Mat classifications variable
	fsClassifications.release();                                            // zamknij plik classifications 

																			// wczytaj zdj treningowe ////////////////////////////////////////////////////////////

	Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector

	FileStorage fsTrainingImages("images.xml", FileStorage::READ);              // otworz treningowy plik

	if (fsTrainingImages.isOpened() == false) {                                                 // jak sie nie udalo otworzyc
		cout << "error, unable to open training images file, exiting program\n\n";         
		return(false);                                                                          // i wyjdz
	}

	fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // zczytaj zdjecia do mattrainingimages 
	fsTrainingImages.release();                                                 // zamknij treningowe pliki

																				// train //////////////////////////////////////////////////////////////////////////////

																				
	kNearest->setDefaultK(1);

	kNearest->train(matTrainingImagesAsFlattenedFloats, ml::ROW_SAMPLE, matClassificationInts);

	return true;
}



