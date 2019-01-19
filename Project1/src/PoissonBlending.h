
#ifndef POISSON_BLENDING_H
#define POISSON_BLENDING_H


#include <opencv.hpp>
#include "Eigen/Sparse"


using namespace cv;
using namespace std;

class Blending
{
public:

	Mat srcImage;
	Mat mask;

	Mat Result;

	Blending(Mat inputImage, Mat forgroundMask, Mat Guidance);


	void Compute();

private:

	int nUnknowns;

	map<pair<int, int>, int> maskPointToNodeIdx; // ±ê¼Ç

	Mat F;// divergence of the guidance field 
	Mat boundaryValues;  // boundaryValues

	void BuildLookUpTable();

};


#endif // INPAINTER_H
