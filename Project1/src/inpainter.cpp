/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "PoissonBlending.h"
#include "inpainter.h"
#include <opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/types_c.h>
#include <vector>


using namespace cv;
using namespace std;

#define PUNISH_MAX 1e7


Mat ExpandMask;  // 扩展mask
Mat GCInputImage;
Mat GcMask;
vector<ST_Offset> GcLABLE;
map<pair<int, int>, int> maskPointToNodeIdx; // 标记
vector<Point2i> GcNodes;

Inpainter::Inpainter(cv::Mat inputImage,cv::Mat InputMask,int halfPatchWidth,int mode)
{
    this->inputImage=inputImage.clone();
    this->mask= InputMask.clone();
    this->workImage=inputImage.clone();
    this->result.create(inputImage.size(),inputImage.type());
    this->halfPatchWidth=halfPatchWidth;
	this->PatchSize = 2 * halfPatchWidth + 1;


	k = 60;  // k main offsets

}

int Inpainter::checkValidInputs(){
    if(this->inputImage.type()!=CV_8UC3)
        return ERROR_INPUT_MAT_INVALID_TYPE;
    if(this->mask.type()!=CV_8UC1)
        return ERROR_INPUT_MASK_INVALID_TYPE;
    if(!CV_ARE_SIZES_EQ(&mask,&inputImage))
        return ERROR_MASK_INPUT_SIZE_MISMATCH;
    if(halfPatchWidth==0)
        return ERROR_HALF_PATCH_WIDTH_ZERO;
    return CHECK_VALID;


	for (int i = 0; i < workImage.rows; i++)
	{
		for (int j = 0; j < workImage.cols; j++)
		{
			if (mask.at<uchar>(i, j) > 0)
			{
				mask.at<uchar>(i, j) = 255;
			}
			else
			{
				mask.at<uchar>(i, j) = 0;
			}
		}
	}
}



bool GreaterSort(ST_Offset a, ST_Offset b) {
	return (a.nVoteNum > b.nVoteNum);
}

#define INT_TO_X(v) ((v)&((1<<12)-1))
#define INT_TO_Y(v) ((v)>>12)

void patchmatch(const Mat & a, const Mat & b, const Mat & mask, int tau, Mat & ann, Mat &annd, int patch_w);




bool isValid(int newX, int newY)
{
	if (newX >= 0 && newY >= 0 && newX < GcMask.cols && newY < GcMask.rows && GcMask.at<uchar>(newY, newX) == 0)
	{
		return true;
	}
	return false;
}

bool isReadable(int newX, int newY)
{
	if (newX >= 0 && newY >= 0 && newX < GcMask.cols && newY < GcMask.rows)
	{
		return true;
	}
	return false;
}


double dataFn(int p, int l)
{
	int CurX = GcNodes[p].x;
	int CurY = GcNodes[p].y;

	// 如果原本就在有效区内，则其偏移为0，才有效
	if (isValid(CurX, CurY) == true)
	{
		if (GcLABLE[l].pOffset.x == 0 && GcLABLE[l].pOffset.y == 0)
		{
			return 0;

		}
		else
		{
			return PUNISH_MAX;
		}
	}

	// 对于其他的node，偏移为0的情况下，其惩罚为最高
	if (GcLABLE[l].pOffset.x == 0 && GcLABLE[l].pOffset.y == 0)
	{
		return PUNISH_MAX;
	}

	// node 点加上对应偏置后是否在有效区域内
	int newX = GcLABLE[l].pOffset.x + GcNodes[p].x;
	int newY = GcLABLE[l].pOffset.y + GcNodes[p].y;

	// 在图像区域内，则dataterm为0，否则为
	if (isValid(newX, newY))
	{
		return 0;
	}

	return PUNISH_MAX;
}


double smoothFn(int p1, int p2, int l1, int l2)
{

	if (l1 == l2)
	{
		return 0;
	}

	int retMe = 0;


	Point2i x1_s_a = GcLABLE[l1].pOffset + GcNodes[p1];
	Point2i x1_s_b = GcNodes[p1] + GcLABLE[l2].pOffset;

	Point2i x2_s_a = GcNodes[p2] + GcLABLE[l1].pOffset;
	Point2i x2_s_b = GcLABLE[l2].pOffset + GcNodes[p2];


	// 确保可读取
	if (isReadable(x1_s_a.x, x1_s_a.y) && isReadable(x2_s_b.x, x2_s_b.y)
		&& isReadable(x1_s_b.x, x1_s_b.y) && isReadable(x2_s_a.x, x2_s_a.y))
	{
		Vec3b v1_a = GCInputImage.at<Vec3b>(x1_s_a.y, x1_s_a.x);
		Vec3b v1_b = GCInputImage.at<Vec3b>(x1_s_b.y, x1_s_b.x);

		for (int i = 0; i < 3; i++)
		{
			retMe += (v1_a[i] - v1_b[i])* (v1_a[i] - v1_b[i]);
		}

		Vec3b v2_a = GCInputImage.at<Vec3b>(x2_s_a.y, x2_s_a.x);
		Vec3b v2_b = GCInputImage.at<Vec3b>(x2_s_b.y, x2_s_b.x);

		for (int i = 0; i < 3; i++)
		{
			retMe += (v2_a[i] - v2_b[i])* (v2_a[i] - v2_b[i]);
		}


		return retMe;
	}

	return PUNISH_MAX;

}


void Inpainter::GetKDominateOffSet()
{
	int nMaxOffSetWidth = 2 * inputImage.cols;
	int nMaxOffSetHeight = 2 * inputImage.rows;

	//Statistic Patch Offset
	Mat VoteMatrix = Mat::zeros(Size(nMaxOffSetWidth, nMaxOffSetHeight), CV_16UC1);

	Point2i centerOffset(inputImage.cols, inputImage.rows);
	// offsets histogram
	for (int i = 0; i < inputImage.rows - PatchSize; i++)
	{
		for (int j = 0; j < inputImage.cols - PatchSize; j++)
		{
			int v = ann.at<int>(i, j);
			int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);

			// mask 区域内的不进行投票
			Rect RE2(j, i, PatchSize, PatchSize);
			Mat CurMask = mask(RE2);
			int unValidNum = 0;
			for (int i = 0; i < CurMask.rows; i++)
			{
				for (int j = 0; j < CurMask.cols; j++)
				{
					if (CurMask.at<uchar>(i, j))
					{
						unValidNum++;
					}
				}
			}

			// mask附近的patch不投票避免干扰
			if (unValidNum * 10 > CurMask.rows * CurMask.cols)
			{
				continue;
			}

			int OffsetX = xbest - j;
			int OffsetY = ybest - i;

			Point2i temp = centerOffset + Point2i(OffsetX, OffsetY);
			VoteMatrix.at<ushort>(temp) += 1;

		}
	}

	// boxFilter(VoteMatrix, VoteMatrix, -1, Size(10, 10));
	GaussianBlur(VoteMatrix, VoteMatrix, Size(9, 9), 1.414, 1.414);

	// non maximul suppress
	vector<ST_Offset> vecOffSet;
	for (int i = 0; i < VoteMatrix.rows; i++)
	{
		for (int j = 0; j < VoteMatrix.cols; j++)
		{
			// minimul requirement
			if (VoteMatrix.at<ushort>(i, j) > 5)
			{
#define NON_MAXIMUL_RADIUS 4
				int nMinX = max(0, j - NON_MAXIMUL_RADIUS);
				int nMaxX = min(VoteMatrix.cols - 1, j + NON_MAXIMUL_RADIUS);

				int nMinY = max(0, i - NON_MAXIMUL_RADIUS);
				int nMaxY = min(VoteMatrix.rows - 1, i + NON_MAXIMUL_RADIUS);

				int flag = 0;
				// 
				for (int a = nMinY; a < nMaxY; a++)
				{
					for (int b = nMinX; b < nMaxX; b++)
					{
						if (VoteMatrix.at<ushort>(i, j) < VoteMatrix.at<ushort>(a, b))
						{
							flag = 1;
							break;
						}
					}

					if (flag)
						break;
				}

				if (flag == 0)
				{
					ST_Offset temp;
					temp.nVoteNum = VoteMatrix.at<ushort>(i, j);

					temp.pOffset.y = i - inputImage.rows;
					temp.pOffset.x = j - inputImage.cols;

					vecOffSet.push_back(temp);
				}
			}
		}
	}


	printf("offset num:%zd\n", vecOffSet.size());
	// 排序
	sort(vecOffSet.begin(), vecOffSet.end(), GreaterSort);//降序排列

	int LableNum = min(int(vecOffSet.size()), k - 1);

	//选出的偏移量
	GcLABLE.assign(vecOffSet.begin(), vecOffSet.begin() + LableNum);

	ST_Offset zeroOffset;
	zeroOffset.pOffset.x = 0;
	zeroOffset.pOffset.y = 0;
	GcLABLE.push_back(zeroOffset);  // 添加0偏移项作为边界点的偏移项


	Mat MatShowHist = Mat::zeros(VoteMatrix.size(), CV_8UC3);

	Point2i Start, End;
	Start.x = centerOffset.x;
	Start.y = 0;
	End.x = centerOffset.x;
	End.y = MatShowHist.rows - 1;
	cv::line(MatShowHist, Start, End, Scalar(255, 0, 0));

	Start.x = 0;
	Start.y = centerOffset.y;
	End.x = MatShowHist.cols - 1;
	End.y = centerOffset.y;
	cv::line(MatShowHist, Start, End, Scalar(255, 0, 0));

	for (int i = 0; i < GcLABLE.size(); i++)
	{
		Point2i T;
		T = centerOffset + GcLABLE[i].pOffset;

		cv::circle(MatShowHist, T, 3, Scalar(0, 255, 0), -1);
	}


	cv::imshow("hist", MatShowHist);
	cvWaitKey(100);
}



void Inpainter::VisualizeResultLabelMap(GCoptimizationGeneralGraph *gc)
{
	std::vector<Vec3i> _labelColors;
	_labelColors.push_back(Vec3i(255, 0, 0));
	_labelColors.push_back(Vec3i(0, 255, 0));
	_labelColors.push_back(Vec3i(0, 0, 255));
	_labelColors.push_back(Vec3i(255, 255, 0));
	_labelColors.push_back(Vec3i(255, 0, 255));
	_labelColors.push_back(Vec3i(0, 255, 255));
	_labelColors.push_back(Vec3i(255, 128, 0));
	_labelColors.push_back(Vec3i(255, 0, 128));
	_labelColors.push_back(Vec3i(128, 255, 0));
	_labelColors.push_back(Vec3i(0, 255, 128));
	_labelColors.push_back(Vec3i(128, 0, 255));
	_labelColors.push_back(Vec3i(0, 128, 255));
	_labelColors.push_back(Vec3i(255, 255, 128));
	_labelColors.push_back(Vec3i(255, 128, 255));
	_labelColors.push_back(Vec3i(128, 255, 255));
	_labelColors.push_back(Vec3i(255, 128, 128));
	_labelColors.push_back(Vec3i(128, 255, 128));
	_labelColors.push_back(Vec3i(128, 128, 255));
	_labelColors.push_back(Vec3i(128, 64, 128));
	_labelColors.push_back(Vec3i(128, 128, 64));

	// 增加标记颜色
	for (int i = 20; i < GcLABLE.size(); i++)
	{
		int h = int((i - 20)*(360.f / GcLABLE.size() - 20));
		Vec3f col(HSVtoRGB(Vec3f(0 + h, (70) / 100., (90) / 100.)));
		Vec3i newC(int(col[0] * 255), int(col[1] * 255), int(col[2] * 255));
		_labelColors.push_back(newC);
	}

	Label = workImage.clone();

	for (int i = 0; i < GcNodes.size(); i++)
	{
		int res = gc->whatLabel(i);

		// zero offset
		if (GcLABLE[res].pOffset.x == 0 && GcLABLE[res].pOffset.y == 0)
		{
			continue;
		}

		Label.at<Vec3b>(GcNodes[i]) = Vec3b(_labelColors[res]);
	}
}

void Inpainter::inpaint()
{

	// 膨胀一次，扩展一圈
	{
		Mat element = getStructuringElement(MORPH_CROSS,
			Size(2 * 1 + 1, 2 * 1 + 1),
			Point(1, 1));

		dilate(mask, ExpandMask, element);
	}

	for (int i =0; i < workImage.rows; i++)
	{
		for (int j = 0; j < workImage.cols; j++)
		{

			if (ExpandMask.at<uchar>(i,j)) //
			{
				GcNodes.push_back(Point2i(j, i));
				maskPointToNodeIdx[make_pair(j, i)] = GcNodes.size() - 1;
			}
			else
			{
				maskPointToNodeIdx[make_pair(j, i)] =  - 1; 
			}
		}
	}

	GcMask = mask;
	GCInputImage = inputImage;
	workImage = inputImage.clone();

	int minLen = min(inputImage.cols, inputImage.rows) / 15;
	tau = minLen * minLen;  //minimul offset value

	// calculate the match corresponce
	patchmatch(inputImage, inputImage, mask, tau, ann, annd, PatchSize);

	GetKDominateOffSet();

	GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(GcNodes.size(), GcLABLE.size());
	gc->setDataCost(&dataFn);
	gc->setSmoothCost(&smoothFn);

	for (int img_y = 0; img_y < mask.rows; img_y++)
	{
		for (int img_x = 1; img_x < mask.cols; img_x++)
		{
			if (maskPointToNodeIdx[make_pair(img_x, img_y)] >= 0 && maskPointToNodeIdx[make_pair(img_x - 1, img_y)] >= 0)
			{
				gc->setNeighbors(maskPointToNodeIdx[make_pair(img_x, img_y)], maskPointToNodeIdx[make_pair(img_x - 1, img_y)], 1);
			}
		}
	}

	for (int img_y = 1; img_y < mask.rows; img_y++)
	{
		for (int img_x = 0; img_x < mask.cols; img_x++)
		{
			if (maskPointToNodeIdx[make_pair(img_x, img_y)] >= 0 && maskPointToNodeIdx[make_pair(img_x, img_y - 1)] >= 0)
			{
				gc->setNeighbors(maskPointToNodeIdx[make_pair(img_x, img_y)], maskPointToNodeIdx[make_pair(img_x, img_y - 1)], 1);
			}
		}
	}

	cout << "Energy Before: " << gc->compute_energy() << endl;
	gc->swap(2);
	cout << "Energy After 1: " << gc->compute_energy() << endl;

	Mat Guidance = Mat::zeros(workImage.size(), CV_32FC3);
	const int offsets[4][2] = { { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 0 } };
	const cv::Rect bounds(0, 0, workImage.cols, workImage.rows);


	//  gradient domain
	Mat gx, gy; 
	Mat kernel = Mat::zeros(3, 1, CV_8S);
	kernel.at<char>(2, 0) = 1;
	kernel.at<char>(1, 0) = -1;
	filter2D(GCInputImage, gy, CV_32F, kernel);

	kernel = Mat::zeros(1,3, CV_8S);
	kernel.at<char>(0, 2) = 1;
	kernel.at<char>(0, 1) = -1;
	filter2D(GCInputImage, gx, CV_32F, kernel);

	// copy gradient domain
	Mat dx =  Mat::zeros(workImage.size(), CV_32FC3);
	Mat dy = Mat::zeros(workImage.size(), CV_32FC3);

	for (int i = 0; i < GcNodes.size(); i++)
	{
		int res = gc->whatLabel(i);

		Point2i TargetP = GcNodes[i] + GcLABLE[res].pOffset;

		dx.at<Vec3f>(GcNodes[i]) = gx.at<Vec3f>(TargetP);
		dy.at<Vec3f>(GcNodes[i]) = gy.at<Vec3f>(TargetP);

		// zero offset
		if (GcLABLE[res].pOffset.x == 0 && GcLABLE[res].pOffset.y == 0)
		{
			continue;
		}
	
		for (int nch = 0; nch < 3; nch++)
		{
			workImage.at<Vec3b>(GcNodes[i])[nch] = GCInputImage.at<Vec3b>(TargetP)[nch];
		}
	}

	result = workImage.clone();


	// calculate the div of filling region
	kernel = Mat::zeros(3, 1, CV_8S);
	kernel.at<char>(2, 0) = 1;
	kernel.at<char>(1, 0) = -1;
	filter2D(dy, gy, CV_32F, kernel);

	kernel = Mat::zeros(1, 3, CV_8S);
	kernel.at<char>(0, 2) = 1;
	kernel.at<char>(0, 1) = -1;
	filter2D(dx, gx, CV_32F, kernel);


	for (int i = 0; i < workImage.rows; i++)
	{
		for (int j = 0; j < workImage.cols; j++)
		{
			if (mask.at<uchar>(i, j)) //
			{
				Guidance.at<Vec3f>(i, j) = gx.at<Vec3f>(i, j) + gy.at<Vec3f>(i, j);
			}
		}
	}


	// imshow("Guidance", Guidance);
	imshow("input", inputImage);
	imshow("completed", workImage);

	VisualizeResultLabelMap(gc);
	imshow("result Label Map", Label);

	Blending blend(workImage, mask, Guidance);
	blend.Compute();


	blend.Result.convertTo(result, CV_8UC3);

	imshow("Poisson blending", result);
	cvWaitKey(0);


}