
#include "PoissonBlending.h"

using namespace Eigen;


void Blending::BuildLookUpTable()
{
	int Num = 0;
	for (int i = 0; i < srcImage.rows; i++)
	{
		for (int j = 0; j < srcImage.cols; j++)
		{

			if (mask.at<uchar>(i, j)) //
			{
				maskPointToNodeIdx[make_pair(j, i)] = Num++;
			}
			else
			{
				maskPointToNodeIdx[make_pair(j, i)] = -1;
			}
		}
	}

	nUnknowns = Num;
}

Blending::Blending(Mat inputImage, Mat forgroundMask, Mat Guidance)
{
	srcImage = inputImage;
	srcImage.convertTo(boundaryValues, CV_32FC3);
	mask = forgroundMask;

	F = Guidance;
	srcImage.convertTo(Result, CV_32FC3);

	BuildLookUpTable();
 }





void Blending::Compute()
{
	const cv::Rect bounds(0, 0, srcImage.cols, srcImage.rows);

	// Directional indices
	const int center = 0;
	const int north = 1;
	const int east = 2;
	const int south = 3;
	const int west = 4;

	// Neighbor offsets in all directions
	const int offsets[5][2] = { { 0, 0 }, { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 0 } };

	// Directional opposite
	const int opposite[5] = { center, south, west, north, east };
	// blending

	std::vector< Eigen::Triplet<float> > lhsTriplets;
	lhsTriplets.reserve(nUnknowns * 5);

	int channels = srcImage.channels();
	Eigen::MatrixXf rhs(nUnknowns, channels);
	rhs.setZero();

	for (int y =0; y < srcImage.rows; y++)
	{
		for (int x = 0; x < srcImage.cols; x++)
		{
			int pid = maskPointToNodeIdx[make_pair(x, y)];
			// 
			if (pid == -1)
			{
				continue;
			}

			float lhs[] = { -4.f, 1.f, 1.f, 1.f, 1.f };

			for (int n = 1; n < 5; ++n) 
			{
				const cv::Point q(x + offsets[n][0], y + offsets[n][1]);

				const bool hasNeighbor = bounds.contains(q);
				const bool isNeighborDirichlet = hasNeighbor && (mask.at<uchar>(q) == 0);

				if (!hasNeighbor) 
				{
					lhs[center] += lhs[n];
					lhs[n] = 0.f;
				}
				else if (isNeighborDirichlet) 
				{

					// Implementation note:
					//
					// Dirichlet boundary conditions (DB) turn neighbor unknowns into knowns (data) and
					// are therefore moved to the right hand side. Alternatively, we could add more
					// equations for these pixels setting the lhs 1 and rhs to the Dirichlet value, but
					// that would unnecessarily blow up the equation system.

					rhs.row(pid) -= lhs[n] * Eigen::Map<Eigen::VectorXf>(boundaryValues.ptr<float>(q.y, q.x), channels);
					lhs[n] = 0.f;
				}
			}


			// Add f to rhs.
			rhs.row(pid) += Eigen::Map<Eigen::VectorXf>(F.ptr<float>(y, x), channels);

			// Build triplets for row              
			for (int n = 0; n < 5; ++n) 
			{
				if (lhs[n] != 0.f) 
				{
					int qId = maskPointToNodeIdx[make_pair(x + offsets[n][0], y + offsets[n][1])];

					lhsTriplets.push_back(Eigen::Triplet<float>(pid, qId, lhs[n]));
				}
			}
		}
	}


	// Solve the sparse linear system of equations
	Eigen::SparseMatrix<float> A(nUnknowns, nUnknowns);
	A.setFromTriplets(lhsTriplets.begin(), lhsTriplets.end());

	Eigen::SparseLU< Eigen::SparseMatrix<float> > solver;
	solver.analyzePattern(A);
	solver.factorize(A);

	Eigen::MatrixXf result(nUnknowns, channels);
	for (int c = 0; c < channels; ++c)
		result.col(c) = solver.solve(rhs.col(c));


	// Copy results back
	for (int y = 0; y < srcImage.rows; ++y) 
	{
		for (int x = 0; x < srcImage.cols; ++x) 
		{
			const int pid = maskPointToNodeIdx[make_pair(x , y )];

			if (pid > -1)
			{
				Eigen::Map<Eigen::VectorXf>(Result.ptr<float>(y, x), channels) = result.row(pid);
			}
		}
	}
}