
/* -------------------------------------------------------------------------
  Minimal (unoptimized) example of PatchMatch. Requires that ImageMagick be installed.

  To improve generality you can:
   - Use whichever distance function you want in dist(), e.g. compare SIFT descriptors computed densely.
   - Search over a larger search space, such as rotating+scaling patches (see MATLAB mex for examples of both)

  To improve speed you can:
   - Turn on optimizations (/Ox /Oi /Oy /fp:fast or -O6 -s -ffast-math -fomit-frame-pointer -fstrength-reduce -msse2 -funroll-loops)
   - Use the MATLAB mex which is already tuned for speed
   - Use multiple cores, tiling the input. See our publication "The Generalized PatchMatch Correspondence Algorithm"
   - Tune the distance computation: manually unroll loops for each patch size, use SSE instructions (see readme)
   - Precompute random search samples (to avoid using rand, and mod)
   - Move to the GPU
  -------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv.hpp>

using namespace cv;

#ifndef MAX
#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))
#endif

/* -------------------------------------------------------------------------
   BITMAP: Minimal image class
   ------------------------------------------------------------------------- */


   /* -------------------------------------------------------------------------
	  PatchMatch, using L2 distance between upright patches that translate only
	  ------------------------------------------------------------------------- */

int g_PathSize = 5;
int g_tau = 0;
int pm_iters = 5;

#define XY_TO_INT(x, y) (((y)<<12)|(x))
#define INT_TO_X(v) ((v)&((1<<12)-1))
#define INT_TO_Y(v) ((v)>>12)

/* Measure distance between 2 patches with upper left corners (ax, ay) and (bx, by), terminating early if we exceed a cutoff distance.
   You could implement your own descriptor here. */
float dist(const Mat & a, const Mat & b,  const Mat & mask, int ax, int ay, int bx, int by) {
	int ans = 0;

	Rect RE2(bx, by, g_PathSize, g_PathSize);
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

	if (unValidNum * 10 > CurMask.rows * CurMask.cols)
	{
		return 1e10;
	}

	Rect RE(ax, ay, g_PathSize, g_PathSize);

	Mat patchA = a(RE);
	Mat patchB = b(RE2);

	return norm(patchA, patchB);
}

void improve_guess(const Mat & a, const Mat &  b, const Mat & mask, int ax, int ay, int &xbest, int &ybest, float &dbest, int bx, int by)
{
	// 如果范围都在过小的范围内，则直接跳过
	if ((ax - bx)*(ax - bx) + (ay - by) * (ay - by) < g_tau)
		return;

	float d = dist(a, b, mask, ax, ay, bx, by);
	if (d < dbest)
	{
		dbest = d;
		xbest = bx;
		ybest = by;
	}
}

/* Match image a to image b, returning the nearest neighbor field mapping a => b coords, stored in an RGB 24-bit image as (by<<12)|bx. */
void patchmatch(const Mat & a, const Mat & b,  const Mat & mask, int tau, Mat & ann, Mat &annd, int patch_w)
{
	g_PathSize = patch_w;
	g_tau = tau;
	/* Initialize with random nearest neighbor field (NNF). */

	ann = Mat::zeros(a.rows, a.cols,  CV_32SC1);
	annd = Mat::zeros(a.rows,  a.cols, CV_32FC1);


	int aew = a.cols - patch_w + 1, aeh = a.rows - patch_w + 1;       /* Effective width and height (possible upper left corners of patches). */
	int bew = b.cols - patch_w + 1, beh = b.rows - patch_w + 1;


	for (int ay = 0; ay < aeh; ay++) 
	{
		for (int ax = 0; ax < aew; ax++) 
		{
			// 最小是tau
			int bx = rand() % bew;
			int by = rand() % beh;

			while (bx * bx + by * by < g_tau) // 重新随机
			{
				bx = rand() % bew;
				by = rand() % beh;
			}

			ann.at<int>(ay,ax) = XY_TO_INT(bx, by);
			annd.at<float>(ay, ax) = dist(a, b, mask, ax, ay, bx, by);
		}
	}
	for (int iter = 0; iter < pm_iters; iter++) 
	{
		/* In each iteration, improve the NNF, by looping in scanline or reverse-scanline order. */
		int ystart = 0, yend = aeh, ychange = 1;
		int xstart = 0, xend = aew, xchange = 1;
		if (iter % 2 == 1) {
			xstart = xend - 1; xend = -1; xchange = -1;
			ystart = yend - 1; yend = -1; ychange = -1;
		}
		for (int ay = ystart; ay != yend; ay += ychange) 
		{
			for (int ax = xstart; ax != xend; ax += xchange)
			{
				/* Current (best) guess. */
				int v = ann.at<int>(ay, ax);
				int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
				float dbest = annd.at<float>(ay, ax);

				/* Propagation: Improve current guess by trying instead correspondences from left and above (below and right on odd iterations). */
				if ((unsigned)(ax - xchange) < (unsigned)aew)
				{
					int vp = ann.at<int>(ay, ax - xchange);
					int xp = INT_TO_X(vp) + xchange, yp = INT_TO_Y(vp);
					if ((unsigned)xp < (unsigned)bew)
 {
						improve_guess(a, b, mask, ax, ay, xbest, ybest, dbest, xp, yp);
					}
				}

				if ((unsigned)(ay - ychange) < (unsigned)aeh)
				{
					int vp = ann.at<int>(ay -ychange, ax );
					int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + ychange;
					if ((unsigned)yp < (unsigned)beh) 
					{
						improve_guess(a, b, mask, ax, ay, xbest, ybest, dbest, xp, yp);
					}
				}

				/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
				int rs_start = 1e10;
				if (rs_start > MAX(b.cols, b.rows)) { rs_start = MAX(b.cols, b.rows); }
				for (int mag = rs_start; mag >= 1; mag /= 2) {
					/* Sampling window */
					int xmin = MAX(xbest - mag, 0), xmax = MIN(xbest + mag + 1, bew);
					int ymin = MAX(ybest - mag, 0), ymax = MIN(ybest + mag + 1, beh);
					int xp = xmin + rand() % (xmax - xmin);
					int yp = ymin + rand() % (ymax - ymin);
					improve_guess(a, b, mask, ax, ay, xbest, ybest, dbest, xp, yp);
				}

				ann.at<int>(ay,ax) = XY_TO_INT(xbest, ybest);
				annd.at<float>(ay, ax) = dbest;

			}
		}
	}
}