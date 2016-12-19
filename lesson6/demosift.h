#ifndef DEMOSIFT_H
#define DEMOSIFT_H


#include <iostream>
#include <vector>
#include <chrono>

#include "opencv2/opencv.hpp"
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>


using namespace cv;



void demoSIFT(int argc, char **argv);



class SIFT
{
public:
	SIFT(int nfeatures = 0, int nOctaveLayers = 3,
		double contrastThreshold = 0.04, double edgeThreshold = 10,
		double sigma = 1.6);

	void printParams();

	//! returns the descriptor size in floats (128)
	int descriptorSize() const;

	//! returns the descriptor type
	int descriptorType() const;

	//! returns the default norm type
	int defaultNorm() const;

	//! finds the keypoints and computes descriptors for them using SIFT algorithm.
	//! Optionally it can compute descriptors for the user-provided keypoints
	void detectAndCompute(InputArray img, InputArray mask,
		std::vector<KeyPoint>& keypoints,
		OutputArray descriptors,
		bool useProvidedKeypoints = false);

	void buildGaussianPyramid(const Mat& base, std::vector<Mat>& pyr, int nOctaves) const;
	void buildDoGPyramid(const std::vector<Mat>& pyr, std::vector<Mat>& dogpyr) const;
	void findScaleSpaceExtrema(const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
		std::vector<KeyPoint>& keypoints) const;

	Ptr<SIFT> create(int _nfeatures, int _nOctaveLayers,
		double _contrastThreshold, double _edgeThreshold, double _sigma);

protected:
	CV_PROP_RW int nfeatures;
	CV_PROP_RW int nOctaveLayers;
	CV_PROP_RW double contrastThreshold;
	CV_PROP_RW double edgeThreshold;
	CV_PROP_RW double sigma;
};


/******************************* Defs and macros *****************************/

// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

#if 0
// intermediate type used for DoG pyramids
typedef short sift_wt;
static const int SIFT_FIXPT_SCALE = 48;
#else
// intermediate type used for DoG pyramids
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;
#endif

static inline void unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
{
	octave = kpt.octave & 255;
	layer = (kpt.octave >> 8) & 255;
	octave = octave < 128 ? octave : (-128 | octave);
	scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
}

static Mat createInitialImage(const Mat& img, bool doubleImageSize, float sigma)
{
	Mat gray, gray_fpt;
	if (img.channels() == 3 || img.channels() == 4)
		cvtColor(img, gray, COLOR_BGR2GRAY);
	else
		img.copyTo(gray);
	gray.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);

	float sig_diff;

	if (doubleImageSize)
	{
		sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f));
		Mat dbl;
		resize(gray_fpt, dbl, Size(gray.cols * 2, gray.rows * 2), 0, 0, INTER_LINEAR);
		GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
		return dbl;
	}
	else
	{
		sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f));
		GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
		return gray_fpt;
	}
}

// Computes a gradient orientation histogram at a specified pixel
static float calcOrientationHist(const Mat& img, Point pt, int radius,
	float sigma, float* hist, int n)
{
	int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1);

	float expf_scale = -1.f / (2.f * sigma * sigma);
	AutoBuffer<float> buf(len * 4 + n + 4);
	float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
	float* temphist = W + len + 2;

	for (i = 0; i < n; i++)
		temphist[i] = 0.f;

	for (i = -radius, k = 0; i <= radius; i++)
	{
		int y = pt.y + i;
		if (y <= 0 || y >= img.rows - 1)
			continue;
		for (j = -radius; j <= radius; j++)
		{
			int x = pt.x + j;
			if (x <= 0 || x >= img.cols - 1)
				continue;

			float dx = (float)(img.at<sift_wt>(y, x + 1) - img.at<sift_wt>(y, x - 1));
			float dy = (float)(img.at<sift_wt>(y - 1, x) - img.at<sift_wt>(y + 1, x));

			X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
			k++;
		}
	}

	len = k;

	// compute gradient values, orientations and the weights over the pixel neighborhood
	cv::hal::exp32f(W, W, len);
	cv::hal::fastAtan2(Y, X, Ori, len, true);
	cv::hal::magnitude32f(X, Y, Mag, len);

	for (k = 0; k < len; k++)
	{
		int bin = cvRound((n / 360.f)*Ori[k]);
		if (bin >= n)
			bin -= n;
		if (bin < 0)
			bin += n;
		temphist[bin] += W[k] * Mag[k];
	}

	// smooth the histogram
	temphist[-1] = temphist[n - 1];
	temphist[-2] = temphist[n - 2];
	temphist[n] = temphist[0];
	temphist[n + 1] = temphist[1];
	for (i = 0; i < n; i++)
	{
		hist[i] = (temphist[i - 2] + temphist[i + 2])*(1.f / 16.f) +
			(temphist[i - 1] + temphist[i + 1])*(4.f / 16.f) +
			temphist[i] * (6.f / 16.f);
	}

	float maxval = hist[0];
	for (i = 1; i < n; i++)
		maxval = std::max(maxval, hist[i]);

	return maxval;
}

//
// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
static bool adjustLocalExtrema(const std::vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
	int& layer, int& r, int& c, int nOctaveLayers,
	float contrastThreshold, float edgeThreshold, float sigma)
{
	const float img_scale = 1.f / (255 * SIFT_FIXPT_SCALE);
	const float deriv_scale = img_scale*0.5f;
	const float second_deriv_scale = img_scale;
	const float cross_deriv_scale = img_scale*0.25f;

	float xi = 0, xr = 0, xc = 0, contr = 0;
	int i = 0;

	for (; i < SIFT_MAX_INTERP_STEPS; i++)
	{
		int idx = octv*(nOctaveLayers + 2) + layer;
		const Mat& img = dog_pyr[idx];
		const Mat& prev = dog_pyr[idx - 1];
		const Mat& next = dog_pyr[idx + 1];

		Vec3f dD((img.at<sift_wt>(r, c + 1) - img.at<sift_wt>(r, c - 1))*deriv_scale,
			(img.at<sift_wt>(r + 1, c) - img.at<sift_wt>(r - 1, c))*deriv_scale,
			(next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);

		float v2 = (float)img.at<sift_wt>(r, c) * 2;
		float dxx = (img.at<sift_wt>(r, c + 1) + img.at<sift_wt>(r, c - 1) - v2)*second_deriv_scale;
		float dyy = (img.at<sift_wt>(r + 1, c) + img.at<sift_wt>(r - 1, c) - v2)*second_deriv_scale;
		float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2)*second_deriv_scale;
		float dxy = (img.at<sift_wt>(r + 1, c + 1) - img.at<sift_wt>(r + 1, c - 1) -
			img.at<sift_wt>(r - 1, c + 1) + img.at<sift_wt>(r - 1, c - 1))*cross_deriv_scale;
		float dxs = (next.at<sift_wt>(r, c + 1) - next.at<sift_wt>(r, c - 1) -
			prev.at<sift_wt>(r, c + 1) + prev.at<sift_wt>(r, c - 1))*cross_deriv_scale;
		float dys = (next.at<sift_wt>(r + 1, c) - next.at<sift_wt>(r - 1, c) -
			prev.at<sift_wt>(r + 1, c) + prev.at<sift_wt>(r - 1, c))*cross_deriv_scale;

		Matx33f H(dxx, dxy, dxs,
			dxy, dyy, dys,
			dxs, dys, dss);

		Vec3f X = H.solve(dD, DECOMP_LU);

		xi = -X[2];
		xr = -X[1];
		xc = -X[0];

		if (std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f)
			break;

		if (std::abs(xi) > (float)(INT_MAX / 3) ||
			std::abs(xr) > (float)(INT_MAX / 3) ||
			std::abs(xc) > (float)(INT_MAX / 3))
			return false;

		c += cvRound(xc);
		r += cvRound(xr);
		layer += cvRound(xi);

		if (layer < 1 || layer > nOctaveLayers ||
			c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER ||
			r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER)
			return false;
	}

	// ensure convergence of interpolation
	if (i >= SIFT_MAX_INTERP_STEPS)
		return false;

	{
		int idx = octv*(nOctaveLayers + 2) + layer;
		const Mat& img = dog_pyr[idx];
		const Mat& prev = dog_pyr[idx - 1];
		const Mat& next = dog_pyr[idx + 1];
		Matx31f dD((img.at<sift_wt>(r, c + 1) - img.at<sift_wt>(r, c - 1))*deriv_scale,
			(img.at<sift_wt>(r + 1, c) - img.at<sift_wt>(r - 1, c))*deriv_scale,
			(next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);
		float t = dD.dot(Matx31f(xc, xr, xi));

		contr = img.at<sift_wt>(r, c)*img_scale + t * 0.5f;
		if (std::abs(contr) * nOctaveLayers < contrastThreshold)
			return false;

		// principal curvatures are computed using the trace and det of Hessian
		float v2 = img.at<sift_wt>(r, c)*2.f;
		float dxx = (img.at<sift_wt>(r, c + 1) + img.at<sift_wt>(r, c - 1) - v2)*second_deriv_scale;
		float dyy = (img.at<sift_wt>(r + 1, c) + img.at<sift_wt>(r - 1, c) - v2)*second_deriv_scale;
		float dxy = (img.at<sift_wt>(r + 1, c + 1) - img.at<sift_wt>(r + 1, c - 1) -
			img.at<sift_wt>(r - 1, c + 1) + img.at<sift_wt>(r - 1, c - 1)) * cross_deriv_scale;
		float tr = dxx + dyy;
		float det = dxx * dyy - dxy * dxy;

		if (det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det)
			return false;
	}

	kpt.pt.x = (c + xc) * (1 << octv);
	kpt.pt.y = (r + xr) * (1 << octv);
	kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5) * 255) << 16);
	kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv) * 2;
	kpt.response = std::abs(contr);

	return true;
}

#endif // DEMOSIFT_H
