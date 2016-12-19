#include "demosift.h"



static void help()
{
	std::cout << "\n This program demonstrates work of SIFT algorithm." << std::endl;
}


const cv::String keys =
"{help h usage ?    |               | Help info                         }"
"{@ref_image        | img/ref.jpg   | Reference image.                  }"
"{@test_image       | img/test1.jpg | Test image.                       }"
"{nfeatures         | 2000          | Number of features.               }"
"{nOctaveLayers     | 3             | Number of octaves in the pyramid. }"
"{contrastThreshold | 0.04          | Contrast threshold.               }"
"{edgeThreshold     | 10            | Edge threshold                    }"
"{sigma             | 1.6           | Init sigma.                       }"
;


bool load_image(cv::Mat& image, const std::string& image_path)
{
	image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);

	if (image.empty()) {
		std::cerr << "Incorrect image path: " << image_path << std::endl;
		return false;
	}

	return true;
}


void demoSIFT(int argc, char** argv)
{
	help();

	// Parse parameters

	cv::CommandLineParser parser(argc, argv, keys);

	if (parser.has("help")) {
		std::cerr << "Incorrect parameters" << std::endl;
		parser.printMessage();
		return;
	}

	const std::string ref_file = parser.get<std::string>(0);
	const std::string test_file = parser.get<std::string>(1);

	int nfeatures = parser.get<size_t>("nfeatures");
	int nOctaveLayers = parser.get<size_t>("nOctaveLayers");
	double contrastThreshold = parser.get<double>("contrastThreshold");
	int int_contrastThreshold = 4;
	int edgeThreshold = parser.get<size_t>("edgeThreshold");
	double sigma = parser.get<double>("sigma");
	int int_sigma = 16;
	int countPoints = 100;

	// Load images
	cv::Mat ref_image;
	cv::Mat test_image;

	if (!load_image(ref_image, ref_file))
		return;
	if (!load_image(test_image, test_file))
		return;

	namedWindow("test_image", cv::WINDOW_NORMAL);
	imshow("test_image", test_image);
	createTrackbar("countPoints", "test_image", &countPoints, 500);

	namedWindow("ref_image", cv::WINDOW_NORMAL);
	imshow("ref_image", ref_image);
	createTrackbar("nfeatures", "ref_image", &nfeatures, 10000);
	createTrackbar("nOctaveLayers", "ref_image", &nOctaveLayers, 20);
	createTrackbar("contrastThreshold", "ref_image", &int_contrastThreshold, 100);
	createTrackbar("edgeThreshold", "ref_image", &edgeThreshold, 50);
	createTrackbar("sigma", "ref_image", &int_sigma, 50);

	for (size_t i = 0; ; ++i) {

		contrastThreshold = int_contrastThreshold / 100.0;
		sigma = int_sigma / 10.0;

		SIFT sift(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
		std::cout << std::endl << std::endl << "Number of iteration: " << i << std::endl;
		sift.printParams();

		auto start = std::chrono::system_clock::now();

		std::vector<KeyPoint> ref_points, test_points;
		Mat ref_descriptors, test_descriptors;
		sift.detectAndCompute(ref_image, Mat::ones(ref_image.size(), CV_8U), ref_points, ref_descriptors);

		std::chrono::duration<double> elapsed_seconds =
				std::chrono::system_clock::now() - start;

		std::cout << "Ref descriptors computed. Elapsed time: " +
					 std::to_string(elapsed_seconds.count()) + "s. "
				  << "Number of points: " << ref_points.size() << std::endl;

		start = std::chrono::system_clock::now();

		sift.detectAndCompute(test_image, Mat::ones(test_image.size(), CV_8U), test_points, test_descriptors);

		elapsed_seconds = std::chrono::system_clock::now() - start;

		std::cout << "Test descriptors computed. Elapsed time: " +
					 std::to_string(elapsed_seconds.count()) + "s. "
				  << "Number of points: " << test_points.size() << std::endl;

		start = std::chrono::system_clock::now();

		std::cout << "FlannBasedMatcher." << std::endl;

		FlannBasedMatcher matcher;
		std::vector<DMatch> matches;
		matcher.match(ref_descriptors, test_descriptors, matches);

		std::sort(matches.begin(), matches.end(), [](const DMatch& mat1, const DMatch& mat2){ return mat1.distance < mat2.distance;});

		std::vector<DMatch> good_matches;

		for( int i = 0; i < ref_descriptors.rows; i++ )
			if(good_matches.size() < (size_t)countPoints)
				good_matches.push_back(matches[i]);

		elapsed_seconds = std::chrono::system_clock::now() - start;

		std::cout << "Matches computed. Elapsed time: " +
					 std::to_string(elapsed_seconds.count()) + "s. "
				  << "Number of mathes: " << countPoints << std::endl;

		Mat img_matches;
		drawMatches(ref_image, ref_points, test_image, test_points,
					good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
					std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

		namedWindow("Good Matches", CV_WINDOW_NORMAL);
		imshow("Good Matches", img_matches);

        std::vector<Point2f> obj;
        std::vector<Point2f> scene;

        for( int i = 0; i < good_matches.size(); i++ ) {
          obj.push_back( ref_points[ good_matches[i].queryIdx ].pt );
          scene.push_back( test_points[ good_matches[i].trainIdx ].pt );
        }

        Mat H = findHomography( obj, scene, CV_RANSAC );

		if (!H.empty()) {
			Mat diff;
			warpPerspective(ref_image, diff, H, test_image.size());

			namedWindow("Diff", CV_WINDOW_NORMAL);
			imshow("Diff", cv::abs(test_image-diff));

            cv::Scalar mean, stddev;
            cv::meanStdDev(diff, mean, stddev);
            std::cout << "Mean: " << mean[0] << "   StdDev: " << stddev[0] << std::endl;
        } else {
            std::cout << "H is empty." << std::endl;
        }

		char c = (char)cv::waitKey(0);
		if (c == 27)
			return;
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////////

struct KeypointResponseGreaterThanThreshold
{
	KeypointResponseGreaterThanThreshold(float _value) :
		value(_value)
	{
	}
	inline bool operator()(const KeyPoint& kpt) const
	{
		return kpt.response >= value;
	}
	float value;
};

struct KeypointResponseGreater
{
	inline bool operator()(const KeyPoint& kp1, const KeyPoint& kp2) const
	{
		return kp1.response > kp2.response;
	}
};

// takes keypoints and culls them by the response
void KeyPointsFilter::retainBest(std::vector<KeyPoint>& keypoints, int n_points)
{
	//this is only necessary if the keypoints size is greater than the number of desired points.
	if (n_points >= 0 && keypoints.size() > (size_t)n_points)
	{
		if (n_points == 0)
		{
			keypoints.clear();
			return;
		}
		//first use nth element to partition the keypoints into the best and worst.
		std::nth_element(keypoints.begin(), keypoints.begin() + n_points, keypoints.end(), KeypointResponseGreater());
		//this is the boundary response, and in the case of FAST may be ambigous
		float ambiguous_response = keypoints[n_points - 1].response;
		//use std::partition to grab all of the keypoints with the boundary response.
		std::vector<KeyPoint>::const_iterator new_end =
			std::partition(keypoints.begin() + n_points, keypoints.end(),
				KeypointResponseGreaterThanThreshold(ambiguous_response));
		//resize the keypoints, given this new end point. nth_element and partition reordered the points inplace
		keypoints.resize(new_end - keypoints.begin());
	}
}

struct RoiPredicate
{
	RoiPredicate(const Rect& _r) : r(_r)
	{}

	bool operator()(const KeyPoint& keyPt) const
	{
		return !r.contains(keyPt.pt);
	}

	Rect r;
};

void KeyPointsFilter::runByImageBorder(std::vector<KeyPoint>& keypoints, Size imageSize, int borderSize)
{
	if (borderSize > 0)
	{
		if (imageSize.height <= borderSize * 2 || imageSize.width <= borderSize * 2)
			keypoints.clear();
		else
			keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(),
				RoiPredicate(Rect(Point(borderSize, borderSize),
					Point(imageSize.width - borderSize, imageSize.height - borderSize)))),
				keypoints.end());
	}
}

struct SizePredicate
{
	SizePredicate(float _minSize, float _maxSize) : minSize(_minSize), maxSize(_maxSize)
	{}

	bool operator()(const KeyPoint& keyPt) const
	{
		float size = keyPt.size;
		return (size < minSize) || (size > maxSize);
	}

	float minSize, maxSize;
};

void KeyPointsFilter::runByKeypointSize(std::vector<KeyPoint>& keypoints, float minSize, float maxSize)
{
	CV_Assert(minSize >= 0);
	CV_Assert(maxSize >= 0);
	CV_Assert(minSize <= maxSize);

	keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), SizePredicate(minSize, maxSize)),
		keypoints.end());
}

class MaskPredicate
{
public:
	MaskPredicate(const Mat& _mask) : mask(_mask) {}
	bool operator() (const KeyPoint& key_pt) const
	{
		return mask.at<uchar>((int)(key_pt.pt.y + 0.5f), (int)(key_pt.pt.x + 0.5f)) == 0;
	}

private:
	const Mat mask;
	MaskPredicate& operator=(const MaskPredicate&);
};

void KeyPointsFilter::runByPixelsMask(std::vector<KeyPoint>& keypoints, const Mat& mask)
{
	if (mask.empty())
		return;

	keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), MaskPredicate(mask)), keypoints.end());
}

struct KeyPoint_LessThan
{
	KeyPoint_LessThan(const std::vector<KeyPoint>& _kp) : kp(&_kp) {}
	bool operator()(int i, int j) const
	{
		const KeyPoint& kp1 = (*kp)[i];
		const KeyPoint& kp2 = (*kp)[j];
		if (kp1.pt.x != kp2.pt.x)
			return kp1.pt.x < kp2.pt.x;
		if (kp1.pt.y != kp2.pt.y)
			return kp1.pt.y < kp2.pt.y;
		if (kp1.size != kp2.size)
			return kp1.size > kp2.size;
		if (kp1.angle != kp2.angle)
			return kp1.angle < kp2.angle;
		if (kp1.response != kp2.response)
			return kp1.response > kp2.response;
		if (kp1.octave != kp2.octave)
			return kp1.octave > kp2.octave;
		if (kp1.class_id != kp2.class_id)
			return kp1.class_id > kp2.class_id;

		return i < j;
	}
	const std::vector<KeyPoint>* kp;
};

void KeyPointsFilter::removeDuplicated(std::vector<KeyPoint>& keypoints)
{
	int i, j, n = (int)keypoints.size();
	std::vector<int> kpidx(n);
	std::vector<uchar> mask(n, (uchar)1);

	for (i = 0; i < n; i++)
		kpidx[i] = i;
	std::sort(kpidx.begin(), kpidx.end(), KeyPoint_LessThan(keypoints));
	for (i = 1, j = 0; i < n; i++)
	{
		KeyPoint& kp1 = keypoints[kpidx[i]];
		KeyPoint& kp2 = keypoints[kpidx[j]];
		if (kp1.pt.x != kp2.pt.x || kp1.pt.y != kp2.pt.y ||
			kp1.size != kp2.size || kp1.angle != kp2.angle)
			j = i;
		else
			mask[kpidx[i]] = 0;
	}

	for (i = j = 0; i < n; i++)
	{
		if (mask[i])
		{
			if (i != j)
				keypoints[j] = keypoints[i];
			j++;
		}
	}
	keypoints.resize(j);
}


// SIFT implementation /////////////////////////////////////////////////////////////////////////////


Ptr<SIFT> SIFT::create(int _nfeatures, int _nOctaveLayers,
	double _contrastThreshold, double _edgeThreshold, double _sigma)
{
	return makePtr<SIFT>(_nfeatures, _nOctaveLayers, _contrastThreshold, _edgeThreshold, _sigma);
}

void SIFT::printParams()
{
	std::cout << "Number of features. " << nfeatures << std::endl
			  << "Number of octaves in the pyramid. " << nOctaveLayers << std::endl
			  << "Contrast threshold. " << contrastThreshold << std::endl
			  << "Edge threshold. " << edgeThreshold << std::endl
			  << "Init sigma. " << sigma << std::endl;
}

//
// Detects features at extrema in DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.
void SIFT::findScaleSpaceExtrema(const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
	std::vector<KeyPoint>& keypoints) const
{
	int nOctaves = (int)gauss_pyr.size() / (nOctaveLayers + 3);
	int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);
	const int n = SIFT_ORI_HIST_BINS;
	float hist[n];
	KeyPoint kpt;

	keypoints.clear();

	for (int o = 0; o < nOctaves; o++)
		for (int i = 1; i <= nOctaveLayers; i++)
		{
			int idx = o*(nOctaveLayers + 2) + i;
			const Mat& img = dog_pyr[idx];
			const Mat& prev = dog_pyr[idx - 1];
			const Mat& next = dog_pyr[idx + 1];
			int step = (int)img.step1();
			int rows = img.rows, cols = img.cols;

			for (int r = SIFT_IMG_BORDER; r < rows - SIFT_IMG_BORDER; r++)
			{
				const sift_wt* currptr = img.ptr<sift_wt>(r);
				const sift_wt* prevptr = prev.ptr<sift_wt>(r);
				const sift_wt* nextptr = next.ptr<sift_wt>(r);

				for (int c = SIFT_IMG_BORDER; c < cols - SIFT_IMG_BORDER; c++)
				{
					sift_wt val = currptr[c];

					// find local extrema with pixel accuracy
					if (std::abs(val) > threshold &&
						((val > 0 && val >= currptr[c - 1] && val >= currptr[c + 1] &&
							val >= currptr[c - step - 1] && val >= currptr[c - step] && val >= currptr[c - step + 1] &&
							val >= currptr[c + step - 1] && val >= currptr[c + step] && val >= currptr[c + step + 1] &&
							val >= nextptr[c] && val >= nextptr[c - 1] && val >= nextptr[c + 1] &&
							val >= nextptr[c - step - 1] && val >= nextptr[c - step] && val >= nextptr[c - step + 1] &&
							val >= nextptr[c + step - 1] && val >= nextptr[c + step] && val >= nextptr[c + step + 1] &&
							val >= prevptr[c] && val >= prevptr[c - 1] && val >= prevptr[c + 1] &&
							val >= prevptr[c - step - 1] && val >= prevptr[c - step] && val >= prevptr[c - step + 1] &&
							val >= prevptr[c + step - 1] && val >= prevptr[c + step] && val >= prevptr[c + step + 1]) ||
							(val < 0 && val <= currptr[c - 1] && val <= currptr[c + 1] &&
								val <= currptr[c - step - 1] && val <= currptr[c - step] && val <= currptr[c - step + 1] &&
								val <= currptr[c + step - 1] && val <= currptr[c + step] && val <= currptr[c + step + 1] &&
								val <= nextptr[c] && val <= nextptr[c - 1] && val <= nextptr[c + 1] &&
								val <= nextptr[c - step - 1] && val <= nextptr[c - step] && val <= nextptr[c - step + 1] &&
								val <= nextptr[c + step - 1] && val <= nextptr[c + step] && val <= nextptr[c + step + 1] &&
								val <= prevptr[c] && val <= prevptr[c - 1] && val <= prevptr[c + 1] &&
								val <= prevptr[c - step - 1] && val <= prevptr[c - step] && val <= prevptr[c - step + 1] &&
								val <= prevptr[c + step - 1] && val <= prevptr[c + step] && val <= prevptr[c + step + 1])))
					{
						int r1 = r, c1 = c, layer = i;
						if (!adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
							nOctaveLayers, (float)contrastThreshold,
							(float)edgeThreshold, (float)sigma))
							continue;
						float scl_octv = kpt.size*0.5f / (1 << o);
						float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers + 3) + layer],
							Point(c1, r1),
							cvRound(SIFT_ORI_RADIUS * scl_octv),
							SIFT_ORI_SIG_FCTR * scl_octv,
							hist, n);
						float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
						for (int j = 0; j < n; j++)
						{
							int l = j > 0 ? j - 1 : n - 1;
							int r2 = j < n - 1 ? j + 1 : 0;

							if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr)
							{
								float bin = j + 0.5f * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[j] + hist[r2]);
								bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
								kpt.angle = 360.f - (float)((360.f / n) * bin);
								if (std::abs(kpt.angle - 360.f) < FLT_EPSILON)
									kpt.angle = 0.f;
								keypoints.push_back(kpt);
							}
						}
					}
				}
			}
		}
}


static void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl,
	int d, int n, float* dst)
{
	Point pt(cvRound(ptf.x), cvRound(ptf.y));
	float cos_t = cosf(ori*(float)(CV_PI / 180));
	float sin_t = sinf(ori*(float)(CV_PI / 180));
	float bins_per_rad = n / 360.f;
	float exp_scale = -1.f / (d * d * 0.5f);
	float hist_width = SIFT_DESCR_SCL_FCTR * scl;
	int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
	// Clip the radius to the diagonal of the image to avoid autobuffer too large exception
	radius = std::min(radius, (int)sqrt(((double)img.cols)*img.cols + ((double)img.rows)*img.rows));
	cos_t /= hist_width;
	sin_t /= hist_width;

	int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1), histlen = (d + 2)*(d + 2)*(n + 2);
	int rows = img.rows, cols = img.cols;

	AutoBuffer<float> buf(len * 6 + histlen);
	float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
	float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

	for (i = 0; i < d + 2; i++)
	{
		for (j = 0; j < d + 2; j++)
			for (k = 0; k < n + 2; k++)
				hist[(i*(d + 2) + j)*(n + 2) + k] = 0.;
	}

	for (i = -radius, k = 0; i <= radius; i++)
		for (j = -radius; j <= radius; j++)
		{
			// Calculate sample's histogram array coords rotated relative to ori.
			// Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
			// r_rot = 1.5) have full weight placed in row 1 after interpolation.
			float c_rot = j * cos_t - i * sin_t;
			float r_rot = j * sin_t + i * cos_t;
			float rbin = r_rot + d / 2 - 0.5f;
			float cbin = c_rot + d / 2 - 0.5f;
			int r = pt.y + i, c = pt.x + j;

			if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
				r > 0 && r < rows - 1 && c > 0 && c < cols - 1)
			{
				float dx = (float)(img.at<sift_wt>(r, c + 1) - img.at<sift_wt>(r, c - 1));
				float dy = (float)(img.at<sift_wt>(r - 1, c) - img.at<sift_wt>(r + 1, c));
				X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
				W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
				k++;
			}
		}

	len = k;
	cv::hal::fastAtan2(Y, X, Ori, len, true);
	cv::hal::magnitude32f(X, Y, Mag, len);
	cv::hal::exp32f(W, W, len);

	for (k = 0; k < len; k++)
	{
		float rbin = RBin[k], cbin = CBin[k];
		float obin = (Ori[k] - ori)*bins_per_rad;
		float mag = Mag[k] * W[k];

		int r0 = cvFloor(rbin);
		int c0 = cvFloor(cbin);
		int o0 = cvFloor(obin);
		rbin -= r0;
		cbin -= c0;
		obin -= o0;

		if (o0 < 0)
			o0 += n;
		if (o0 >= n)
			o0 -= n;

		// histogram update using tri-linear interpolation
		float v_r1 = mag*rbin, v_r0 = mag - v_r1;
		float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
		float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
		float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
		float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
		float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
		float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

		int idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + o0;
		hist[idx] += v_rco000;
		hist[idx + 1] += v_rco001;
		hist[idx + (n + 2)] += v_rco010;
		hist[idx + (n + 3)] += v_rco011;
		hist[idx + (d + 2)*(n + 2)] += v_rco100;
		hist[idx + (d + 2)*(n + 2) + 1] += v_rco101;
		hist[idx + (d + 3)*(n + 2)] += v_rco110;
		hist[idx + (d + 3)*(n + 2) + 1] += v_rco111;
	}

	// finalize histogram, since the orientation histograms are circular
	for (i = 0; i < d; i++)
		for (j = 0; j < d; j++)
		{
			int idx = ((i + 1)*(d + 2) + (j + 1))*(n + 2);
			hist[idx] += hist[idx + n];
			hist[idx + 1] += hist[idx + n + 1];
			for (k = 0; k < n; k++)
				dst[(i*d + j)*n + k] = hist[idx + k];
		}
	// copy histogram to the descriptor,
	// apply hysteresis thresholding
	// and scale the result, so that it can be easily converted
	// to byte array
	float nrm2 = 0;
	len = d*d*n;
	for (k = 0; k < len; k++)
		nrm2 += dst[k] * dst[k];
	float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
	for (i = 0, nrm2 = 0; i < k; i++)
	{
		float val = std::min(dst[i], thr);
		dst[i] = val;
		nrm2 += val*val;
	}
	nrm2 = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);

#if 1
	for (k = 0; k < len; k++)
	{
		dst[k] = saturate_cast<uchar>(dst[k] * nrm2);
	}
#else
	float nrm1 = 0;
	for (k = 0; k < len; k++)
	{
		dst[k] *= nrm2;
		nrm1 += dst[k];
	}
	nrm1 = 1.f / std::max(nrm1, FLT_EPSILON);
	for (k = 0; k < len; k++)
	{
		dst[k] = std::sqrt(dst[k] * nrm1);//saturate_cast<uchar>(std::sqrt(dst[k] * nrm1)*SIFT_INT_DESCR_FCTR);
	}
#endif
}

static void calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints,
	Mat& descriptors, int nOctaveLayers, int firstOctave)
{
	int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

	for (size_t i = 0; i < keypoints.size(); i++)
	{
		KeyPoint kpt = keypoints[i];
		int octave, layer;
		float scale;
		unpackOctave(kpt, octave, layer, scale);
		CV_Assert(octave >= firstOctave && layer <= nOctaveLayers + 2);
		float size = kpt.size*scale;
		Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
		const Mat& img = gpyr[(octave - firstOctave)*(nOctaveLayers + 3) + layer];

		float angle = 360.f - kpt.angle;
		if (std::abs(angle - 360.f) < FLT_EPSILON)
			angle = 0.f;
		calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
	}
}

//////////////////////////////////////////////////////////////////////////////////////////

SIFT::SIFT(int _nfeatures, int _nOctaveLayers,
	double _contrastThreshold, double _edgeThreshold, double _sigma)
	: nfeatures(_nfeatures), nOctaveLayers(_nOctaveLayers),
	contrastThreshold(_contrastThreshold), edgeThreshold(_edgeThreshold), sigma(_sigma)
{
}

int SIFT::descriptorSize() const
{
	return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}

int SIFT::descriptorType() const
{
	return CV_32F;
}

int SIFT::defaultNorm() const
{
	return NORM_L2;
}


void SIFT::detectAndCompute(InputArray _image, InputArray _mask,
	std::vector<KeyPoint>& keypoints,
	OutputArray _descriptors,
	bool useProvidedKeypoints)
{
	int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
	Mat image = _image.getMat(), mask = _mask.getMat();

	if (image.empty() || image.depth() != CV_8U)
		CV_Error(Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

	if (!mask.empty() && mask.type() != CV_8UC1)
		CV_Error(Error::StsBadArg, "mask has incorrect type (!=CV_8UC1)");

	if (useProvidedKeypoints)
	{
		firstOctave = 0;
		int maxOctave = INT_MIN;
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			int octave, layer;
			float scale;
			unpackOctave(keypoints[i], octave, layer, scale);
			firstOctave = std::min(firstOctave, octave);
			maxOctave = std::max(maxOctave, octave);
			actualNLayers = std::max(actualNLayers, layer - 2);
		}

		firstOctave = std::min(firstOctave, 0);
		CV_Assert(firstOctave >= -1 && actualNLayers <= nOctaveLayers);
		actualNOctaves = maxOctave - firstOctave + 1;
	}

	Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);
	std::vector<Mat> gpyr, dogpyr;
	int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(std::log((double)std::min(base.cols, base.rows)) / std::log(2.) - 2) - firstOctave;

	buildGaussianPyramid(base, gpyr, nOctaves);
	buildDoGPyramid(gpyr, dogpyr);

	if (!useProvidedKeypoints)
	{
		findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
		KeyPointsFilter::removeDuplicated(keypoints);

		if (nfeatures > 0)
			KeyPointsFilter::retainBest(keypoints, nfeatures);

		if (firstOctave < 0)
			for (size_t i = 0; i < keypoints.size(); i++)
			{
				KeyPoint& kpt = keypoints[i];
				float scale = 1.f / (float)(1 << -firstOctave);
				kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
				kpt.pt *= scale;
				kpt.size *= scale;
			}

		if (!mask.empty())
			KeyPointsFilter::runByPixelsMask(keypoints, mask);
	}
	else
	{
		// filter keypoints by mask
		KeyPointsFilter::runByPixelsMask( keypoints, mask );
	}

	if (_descriptors.needed())
	{
		//t = (double)getTickCount();
		int dsize = descriptorSize();
		_descriptors.create((int)keypoints.size(), dsize, CV_32F);
		Mat descriptors = _descriptors.getMat();

		calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
		//t = (double)getTickCount() - t;
		//printf("descriptor extraction time: %g\n", t*1000./tf);
	}
}



void SIFT::buildGaussianPyramid(const Mat& base, std::vector<Mat>& pyr, int nOctaves) const
{
	std::vector<double> sig(nOctaveLayers + 3);
	pyr.resize(nOctaves*(nOctaveLayers + 3));

	// precompute Gaussian sigmas using the following formula:
	//  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
	sig[0] = sigma;
	double k = std::pow(2., 1. / nOctaveLayers);
	for (int i = 1; i < nOctaveLayers + 3; i++)
	{
		double sig_prev = std::pow(k, (double)(i - 1))*sigma;
		double sig_total = sig_prev*k;
		sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
	}

	for (int o = 0; o < nOctaves; o++)
	{
		for (int i = 0; i < nOctaveLayers + 3; i++)
		{
			Mat& dst = pyr[o*(nOctaveLayers + 3) + i];
			if (o == 0 && i == 0)
				dst = base;
			// base of new octave is halved image from end of previous octave
			else if (i == 0)
			{
				const Mat& src = pyr[(o - 1)*(nOctaveLayers + 3) + nOctaveLayers];
				resize(src, dst, Size(src.cols / 2, src.rows / 2),
					0, 0, INTER_NEAREST);
			}
			else
			{
				const Mat& src = pyr[o*(nOctaveLayers + 3) + i - 1];
				GaussianBlur(src, dst, Size(), sig[i], sig[i]);
			}
		}
	}
}


void SIFT::buildDoGPyramid(const std::vector<Mat>& gpyr, std::vector<Mat>& dogpyr) const
{
	int nOctaves = (int)gpyr.size() / (nOctaveLayers + 3);
	dogpyr.resize(nOctaves*(nOctaveLayers + 2));

	for (int o = 0; o < nOctaves; o++)
	{
		for (int i = 0; i < nOctaveLayers + 2; i++)
		{
			const Mat& src1 = gpyr[o*(nOctaveLayers + 3) + i];
			const Mat& src2 = gpyr[o*(nOctaveLayers + 3) + i + 1];
			Mat& dst = dogpyr[o*(nOctaveLayers + 2) + i];
			subtract(src2, src1, dst, noArray(), DataType<sift_wt>::type);
		}
	}
}

