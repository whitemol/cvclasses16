#include <iostream>
#include <vector>
#include <chrono>
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

static string imgname;
static Mat imgOrigin, img2Show;
static Rect rect;
static bool rectIsSet = false;
static const string winName = "Grabcut";
static const Scalar lineColor = Scalar(0, 255, 0);
static const Scalar bgdChooseColor = Scalar(0, 0, 255);
static const Scalar fgdChooseColor = Scalar(255, 0, 0);
static const int lineWidth = 2;
static int iterCount = 0;
static const int BGD_KEY = EVENT_FLAG_CTRLKEY;
static const int FGD_KEY = EVENT_FLAG_SHIFTKEY;
static vector<Point2i> fgdPoints;
static vector<Point2i> bgdPoints;

static void help()
{
    cout << "\nThis program demonstrates the grabcut algorithm step-by-step.\n"
            "Usage:\n"
            "./...";
}

static Mat prepareFrame()
{
    Mat frame;
    img2Show.copyTo(frame);
    if (rectIsSet)
    {
        rectangle(frame, rect, lineColor, lineWidth);
    }
    for (auto i : fgdPoints)
    {
        circle(frame, i, 2, fgdChooseColor);
    }
    for (auto i : bgdPoints)
    {
        circle(frame, i, 2, bgdChooseColor);
    }
    return frame;
}

static void redrawWindow()
{
    Mat frame = prepareFrame();
    imshow(winName, frame);
}

static void onMouse(int event, int x, int y, int flags, void*)
{
    static Point2i rectCorn1;
    if (x < 0 || x >= imgOrigin.cols || y < 0 || y >= imgOrigin.rows)
    {
        return;
    }

    if (event == EVENT_LBUTTONDOWN)
    {
        if (flags & BGD_KEY)
        {
            bgdPoints.push_back(Point2i(x, y));
        }
        else if (flags & FGD_KEY)
        {
            fgdPoints.push_back(Point2i(x, y));
        }
        else
        {
            rectIsSet = false;
            rectCorn1 = Point2i(x, y);
        }
    }
    else if (event == EVENT_LBUTTONUP)
    {

    }
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
    {
        if (flags & BGD_KEY)
        {
            bgdPoints.push_back(Point2i(x, y));
        }
        else if (flags & FGD_KEY)
        {
            fgdPoints.push_back(Point2i(x, y));
        }
        else
        {
            Point2i rectCorn2 = Point2i(x, y);
            // Select top left corner and calculate rectangle size
            int topLeftX = min(rectCorn1.x, rectCorn2.x);
            int topLeftY = min(rectCorn1.y, rectCorn2.y);
            int rectWidth = abs(rectCorn1.x - rectCorn2.x);
            int rectHeight = abs(rectCorn1.y - rectCorn2.y);
            rect = Rect(topLeftX, topLeftY, rectWidth, rectHeight);
            rectIsSet = true;
        }
    }
    redrawWindow();
}

static void reset()
{
    imgOrigin.copyTo(img2Show);
    rectIsSet = false;
    iterCount = 0;
    bgdPoints.clear();
    fgdPoints.clear();
    redrawWindow();
}

static void dumpImages()
{
    string prefix = "gc_" + imgname + "_" + to_string(iterCount) + "_";
    imwrite(prefix + "ui.png", prepareFrame());
    imwrite(prefix + "res.png", img2Show);
}

static void grabCutIter(bool firstIter)
{
    using namespace std::chrono;

    static Mat mask;
    static Mat bgdModel;
    static Mat fgdModel;

    high_resolution_clock::time_point startTime, endTime;

    if (firstIter)
    {
        if (rectIsSet)
        {
            mask = Mat::ones(imgOrigin.size(), CV_8UC1) * GC_BGD;
            Mat roi = mask(rect);
            roi.setTo(GC_PR_FGD);
        }
        else
        {
            mask = Mat::ones(imgOrigin.size(), CV_8UC1) * GC_PR_FGD;
        }
        bgdModel = Mat();
        fgdModel = Mat();
    }

    for (auto pt : bgdPoints)
    {
        mask.at<char>(pt.y, pt.x) = GC_BGD;
    }
    for (auto pt : fgdPoints)
    {
        mask.at<char>(pt.y, pt.x) = GC_FGD;
    }

    startTime = high_resolution_clock::now();
    int flags = firstIter? GC_INIT_WITH_MASK : GC_EVAL;
    grabCut(imgOrigin, mask, rect, bgdModel, fgdModel, 1, flags);
    endTime = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(endTime - startTime);

    iterCount++;
    img2Show = Mat();
    imgOrigin.copyTo(img2Show, (mask & GC_FGD));
    redrawWindow();
    cout << "Iteration " << iterCount << ", time: "
              << time_span.count() << " s." << endl;
    dumpImages();
}

int graphcut(int argc, char** argv)
{
    CommandLineParser parser(argc, argv,
                             "{help h | | }{ @input | ../data/fruits.jpg | }");
    if (parser.get<bool>("help"))
    {
        help();
        return 0;
    }
    imgname = parser.get<string>("@input");
    imgOrigin = imread(imgname, CV_LOAD_IMAGE_COLOR);
    if (imgOrigin.empty())
    {
        cout << "Failed to open image " << imgname << "." << endl;
        exit(EXIT_FAILURE);
    }
    imwrite("gc_" + imgname + ".png", imgOrigin);

    reset();

    namedWindow(winName, 1);
    imshow(winName, imgOrigin);
    setMouseCallback(winName, onMouse, 0);

    for(;;)
    {
        int c = waitKey(0);

        // ESC
        if ((char) c == 27)
        {
            break;
        }

        // Restore
        if ((char) c == 'r')
        {
            cout << "----Reset----" << endl;
            reset();
        }

        // Start demo
        if ((char) c == ' ')
        {
            grabCutIter(true);
        }

        // Next step
        if ((char) c == 'n')
        {
            if (iterCount > 0)
            {
                grabCutIter(false);
            }
            else
            {
                cout << "Push space to start iterations or push r to reset" << endl;
            }
        }
    }

    return 0;
}
