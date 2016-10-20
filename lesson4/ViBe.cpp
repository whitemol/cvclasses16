/*
    Реализация алгоритма сегментации движения ViBe.
*/

#include "ViBe.h"

using namespace cv;

const uchar BackGround = 0;
const uchar ForeGround = 255;

// Функция, вычисляющая квадрат расстояния между двумя точками.
static double computeDistanceSqr(const Point3_<uchar> &pixel,
                                 const Point3_<uchar> &sample)
{
    double sum = 0;
    sum += (pixel.x - sample.x) * (pixel.x - sample.x);
    sum += (pixel.y - sample.y) * (pixel.y - sample.y);
    sum += (pixel.z - sample.z) * (pixel.z - sample.z);
    return sum;
}

ViBe::ViBe() : history_depth_(20), sqr_rad_(20 * 20), min_overlap_(2),
               probability_(16), samples_(), generator_()
{
}

ViBe::ViBe(int history_depth, int rad, int min_overlap, int prob)
    : history_depth_(history_depth), sqr_rad_(rad*rad),
      min_overlap_(min_overlap), probability_(prob), samples_(),
      bg_mat_(), generator_()
{
}

ViBe::~ViBe()
{
    if (samples_.empty() == true)
        return;

    for (int y = 0; y < samples_.rows; ++y)
    {
        for (int x = 0; x < samples_.cols; ++x)
        {
            delete[] samples_(y, x);
        }
    }

    samples_.release();
    bg_mat_.release();
}

void ViBe::initialize(const Mat &image)
{
    samples_.release();
    bg_mat_.release();
    samples_.create(image.rows, image.cols);
    bg_mat_.create(image.rows, image.cols, CV_8UC3);

    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            samples_(y, x) = new Point3_<uchar>[history_depth_];
            // Заполняем первое значение модели значением текущего пикселя.
            samples_(y, x)[0] = {image.ptr(y)[3 * x],
                                 image.ptr(y)[3 * x + 1],
                                 image.ptr(y)[3 * x + 2]};

            //Остальные значения модели заполняем значениями соседних пикселей.
            for (int k = 1; k < history_depth_; ++k)
            {
                Point2i neib_pixel = getRandomNeiborPixel(Point2i(x, y));
                samples_(y, x)[k] = {
                    image.ptr(neib_pixel.y)[3 * neib_pixel.x],
                    image.ptr(neib_pixel.y)[3 * neib_pixel.x + 1],
                    image.ptr(neib_pixel.y)[3 * neib_pixel.x + 2]};
            }

            // Значение фона равно значению текущего пикселя.
            bg_mat_.ptr(y)[3 * x]     = image.ptr(y)[3 * x];
            bg_mat_.ptr(y)[3 * x + 1] = image.ptr(y)[3 * x + 1];
            bg_mat_.ptr(y)[3 * x + 2] = image.ptr(y)[3 * x + 2];
        }
    }
}

Point2i ViBe::getRandomNeiborPixel(const Point2i &pixel)
{
    Point2i neib_pixel = { 0 };
    do
    {
        if (pixel.x == 0)
            neib_pixel.x = generator_.uniform(0, 2);
        else if (pixel.x == samples_.cols - 1)
            neib_pixel.x = generator_.uniform(samples_.cols - 2, samples_.cols);
        else
            neib_pixel.x = generator_.uniform(pixel.x - 1, pixel.x + 2);

        if (pixel.y == 0)
            neib_pixel.y = generator_.uniform(0, 2);
        else if (pixel.y == samples_.rows - 1)
            neib_pixel.y = generator_.uniform(samples_.rows - 2, samples_.rows);
        else
            neib_pixel.y = generator_.uniform(pixel.y - 1, pixel.y + 2);

    } while ((neib_pixel.x == pixel.x) && (neib_pixel.y == pixel.y));

    return neib_pixel;
}

void ViBe::apply(const InputArray &image, OutputArray &fgmask, double learning_rate)
{
    const Mat image_ = image.getMat();
    fgmask.create(image_.rows, image_.cols, CV_8U);
    Mat fgmask_ = fgmask.getMat();

    if ((samples_.empty() == 1) || (samples_.rows != image_.rows) ||
        (samples_.cols != image_.cols))
    {
        initialize(image_);
        return;
    }

    int probability = 0;
    if (learning_rate > 0 && learning_rate <= 1)
        probability = (uint)(1.0 / learning_rate);
    else if (learning_rate == 0)
        probability = 1;
    else
        probability = probability_;

    for (int y = 0; y < image_.rows; ++y)
    {
        const uchar* src = image_.ptr(y);
        uchar* dst = fgmask_.ptr(y);
        for (int x = 0; x < image_.cols; ++x)
        {
            // Находим количество пересечений текущего значения пикселя с моделью.
            Point3_<uchar> pixel(src[x * 3], src[x * 3 + 1], src[x * 3 + 2]);

            int counter = 0;
            for (int i = 0; i < history_depth_; ++i)
            {
                Point3_<uchar> model_pixel = samples_(y, x)[i];
                double dist = computeDistanceSqr(pixel, model_pixel);
                if (dist < sqr_rad_)
                {
                    counter++;
                    if (counter >= min_overlap_)
                        break;
                }
            }

            if (counter >= min_overlap_)
            {
                dst[x] = BackGround;
                // Обновление модели текущего пикселя.
                int rand_number = generator_.uniform(0, probability);
                if (rand_number == 0)
                {
                    rand_number = generator_.uniform(0, history_depth_);
                    samples_(y, x)[rand_number] = pixel;
                    bg_mat_.ptr(y)[3 * x] = pixel.x;
                    bg_mat_.ptr(y)[3 * x + 1] = pixel.y;
                    bg_mat_.ptr(y)[3 * x + 2] = pixel.z;
                }

                // Обновление модели случайного соседа из восьмисвязной области.
                rand_number = generator_.uniform(0, probability);
                if (rand_number == 0)
                {
                    Point2i neib_pixel = getRandomNeiborPixel(Point2i(x, y));
                    rand_number = generator_.uniform(0, history_depth_);
                    samples_(neib_pixel.y, neib_pixel.x)[rand_number] = pixel;
                }
            }
            else
            {
                dst[x] = ForeGround;
            }
        }
    }
}

void ViBe::getBackgroundImage(cv::OutputArray &image) const
{
    bg_mat_.copyTo(image);
}
