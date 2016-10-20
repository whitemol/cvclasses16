/*
    Алгоритм сегментации движения ViBe.
*/

#ifndef __VIBE_H__
#define __VIBE_H__

#include <opencv2/core.hpp>
#include <opencv2/video.hpp>

class ViBe : public cv::BackgroundSubtractor
{
public:
    // Функция вычисляет маску сегментации и обновляет модель.
    // TODO: вычислять prob через learningrate.
    void apply(const cv::InputArray &Image, cv::OutputArray &mask, double);
    // Функция вычисляет изображение фона.
    void getBackgroundImage(cv::OutputArray& backgroundImage) const;

    ViBe();
    ViBe(int history_depth, int radius, int min_overlap, int probability);
    ~ViBe();

private:
    int history_depth_; // Количество хранимых значений для каждого пикселя.
    int sqr_rad_; // Квадрат максимального расстояния для включения точки в модель.
    int min_overlap_; // Минимальное количество совпадений значения пикселя с моделью.
    int probability_; // Вероятность обновления модели.
    cv::Mat_<cv::Point3_<uchar>*> samples_; // Матрица для хранения значений пикселей.
    cv::Mat bg_mat_; // Матрица для хранения фона.
    cv::RNG generator_; // Генератор случайных чисел (используется равномерный закон распределения).

    // Функция инизиализации модели.
    void initialize(const cv::Mat &);
    // Функция выдаёт случайную точку из восьмисвязной области.
    cv::Point2i getRandomNeiborPixel(const cv::Point2i &);

    // Копирование запрещено
    void operator=(const ViBe &) = delete;
};

#endif // __VIBE_H__
