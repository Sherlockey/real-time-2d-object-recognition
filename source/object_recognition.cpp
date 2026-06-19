/*
    Jonathon Davis
    2026-06-18
    Identifies a specified set of objects and shows the position and category of the object in an
   output image. If provided a video sequence, it performs this in real time.
*/

#include "k_means.hpp"
#include <opencv2/opencv.hpp>

/*
    Pre-process a BGR frame into a single-channel image suited for thresholding.
    Blurs to smooth regions, then darkens highly-saturated pixels so that
    strongly colored objects move away from the unsaturated background.

    @param src input BGR frame
    @param dst output single-channel image, CV_8UC1
*/
void preprocess(const cv::Mat& src, cv::Mat& dst, int median_blur_size = 3)
{
    cv::Mat blurred;
    cv::medianBlur(src, blurred, median_blur_size);

    cv::Mat hsv;
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

    // split channels
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    cv::Mat saturation = channels[1];
    cv::Mat value = channels[2];

    // darken strongly-colored pixels: scale value down by saturation
    dst.create(value.size(), CV_8UC1);
    for (int i = 0; i < value.rows; i++)
    {
        const uchar* v_row = value.ptr<uchar>(i);
        const uchar* s_row = saturation.ptr<uchar>(i);
        uchar* d_row = dst.ptr<uchar>(i);
        for (int j = 0; j < value.cols; j++)
        {
            float factor = 1.0f - (s_row[j] / 255.0f);
            d_row[j] = static_cast<uchar>(v_row[j] * factor);
        }
    }
}

/*
    Compute a threshold value using the kmeans algorithm (K=2) on a random
    sample of pixels. Returns the midpoint of the two cluster means.

    @param src single-channel input image, CV_8UC1
    @return threshold value in [0, 255]
*/
int compute_threshold(const cv::Mat& src, int sample_divisor = 16, int K = 2,
                      int max_iterations = 10, int stop_threshold = 1, int fallback = 128,
                      int seed = 123)
{
    int total = src.rows * src.cols;
    int sample_count = total / sample_divisor;

    std::vector<cv::Vec3b> samples;
    samples.reserve(sample_count);

    cv::RNG rng(seed);
    for (int i = 0; i < sample_count; i++)
    {
        int r = rng.uniform(0, src.rows);
        int c = rng.uniform(0, src.cols);
        uchar v = src.at<uchar>(r, c);
        samples.push_back(cv::Vec3b(v, v, v));
    }

    std::vector<cv::Vec3b> means;
    std::vector<int> labels(samples.size());

    int result = kmeans(samples, means, labels.data(), K, max_iterations, stop_threshold);
    if (result != 0 || means.size() < K)
    {
        return fallback;
    }

    return (means[0][0] + means[1][0]) / 2;
}

/*
    Clean up a binary image with morphological filtering.
    Close (dilate then erode) to fill small holes inside objects;

    @param src input binary image, CV_8UC1
    @param dst output cleaned binary image, CV_8UC1
*/
void clean_up(const cv::Mat& src, cv::Mat& dst)
{
    // Fill holes inside the objects.
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(src, dst, cv::MORPH_CLOSE, kernel);
}

/*
    Program entry point

    @param argc argument count with how many command line arguments provided
    @param argv argument vector is an array of strings with command line args
    @return exit status / return code sent back to OS
*/
int main(int argc, char* argv[])
{
    // open video device
    cv::VideoCapture video_capture(0);
    if (!video_capture.isOpened())
    {
        std::cout << "Unable to open video device\n";
        return -1;
    }

    // create windows and frame
    cv::namedWindow("Original Video", 1);
    cv::moveWindow("Original Video", 0, 0);
    cv::Rect original_rect = cv::getWindowImageRect("Original Video");

    cv::namedWindow("Thresholded Video", 1);
    cv::moveWindow("Thresholded Video", original_rect.width, 0);

    cv::namedWindow("Cleaned Video", 1);
    cv::moveWindow("Cleaned Video", original_rect.width * 2, 0);

    cv::Mat frame;

    // main loop
    bool should_quit = false;
    while (!should_quit)
    {
        video_capture >> frame;
        if (frame.empty())
        {
            std::cout << "Frame is empty\n";
            break;
        }

        // preprocess via blur and darken highly saturated pixels
        cv::Mat processed;
        preprocess(frame, processed, 5);

        // threshold via kmeans
        int threshold = compute_threshold(processed);
        cv::Mat thresholded;
        cv::threshold(processed, thresholded, threshold, 255, cv::THRESH_BINARY_INV);

        // clean up via morphological filtering
        cv::Mat cleaned;
        clean_up(thresholded, cleaned);

        // display
        cv::imshow("Original Video", frame);
        cv::imshow("Thresholded Video", thresholded);
        cv::imshow("Cleaned Video", cleaned);

        // handle input
        char key = cv::pollKey();
        if (key == 'q' || key == 27) // 27 = escape
        {
            should_quit = true;
        }
    }

    return 0;
}
