/*
    Jonathon Davis
    2026-06-18
    Identifies a specified set of objects and shows the position and category of the object in an
   output image. If provided a video sequence, it performs this in real time.
*/

#include "k_means.hpp"
#include <fstream>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

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
    Run connected components, filter small regions, keep the largest N,
    and renumber survivors sequentially (1..M) in a fresh label map.

    @param src input binary image, CV_8UC1 (objects = 255)
    @param out_labels output CV_32S map, survivors renumbered 1..M, else 0
    @param out_stats stats rows for survivors, in new-ID order (index 0 = region 1)
    @param out_centroids centroid (x,y) for survivors, in new-ID order
    @param min_area regions below this area are discarded
    @param max_regions keep only the N largest
    @return number of regions kept, M
*/
int segment(const cv::Mat& src, cv::Mat& out_labels, std::vector<cv::Vec3i>& out_stats,
            std::vector<cv::Point2d>& out_centroids, int min_area, int max_regions)
{
    // get connected components with stats from src image
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(src, labels, stats, centroids);

    // filter by size (skipping background)
    std::vector<std::pair<int, int>> kept; // <area, original_label>
    for (int label = 1; label < n; label++)
    {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area >= min_area)
        {
            kept.push_back({area, label});
        }
    }

    // sort in descending order
    std::sort(kept.begin(), kept.end(), std::greater<std::pair<int, int>>());
    if (max_regions > 0 && kept.size() > max_regions)
    {
        kept.resize(max_regions);
    }

    // build out_stats and out_centroids
    // map original label -> new sequential ID (1..M), background still = (0)
    std::vector<int> remap(n, 0); // <original, new_id>
    out_stats.clear();
    out_centroids.clear();
    for (int i = 0; i < kept.size(); i++)
    {
        int orig = kept[i].second;
        int new_id = i + 1;
        remap[orig] = new_id;
        out_stats.push_back(cv::Vec3i(stats.at<int>(orig, cv::CC_STAT_AREA),
                                      stats.at<int>(orig, cv::CC_STAT_LEFT),
                                      stats.at<int>(orig, cv::CC_STAT_TOP)));
        out_centroids.push_back(
            cv::Point2d(centroids.at<double>(orig, 0), centroids.at<double>(orig, 1)));
    }

    // build the renumbered label map
    out_labels.create(labels.size(), CV_32S);
    for (int i = 0; i < labels.rows; i++)
    {
        const int* src_row = labels.ptr<int>(i);
        int* dst_row = out_labels.ptr<int>(i);
        for (int j = 0; j < labels.cols; j++)
        {
            dst_row[j] = remap[src_row[j]];
        }
    }

    return static_cast<int>(kept.size());
}

/*
    Color a renumbered label map for display using evenly-spaced hues.
    Region 0 (background) is black; regions 1..M get hues spread evenly
    around the color wheel based on max_regions at full saturation and value.

    @param labels CV_32S label map with IDs 0..M
    @param dst output BGR image, CV_8UC3
    @param max_regions maximum number of regions, should match segment's max_regions
*/
void color_regions(const cv::Mat& labels, cv::Mat& dst, int max_regions)
{
    // build palette: index 0 = background (black), 1..M = evenly-spaced hues
    std::vector<cv::Vec3b> palette(max_regions + 1, cv::Vec3b(0, 0, 0));
    if (max_regions > 0)
    {
        cv::Mat hsv(1, max_regions, CV_8UC3);
        for (int i = 0; i < max_regions; i++)
        {
            // OpenCV hue range is 0..179
            uchar hue = static_cast<uchar>((i * 180) / max_regions);
            hsv.at<cv::Vec3b>(0, i) = cv::Vec3b(hue, 255, 255);
        }

        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

        for (int i = 0; i < max_regions; i++)
        {
            palette[i + 1] = bgr.at<cv::Vec3b>(0, i);
        }
    }

    // apply the palette to dst
    dst.create(labels.size(), CV_8UC3);
    for (int r = 0; r < labels.rows; r++)
    {
        const int* l_row = labels.ptr<int>(r);
        cv::Vec3b* d_row = dst.ptr<cv::Vec3b>(r);
        for (int c = 0; c < labels.cols; c++)
        {
            int id = l_row[c];
            d_row[c] = (id >= 0 && id <= max_regions) ? palette[id] : cv::Vec3b(0, 0, 0);
        }
    }
}

struct RegionFeatures
{
    cv::Point2d centroid;      // region center of mass (x, y)
    double angle;              // axis of least central moment in radians
    double percent_filled;     // region area / OBB area. is a rigid transform
    double aspect_ratio;       // OBB longer side / OBB shorter side. is a rigid transform
    double hu1;                // first Hu moment (log-scaled)
    double hu2;                // second Hu moment (log-scaled)
    cv::Point2f box_points[4]; // OBB corners
};

/*
    Compute features for a given region in a label map.

    @param labels    CV_32S label map
    @param region_id the label value of the region to analyze
    @param out       output features
    @return          true if the region had pixels, false otherwise
*/
bool compute_features(const cv::Mat& labels, int region_id, RegionFeatures& out)
{
    // build a binary mask of just this region using == operator to perform element-wise comparison
    cv::Mat mask = (labels == region_id);

    // get moments from region
    cv::Moments moments = cv::moments(mask, true);
    if (moments.m00 <= 0)
    {
        return false; // empty region
    }

    // get Hu moments from moments
    double hu[7];
    cv::HuMoments(moments, hu);

    // helper lambda to log scale the hu moments
    auto log_scale = [](double h)
    { return (h == 0.0) ? 0.0 : std::copysign(std::log10(std::abs(h)), h); };
    out.hu1 = log_scale(hu[0]);
    out.hu2 = log_scale(hu[1]);

    // centroid from raw moments
    out.centroid = cv::Point2d(moments.m10 / moments.m00, moments.m01 / moments.m00);

    // axis of least central moment from second-order central moments
    out.angle = 0.5 * std::atan2(2.0 * moments.mu11, moments.mu20 - moments.mu02);

    // project all region pixels onto the axis and its perpendicular to get
    // the oriented bounding box extents
    double cos = std::cos(out.angle);
    double sin = std::sin(out.angle);

    // along the axis
    double min_proj = std::numeric_limits<double>::max(),
           max_proj = std::numeric_limits<double>::min();
    // perpendicular to it
    double min_perp = std::numeric_limits<double>::max(),
           max_perp = std::numeric_limits<double>::min();

    for (int r = 0; r < labels.rows; r++)
    {
        const int* row = labels.ptr<int>(r);
        for (int c = 0; c < labels.cols; c++)
        {
            if (row[c] != region_id)
                continue;

            double dx = c - out.centroid.x;
            double dy = r - out.centroid.y;

            double proj = dx * cos + dy * sin;  // distance along axis
            double perp = -dx * sin + dy * cos; // distance perpendicular

            if (proj < min_proj)
            {
                min_proj = proj;
            }
            if (proj > max_proj)
            {
                max_proj = proj;
            }
            if (perp < min_perp)
            {
                min_perp = perp;
            }
            if (perp > max_perp)
            {
                max_perp = perp;
            }
        }
    }

    double height = max_proj - min_proj; // extent along the axis
    double width = max_perp - min_perp;  // extent perpendicular

    // percent filled = region area divided by OBB area
    double bbox_area = height * width;
    out.percent_filled = (bbox_area > 0) ? (moments.m00 / bbox_area) : 0.0;

    // aspect ratio = longer side / shorter side (to ensure rotation invariant)
    out.aspect_ratio =
        (width > 0 && height > 0) ? (std::max(height, width) / std::min(height, width)) : 0.0;

    // compute the four OBB corners (in order) for drawing

    // helper lambda to convert to image coordinates
    auto to_image = [&](double proj, double perp)
    {
        double x = out.centroid.x + proj * cos - perp * sin;
        double y = out.centroid.y + proj * sin + perp * cos;
        return cv::Point2f((float)x, (float)y);
    };

    out.box_points[0] = to_image(min_proj, min_perp);
    out.box_points[1] = to_image(min_proj, max_perp);
    out.box_points[2] = to_image(max_proj, max_perp);
    out.box_points[3] = to_image(max_proj, min_perp);

    return true;
}

/*
    Draw the oriented bounding box and primary axis for a region's features
    onto a color image.

    @param dst BGR image to draw on
    @param region_features the region's features
*/
void draw_features(cv::Mat& dst, const RegionFeatures& rf)
{
    // draw OBB
    for (int i = 0; i < 4; i++)
    {
        cv::line(dst, rf.box_points[i], rf.box_points[(i + 1) % 4], cv::Scalar(255, 0, 0), 2);
    }

    // draw primary axis
    double axis_len = 60.0;
    cv::Point2d p1(rf.centroid.x - axis_len * std::cos(rf.angle),
                   rf.centroid.y - axis_len * std::sin(rf.angle));
    cv::Point2d p2(rf.centroid.x + axis_len * std::cos(rf.angle),
                   rf.centroid.y + axis_len * std::sin(rf.angle));
    cv::line(dst, p1, p2, cv::Scalar(0, 255, 0), 2);
}

/*
    Pack a region's translation, scale, and rotation invariant features into a vector.

    @param f computed region features
    @return feature vector [percent_filled, aspect_ratio]
*/
std::vector<double> make_feature_vector(const RegionFeatures& f)
{
    return {f.percent_filled, f.aspect_ratio, f.hu1, f.hu2};
}

/*
    Format a feature vector for display.

    @param fv feature vector
    @return formatted string"
*/
std::string format_feature_vector(const std::vector<double>& fv)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << "[";
    for (size_t i = 0; i < fv.size(); i++)
    {
        oss << fv[i];
        if (i + 1 < fv.size())
        {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

struct TrainingExample
{
    std::string label;
    std::vector<double> features;
};

/*
    Append a labeled feature vector to the database file (CSV).
    Writes a header row if the file is new or empty.

    @param path path to the CSV database
    @param label object label
    @param rf features to store
*/
void save_training_example(const std::string& path, const std::string& label,
                           const RegionFeatures& rf)
{
    std::ifstream check(path);
    bool empty = check.peek() == std::ifstream::traits_type::eof();
    check.close();

    std::ofstream file(path, std::ios::app);
    if (!file)
    {
        std::cout << "Could not open DB file for writing: " << path << "\n";
        return;
    }

    if (empty)
    {
        file << "label,percent_filled,aspect_ratio,hu1,hu2\n";
    }

    file << label << "," << rf.percent_filled << "," << rf.aspect_ratio << "," << rf.hu1 << ","
         << rf.hu2 << "\n";
}

/*
    Loads the training examples from the given path and saves them in a vector.

    @param path path to the CSV database
    @return a vector containing all the training examples from the CSV
*/
std::vector<TrainingExample> load_training_examples(const std::string& path)
{
    std::vector<TrainingExample> result;
    std::ifstream file(path);
    std::string line;
    bool first = true;
    while (std::getline(file, line))
    {
        if (first) // skip header
        {
            first = false;
            continue;
        }

        if (line.empty())
        {
            continue;
        }

        TrainingExample te;
        std::stringstream ss(line);
        std::string token;
        std::getline(ss, te.label, ',');
        while (std::getline(ss, token, ','))
        {
            te.features.push_back(std::stod(token));
        }
        result.push_back(te);
    }
    return result;
}

/*
    Compute the per-feature standard deviation across all training examples.
    Used to scale the Euclidean distance so each feature contributes
    proportionally to its natural spread.

    @param examples loaded training database
    @return vector of standard deviations, one per feature
*/
std::vector<double> compute_feature_std_devs(const std::vector<TrainingExample>& training_examples)
{
    if (training_examples.empty())
    {
        return {};
    }

    size_t num_features = training_examples[0].features.size();
    std::vector<double> means(num_features, 0.0);
    std::vector<double> result(num_features, 0.0);

    // mean of each feature
    for (const auto& te : training_examples)
    {
        for (size_t i = 0; i < num_features; i++)
        {
            means[i] += te.features[i];
        }
    }
    for (double& mean : means)
        mean /= training_examples.size();

    // variance, then standard deviation of each feature
    for (const auto& te : training_examples)
    {
        for (size_t i = 0; i < num_features; i++)
        {
            double difference = te.features[i] - means[i];
            result[i] += difference * difference;
        }
    }
    for (double& s : result)
    {
        s = std::sqrt(s / training_examples.size());
    }

    return result;
}

/*
    Classify a feature vector by nearest neighbor under scaled Euclidean distance.

    @param feature_vector object's feature vector
    @param training_examples training database's training examples
    @param std_devs per-feature standard deviations (from compute_feature_std_devs)
    @param unknown_threshold the threshold at which or exceeding an object will be labeled "unknown"
    @return label of the nearest example, or "unknown" if DB is empty or no strong match
*/
std::string classify(const std::vector<double>& feature_vector,
                     const std::vector<TrainingExample>& training_examples,
                     const std::vector<double>& std_devs, const double unknown_treshold = 0.4)
{
    std::string result = "";
    double closest_distance = std::numeric_limits<double>::max();

    for (const auto& training_example : training_examples)
    {
        if (training_example.features.size() != feature_vector.size())
            continue;

        double sum = 0.0;
        for (size_t i = 0; i < feature_vector.size(); i++)
        {
            // 1e-9 is a popular epsilon value due to precison errors
            double s = (i < std_devs.size() && std_devs[i] > 1e-9) ? std_devs[i] : 1.0;
            double difference = (feature_vector[i] - training_example.features[i]) / s;
            sum += difference * difference;
        }
        double distance = std::sqrt(sum);

        if (distance < closest_distance)
        {
            closest_distance = distance;
            result = training_example.label;
        }
    }

    if (closest_distance > unknown_treshold)
    {
        result = "unknown";
    }

    return result;
}

/*
    Accumulates (true_label, predicted_label) pairs into a confusion matrix.
*/
struct ConfusionMatrix
{
    // counts[true_label][predicted_label] = number of times that pair occurred
    std::map<std::string, std::map<std::string, int>> counts;
    std::set<std::string> labels; // every label seen, for consistent ordering

    void record(const std::string& true_label, const std::string& predicted)
    {
        counts[true_label][predicted]++;
        labels.insert(true_label);
        labels.insert(predicted);
    }
};

/*
    Print a confusion matrix as a grid: rows = true label, columns = predicted.

    @param cm  the accumulated confusion matrix
*/
void print_confusion_matrix(const ConfusionMatrix& cm)
{
    std::vector<std::string> labels(cm.labels.begin(), cm.labels.end());

    const int w = 12; // column width

    // Header row.
    std::cout << "\n" << std::setw(w) << "true\\pred";
    for (const auto& p : labels)
    {
        std::cout << std::setw(w) << p;
    }
    std::cout << "\n";

    // One row per true label.
    for (const auto& t : labels)
    {
        std::cout << std::setw(w) << t;
        for (const auto& p : labels)
        {
            int count = 0;
            auto row = cm.counts.find(t);
            if (row != cm.counts.end())
            {
                auto cell = row->second.find(p);
                if (cell != row->second.end())
                    count = cell->second;
            }
            std::cout << std::setw(w) << count;
        }
        std::cout << "\n";
    }

    // Overall accuracy.
    int correct = 0, total = 0;
    for (const auto& t : labels)
    {
        auto row = cm.counts.find(t);
        if (row == cm.counts.end())
            continue;
        for (const auto& p : labels)
        {
            auto cell = row->second.find(p);
            if (cell == row->second.end())
                continue;
            total += cell->second;
            if (t == p)
                correct += cell->second;
        }
    }
    if (total > 0)
    {
        std::cout << "\naccuracy: " << correct << "/" << total << " = " << std::fixed
                  << std::setprecision(3) << (100.0 * correct / total) << "%\n";
    }
}

/*
    Program entry point.

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

    cv::namedWindow("Region Map", 1);
    cv::moveWindow("Region Map", original_rect.width * 3, 0);

    cv::namedWindow("Features", 1);
    cv::moveWindow("Features", 0, original_rect.height);

    const int min_region_area =
        static_cast<int>(0.0025 * video_capture.get(cv::CAP_PROP_FRAME_WIDTH) *
                         video_capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::vector<TrainingExample> training_db = load_training_examples("data/object_db.csv");
    std::vector<double> feature_std_devs = compute_feature_std_devs(training_db);

    ConfusionMatrix confusion_matrix;
    const double unknown_threshold = 0.4;

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

        // segment the image into regions and color them
        cv::Mat region_labels;
        std::vector<cv::Vec3i> region_stats;
        std::vector<cv::Point2d> region_centroids;
        const int max_regions = 5;
        int num_regions = segment(cleaned, region_labels, region_stats, region_centroids,
                                  min_region_area, max_regions);

        cv::Mat region_map;
        color_regions(region_labels, region_map, max_regions);

        // draw features
        cv::Mat features = frame.clone();
        std::vector<std::vector<double>> frame_vectors; // this frame's vectors
        std::vector<RegionFeatures> frame_features;     // this frame's RegionFeatures

        // feature loop
        for (int i = 0; i < num_regions; i++)
        {
            RegionFeatures region_features;
            if (compute_features(region_labels, i + 1, region_features))
            {
                draw_features(features, region_features);
                frame_features.push_back(region_features);

                std::vector<double> feature_vector = make_feature_vector(region_features);
                frame_vectors.push_back(feature_vector);

                // classify
                std::string classification_label =
                    classify(feature_vector, training_db, feature_std_devs, unknown_threshold);

                // display text
                cv::putText(features, classification_label,
                            cv::Point((int)region_features.centroid.x + 10,
                                      (int)region_features.centroid.y),
                            cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                std::string id_text = "ID: " + std::to_string(i + 1);
                cv::putText(features, id_text,
                            cv::Point((int)region_features.centroid.x + 10,
                                      (int)region_features.centroid.y + 18),
                            cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                // std::string feature_titles = "percent_filled, aspect_ratio";
                // cv::putText(features, feature_titles,
                //             cv::Point((int)region_features.centroid.x + 10,
                //                       (int)region_features.centroid.y + 36),
                //             cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                // std::string feature_values = format_feature_vector(feature_vector);
                // cv::putText(features, feature_values,
                //             cv::Point((int)region_features.centroid.x + 10,
                //                       (int)region_features.centroid.y + 54),
                //             cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }
        }

        // display
        cv::imshow("Original Video", frame);
        cv::imshow("Thresholded Video", thresholded);
        cv::imshow("Cleaned Video", cleaned);
        cv::imshow("Region Map", region_map);
        cv::imshow("Features", features);

        // handle input
        char key = cv::pollKey();
        if (key == 'q' || key == 27) // 27 = escape. quit program
        {
            should_quit = true;
        }
        if (key == 'p') // print feature vectors for the report
        {
            std::cout << "feature vectors: percent_filled, aspect_ratio\n";
            for (size_t i = 0; i < frame_vectors.size(); i++)
            {
                std::cout << "    region " << (i + 1) << ": "
                          << format_feature_vector(frame_vectors[i]) << "\n";
            }
        }
        if (key == 'n') // save training examples
        {
            if (frame_features.empty())
            {
                std::cout << "No regions to label.\n";
            }
            else
            {
                for (int i = 0; i < frame_features.size(); i++)
                {
                    std::cout << "Enter label for region " << (i + 1) << " (blank to skip): ";
                    std::string label;
                    std::getline(std::cin, label);

                    if (label.empty())
                    {
                        std::cout << "Skipped region " << (i + 1) << ".\n";
                        continue;
                    }

                    save_training_example("data/object_db.csv", label, frame_features[i]);
                    std::cout << "Saved '" << label << "'\n";
                }
                // reload so new examples take effect immediately
                training_db = load_training_examples("data/object_db.csv");
                feature_std_devs = compute_feature_std_devs(training_db);
            }
        }
        if (key == 'e') // evaluate: record (true, predicted) for the confusion matrix
        {
            if (frame_features.size() != 1)
            {
                std::cout << "Evaluation needs exactly one region in view (found "
                          << frame_features.size() << ").\n";
            }
            else
            {
                std::cout << "Enter TRUE label for this object: ";
                std::string true_label;
                std::getline(std::cin, true_label);

                if (!true_label.empty())
                {
                    std::vector<double> fv = make_feature_vector(frame_features[0]);
                    std::string predicted =
                        classify(fv, training_db, feature_std_devs, unknown_threshold);
                    confusion_matrix.record(true_label, predicted);
                    std::cout << "true=" << true_label << "  predicted=" << predicted
                              << (true_label == predicted ? "  [correct]" : "  [WRONG]") << "\n";
                }
            }
        }
        if (key == 'm') // print the confusion matrix
        {
            print_confusion_matrix(confusion_matrix);
        }
    }

    return 0;
}
