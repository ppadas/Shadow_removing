#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ximgproc/ridgefilter.hpp>

float inv_relu(float input_){
	return input_ > 0 ? 0 : input_;
}

void add_to_video(const cv::Mat& image, cv::VideoWriter& video) {
    cv::Mat to_video_image = image;
    to_video_image.convertTo(to_video_image, CV_8UC1);
    cv::Mat channels[3] = {to_video_image, to_video_image, to_video_image};
    cv::Mat relief_3channels;
    cv::merge(channels, 3, relief_3channels);
    cv::applyColorMap(relief_3channels, relief_3channels, cv::COLORMAP_JET);
    video << relief_3channels;
}

bool check_converged(cv::Mat& relief, cv::Mat& previous_relief) {
    cv::Mat diff = relief - previous_relief;
    float delta = 0.01;
    for (int x = 0; x < diff.cols; ++x) { 
        for (int y = 0; y < diff.rows; ++y) {
            if (diff.at<float>(x, y) > delta || diff.at<float>(x, y) < -delta) {
                return false;
            }
        }
    }
    return true;
}

int waterFilling(cv::Mat& dst_image, cv::Mat& src_image, int max_time = 500) {
    if (src_image.size().width == 0 || src_image.size().height == 0) {
        std::cout << "Error" << "\n";
        return -1;
    }

    src_image.convertTo(src_image, CV_32F);

    cv::Mat ground_height;
	double scale_factor = 0.2;
	cv::resize(src_image, ground_height, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);

    int height = ground_height.rows;
	int width = ground_height.cols;
	cv::Mat water_function = cv::Mat(height, width, CV_32FC1, cv::Scalar(0));
	cv::Mat relief = cv::Mat(height, width, CV_32FC1, cv::Scalar(0));

    cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 50, cv::Size(width, height));

    double peak_value = 0;
    float teta = 0.2;

    for (int time = 0; time < max_time; ++time) {
        relief = water_function + ground_height;
        cv::minMaxLoc(relief, NULL, &peak_value);
        
        for (int x = 1; x < ground_height.rows - 2; ++x) { 
            for (int y = 1; y < ground_height.cols - 2; ++y) {
                double current_value = relief.at<float>(x, y);
                double pouring = exp(-time) * (peak_value - current_value);
                double delta_water = teta * 
                    (inv_relu(-current_value + relief.at<float>(x - 1, y)) +
					inv_relu(-current_value + relief.at<float>(x + 1, y)) +
					inv_relu(-current_value + relief.at<float>(x, y - 1)) +
					inv_relu(-current_value + relief.at<float>(x, y + 1)));
                water_function.at<float>(x, y) += pouring + delta_water;
                if (water_function.at<float>(x, y) < 0) {
                    water_function.at<float>(x, y) = 0;
                }
            }
        }
        add_to_video(relief, video);
    }
    
	cv::Size out_size(src_image.cols, src_image.rows);
	resize(relief, dst_image, out_size, 0, 0, cv::INTER_LINEAR);
	dst_image.convertTo(dst_image, CV_8UC1);
    return 0;
}

int incre_filling(cv::Mat& dst_image, cv::Mat& src_image, cv::Mat& original_image,
    int max_time = 100) {

    cv::Mat ground_height = src_image;
    ground_height.convertTo(ground_height, CV_32FC1);
    int height = ground_height.rows; //убрать
	int width = ground_height.cols; //убрать
	cv::Mat water_function = cv::Mat(height, width, CV_32FC1, cv::Scalar(0));
	cv::Mat relief = cv::Mat(height, width, CV_32FC1, cv::Scalar(0));

    float teta = 0.2;
    int time = 0;

    while(1) {
        ++time;
        relief = water_function + ground_height;
        cv::Mat previous_relief = relief;
        for (int x = 1; x < ground_height.rows - 2; ++x) { 
            for (int y = 1; y < ground_height.cols - 2; ++y) {
                double delta_water = teta * 
                    (relief.at<float>(x - 1, y) + relief.at<float>(x + 1, y) +
					relief.at<float>(x, y - 1) + relief.at<float>(x, y + 1) - 
                    4. * relief.at<float>(x, y));
                water_function.at<float>(x, y) += delta_water;
                if (water_function.at<float>(x, y) < 0) {
                    water_function.at<float>(x, y) = 0;
                }
            }
        }
        if (check_converged(relief, previous_relief)) {
            break;
        }
        if (time == 100) {
           break;
        }
    }
    std::cout << time << "\n";

    original_image.convertTo(original_image, CV_32F);
    dst_image = 0.85 * original_image / relief * 255;
    dst_image.convertTo(dst_image, CV_8UC1);
    return 0;
}

int main(int argc, char* argv[]) {
    std::string path = argv[1];
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    
    std::string s = "src.png";

    cv::imshow("input", image);
    cv::waitKey(0);

    cv::Mat image_YCC;
    cv::cvtColor(image, image_YCC, cv::COLOR_BGR2YCrCb);

    cv::Mat channels[3];
    cv::split(image_YCC, channels);
    cv::Mat channel_Y = channels[0];

    cv::imshow("Src_Y", channel_Y);
    cv::waitKey(0);

    cv::Mat dst_Y_image = cv::Mat(channel_Y.cols, channel_Y.rows, CV_8UC1, cv::Scalar(0));
    waterFilling(dst_Y_image, channel_Y, 2500);
    cv::imshow("New_Y", dst_Y_image);
    cv::waitKey(0);

    //cv::Mat color_map;
    //applyColorMap(dst_Y_image, color_map, cv::COLORMAP_JET);
    //cv::imshow("New_Y_map", color_map);
    //cv::waitKey(0);
    
    //cv::Mat answer_Y = cv::Mat::zeros(dst_Y_image.size(), dst_Y_image.type());
    cv::Mat answer_Y = cv::Mat(channel_Y.cols, channel_Y.rows, CV_8UC1, cv::Scalar(0));
    incre_filling(answer_Y, dst_Y_image, channel_Y, 10);

    cv::imshow("Answer_Y", answer_Y);
    cv::waitKey(0);

    cv::Mat output_image_YCC;
    cv::Mat output_image;
    channels[0] = answer_Y;
    cv::merge(channels, 3, output_image_YCC);

    cv::cvtColor(output_image_YCC, output_image, cv::COLOR_YCrCb2BGR);
    cv::imshow("Answer", output_image);
    cv::waitKey(0);

    cv::imwrite("shadow_removing.jpg", output_image);
    return 0;
}