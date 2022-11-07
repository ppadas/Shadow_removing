#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ximgproc/ridgefilter.hpp>

int waterFilling(cv::Mat& dst_image, cv::Mat& src_image) {
    if (src_image.size().width == 0 || src_image.size().height == 0) {
        std::cout << "Error" << "\n";
        return -1;
    }

    src_image.convertTo(src_image, CV_32F);

    cv::Mat height;
	double scale_factor = 0.2;
	cv::resize(src_image, height, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);

    int height = height.rows;
	int width = height.cols;
	cv::Mat w_ = Mat(height, width, CV_32F, Scalar(0, 0, 0));
	cv::Mat G_ = Mat(height, width, CV_32F, Scalar(0, 0, 0));

    

    std::cout << height.size()<< "\n";
    return 0;
}

int main() {
    std::string path = "/home/krolchonok/Documents/Study/4_term/Polevoy/Data/example1.png";
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    
    std::string s = "src.png";

    cv::Mat image_YCC;
    cv::cvtColor(image, image_YCC, cv::COLOR_BGR2YCrCb);

    cv::Mat channels[3];
    cv::split(image_YCC, channels);
    cv::Mat channel_Y = channels[2];

    //cv::imshow(s, channel_Y);
    //cv::waitKey(0);

    cv::Mat dst_image;
    waterFilling(dst_image, channel_Y);

    //uint8_t Y_min = 255;
    //uint8_t Y_max = 0;
//
//
    //for (int x = 0; x < channel_Y.size[1]; ++x) {
    //    for (int y = 0; y < channel_Y.size[0]; ++y) {
    //        Y_max = std::max(Y_max, channel_Y.at<uint8_t>(x, y));
    //        Y_min = std::min(Y_min, channel_Y.at<uint8_t>(x, y));
    //    }
    //}
    //
    //std::cout << (int)Y_max << " " << (int)Y_min << "\n";

    //cv::imshow(s, image);
    //cv::waitKey(0);
    return 0;
}