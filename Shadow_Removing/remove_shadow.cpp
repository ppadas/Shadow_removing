#include<remove_shadow.h>
#include <iostream>

void ShadowRemoving::RemoveShadow(
    cv::Mat& dst_image, 
    const cv::Mat& src_image) {
    LogMain_("0_rm_src.png", src_image);

    cv::Mat image_YCC;
    cv::cvtColor(src_image, image_YCC, cv::COLOR_BGR2YCrCb);
    
    cv::Mat channels[3];
    cv::split(image_YCC, channels);
    cv::Mat channel_Y = channels[0];

    if (!log_all_path_.empty()) {
        cv::Mat channel_Y_heatMap;
        GetHeatMapImage_(channel_Y_heatMap, channel_Y);
        LogSteps_("1_1_rm_Src_Y.png", channel_Y);
        LogSteps_("1_2_rm_Src_Y.png", channel_Y_heatMap);
    }

    cv::Mat dst_Y_image = cv::Mat(channel_Y.cols, channel_Y.rows, CV_8UC1, cv::Scalar(0));
    WaterFilling_(dst_Y_image, channel_Y, 1000);

    if (!log_all_path_.empty()) {
        cv::Mat channel_Y_heatMap;
        GetHeatMapImage_(channel_Y_heatMap, dst_Y_image);
        LogSteps_("2_1_rm_new_Y.png", dst_Y_image);
        LogSteps_("2_2_rm_new_Y.png", channel_Y_heatMap);
    }

    cv::Mat answer_Y = cv::Mat(channel_Y.cols, channel_Y.rows, CV_8UC1, cv::Scalar(0));
    GiveNewYChannel_(answer_Y, channel_Y, dst_Y_image);
    
    LogSteps_("3_rm_answer_Y.png", answer_Y);

    cv::Mat output_image_YCC;
    cv::Mat output_image;
    channels[0] = answer_Y;
    cv::merge(channels, 3, output_image_YCC);
    cv::cvtColor(output_image_YCC, output_image, cv::COLOR_YCrCb2BGR);

    LogMain_("4_rm_dst.png", output_image);

    return;
}

void ShadowRemoving::GetShadowMask (
    cv::Mat& shadowMask, 
    const cv::Mat& src_image) {
    LogMain_("0_mask_src.png", src_image);

    cv::Mat image_YCC;
    cv::cvtColor(src_image, image_YCC, cv::COLOR_BGR2YCrCb);
    
    cv::Mat channels[3];
    cv::split(image_YCC, channels);
    cv::Mat channel_Y = channels[0];

    if (!log_all_path_.empty()) {
        cv::Mat channel_Y_heatMap;
        GetHeatMapImage_(channel_Y_heatMap, channel_Y);
        LogSteps_("1_1_mask_Src_Y.png", channel_Y);
        LogSteps_("1_2_mask_Src_Y.png", channel_Y_heatMap);
    }

    cv::Mat dst_Y_image = cv::Mat(channel_Y.cols, channel_Y.rows, CV_8UC1, cv::Scalar(0));
    WaterFilling_(dst_Y_image, channel_Y, 1000);

    if (!log_all_path_.empty()) {
        cv::Mat channel_Y_heatMap;
        GetHeatMapImage_(channel_Y_heatMap, dst_Y_image);
        LogSteps_("2_1_mask_new_Y.png", dst_Y_image);
        LogSteps_("2_2_mask_new_Y.png", channel_Y_heatMap);
    }

    double treshold = 160; //параметр
    cv::threshold(dst_Y_image, shadowMask, treshold, 255, cv::THRESH_BINARY_INV);
    LogMain_("3_mask_dst.png", shadowMask);
}

bool ShadowRemoving::CheckConverged_(
    cv::Mat& relief, 
    cv::Mat& previous_relief) {
    cv::Mat diff = relief - previous_relief;
    float delta = 0.01; // params
    for (int x = 0; x < diff.rows; ++x) { 
        for (int y = 0; y < diff.cols; ++y) {
            if (diff.at<float>(x, y) > delta || diff.at<float>(x, y) < -delta) {
                return false;
            }
        }
    }
    return true;
}

void ShadowRemoving::GetHeatMapImage_(
    cv::Mat& dst, 
    const cv::Mat& src) {
    cv::Mat channels[3] = {src, src, src};
    cv::merge(channels, 3, dst);
    cv::applyColorMap(dst, dst, cv::COLORMAP_JET);
}

int ShadowRemoving::WaterFilling_(
    cv::Mat& dst_image, 
    const cv::Mat& src_image, 
    int max_time) {
    if (src_image.size().width == 0 || src_image.size().height == 0) {
        std::cout << "Error" << "\n";
        return -1;
    }
    cv::Mat src_copy = src_image;
    src_copy.convertTo(src_copy, CV_32F);

    cv::Mat ground_height;
	double scale_factor = 0.2;
	cv::resize(src_copy, ground_height, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);

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
                    (InvRelu_(-current_value + relief.at<float>(x - 1, y)) +
					InvRelu_(-current_value + relief.at<float>(x + 1, y)) +
					InvRelu_(-current_value + relief.at<float>(x, y - 1)) +
					InvRelu_(-current_value + relief.at<float>(x, y + 1)));
                water_function.at<float>(x, y) += pouring + delta_water;
                if (water_function.at<float>(x, y) < 0) {
                    water_function.at<float>(x, y) = 0;
                }
            }
        }
        //add_to_video(relief, video);
    }
    
	cv::Size out_size(src_copy.cols, src_copy.rows);
	resize(relief, dst_image, out_size, 0, 0, cv::INTER_LINEAR);
	dst_image.convertTo(dst_image, CV_8UC1);
    return 0;
}

int ShadowRemoving::IncreFilling_(
    cv::Mat& dst_image, 
    const cv::Mat& src_image,
    int max_time) {
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
        if (CheckConverged_(relief, previous_relief)) {
            break;
        }
        if (time == 100) {
           break;
        }
    }
    std::cout << time << "\n";
    return 0;
}

int ShadowRemoving::GiveNewYChannel_(
    cv::Mat& new_Y,
    const cv::Mat& old_Y,
    const cv::Mat& fillngResult) {
    cv::Mat old_Y_float = old_Y;
    old_Y_float.convertTo(old_Y_float, CV_32F);
    cv::Mat fillngResult_float = fillngResult;
    fillngResult_float.convertTo(fillngResult_float, CV_32F);

    new_Y = 0.85 * old_Y_float / fillngResult_float * 255;
    new_Y.convertTo(new_Y, CV_8UC1);
    return 0;
}

float ShadowRemoving::InvRelu_(float input) {
    return input > 0 ? 0 : input;
}

void ShadowRemoving::SetLogMain(const std::string& path) {
    log_main_path_ = path;
}

void ShadowRemoving::SetLogAll(const std::string& path) {
    log_all_path_ = path;
}

void ShadowRemoving::LogMain_(
    const std::string& name, 
    const cv::Mat& image) {
    if (log_main_path_.empty()) {
        return;
    }
    cv::imwrite(log_main_path_ + name, image);
}

void ShadowRemoving::LogSteps_(
    const std::string& name, 
    const cv::Mat& image) {
    if (log_all_path_.empty()) {
        return;
    }
    cv::imwrite(log_all_path_ + name, image);
}