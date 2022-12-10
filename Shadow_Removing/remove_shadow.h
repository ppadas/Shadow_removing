#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ximgproc/ridgefilter.hpp>

class ShadowRemoving {
public:
    ShadowRemoving() = default;

    void RemoveShadow (
        cv::Mat& dst_image, 
        const cv::Mat& src_image);

    void GetShadowMask (
        cv::Mat& shadowMask, 
        const cv::Mat& src_image);
    
    void SetLogMain(const std::string& path);

    void SetLogAll(const std::string& path);

private:

    bool CheckConverged_(
        cv::Mat& relief, 
        cv::Mat& previous_relief);

    void GetHeatMapImage_(
        cv::Mat& dst, 
        const cv::Mat& src);
    
    int WaterFilling_(
        cv::Mat& dst_image, 
        const cv::Mat& src_image, 
        int max_time = 500);
    
    int IncreFilling_(
        cv::Mat& dst_image, 
        const cv::Mat& src_image,
        int max_time = 100);

    int GiveNewYChannel_(
        cv::Mat& new_Y,
        const cv::Mat& old_Y,
        const cv::Mat& fillngResult);

    void LogMain_(
        const std::string& name, 
        const cv::Mat& image);
    
    void LogSteps_(
        const std::string& name, 
        const cv::Mat& image);

    float InvRelu_(float input);

    std::string log_main_path_ = "";
    std::string log_all_path_ = "";
};