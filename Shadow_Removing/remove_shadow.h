#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ximgproc/ridgefilter.hpp>

/**
 * @class ShadowRemoving
 * @brief 
 * A class that provides the ability to process images of documents. 
 * Allows you to remove shadows from the document image or get a binary shadow mask 
 * on the document image. 
 */

class ShadowRemoving {
public:
    ShadowRemoving() = default;

/**
 * @brief Shadow Removing
 *
 * Function for removing shadows on a document image.
 * The water filling algorithm is used for implementation.
 * Some inner steps can be logged. 
 *
 * @param dst_image Image to save result
 * @param src_image Image to process
 * @param max_time Number of iterations in water filling algorithm
 *
 * @return void
 * 
 * @see SetLogMain
 * @see SetLogAll
 */
    void RemoveShadow (
        cv::Mat& dst_image, 
        const cv::Mat& src_image, 
        int max_time = 1000);

/**
 * @brief Provides a shadow mask
 *
 * A function to get a shadow mask on a document image. 
 * The implementation uses a water-filling algorithm and then Otsu binarization on the resulting result.
 * Some inner steps can be logged. 
 * 
 * @param shadowMask Image to save result
 * @param src_image Image to process
 *
 * @return void
 * 
 * @see SetLogMain
 * @see SetLogAll
 */
    void GetShadowMask (
        cv::Mat& shadowMask, 
        const cv::Mat& src_image);

/**
 * @brief Saving source and result
 *
 * Function to set path to the directiry to save source and result images.
 * It must be called before the processing function is called. 
 * You can call it several times in the code, then the data for different 
 * functions will be written to different places.
 * To stop logging, call the function with an empty string.
 *
 * @param path Path to the directiry
 *
 * @return void
 * @see SetLogAll
 */
    void SetLogMain(const std::string& path);

/**
 * @brief Saving some inner steps in functions
 *
 * Function to set path to the directiry inner steps images.
 * It must be called before the processing function is called. 
 * You can call it several times in the code, then the data for different 
 * functions will be written to different places.
 * This path is only for inner steps. 
 * To save source and result too you should call SetLogMain.
 * To stop logging, call the function with an empty string.
 *
 * @param path Path to the directiry
 *
 * @return void
 * @see SetLogMain
 */
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