#include <remove_shadow.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace cv;

const std::string global_path = "/home/krolchonok/Documents/Study/4_term/Polevoy/Shadow_removing/Quality/build/Images/";

Scalar getMSSIM( const Mat& i1, const Mat& i2) {
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(9, 9), 1.5);
    GaussianBlur(I2, mu2, Size(9, 9), 1.5);
    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(9, 9), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(9, 9), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(9, 9), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    Mat ssim_map;
    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;
    Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
    return mssim;
}

int main(int argc, char* argv[]) {
    std::string path_to_files = argv[1];
    std::ifstream file(path_to_files);

    std::string image_name;
    std::string mask_name;
    ShadowRemoving s;
    int iter = 0;
    cv::Mat shadow_mask;
    double value = 0;
    while(file >> image_name >> mask_name) {
        cv::Mat image = cv::imread(image_name, cv::IMREAD_COLOR);
        cv::Mat shadow_mask_true = cv::imread(mask_name, cv::IMREAD_GRAYSCALE);
        std::ostringstream ss;
        ss << global_path << iter << "_";
        s.SetLogMain(ss.str());
        s.GetShadowMask(shadow_mask, image);
        value += getMSSIM(shadow_mask_true, shadow_mask)[0];
        ++iter;
    }
    std::cout << "Mean MSSIM: " << value / iter << "\n";
    
    return 0;
}