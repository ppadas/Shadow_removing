#include<remove_shadow.h>

int main(int argc, char* argv[]) {
    ShadowRemoving s;
    std::string path = argv[1];
    std::string main_log = argv[2];
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    cv::Mat dst_image;
    s.SetLogMain(main_log);
    //s.SetLogAll("/home/krolchonok/Documents/Study/4_term/Polevoy/Shadow_removing/Shadow_Removing/build/Images/");
    s.RemoveShadow(dst_image, image);
    cv::Mat shadow_mask;
    s.GetShadowMask(shadow_mask, image);
    return 0;
}