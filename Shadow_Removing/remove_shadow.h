#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ximgproc/ridgefilter.hpp>

/**
 * @class ShadowRemoving
 * @brief 
 * Класс предоставляет возможность обрабатывать изображения документов.
 * Можно удалить тень с документа или получить бинарную маску тени. 
 */

class ShadowRemoving {
public:
    ShadowRemoving() = default;

/**
 * @brief
 *
 * Функция для удаления тени с изображения документа
 * В реализации используется алгоритм водоразлива.
 * Исходное изображение и результат, а также промежуточные шаги могут быть сохранены.
 *
 * @param dst_image Изображение, куда будет положен результат
 * @param src_image Изображение для обработки
 * @param max_time Количество итераций в алгоритме водоразлива
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
 * @brief
 *
 * Функция, предоставляющая маску тени на изображении документа.
 * В реализации используется алгоритм водоразлива и банаризация Отцу для
 * полученного после водоразлива изображения.
 * Исходное изображение и результат, а также промежуточные шаги могут быть сохранены.
 * 
 * @param shadowMask Изображение, куда будет положен результат
 * @param src_image Изображение для обработки
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
 * @brief
 *
 * Функция выставляет путь, куда будут записаны входное изображение
 * и выходное.
 * Ее следует вызывать перед вызовом функции, результат которой вы хотите сохранить.
 * Функция может быть вызвана в коде несколько раз, при каждом вызове путь, по которому
 * будут сохраняться картинки, будет меняться и результаты последующих функций 
 * будут записаны в новом месте.
 * Чтобы прекратить логирование вызовите функцию с аргументом -- пустой строкой. 
 *
 * @param path Путь к директории
 *
 * @return void
 * @see SetLogAll
 */
    void SetLogMain(const std::string& path);

/**
 * @brief
 *
 * Функция выставляет путь, куда будут записаны изображения, 
 * иллюстрирующие промежуточные шаги функции.
 * Ее следует вызывать перед вызовом функции, шаги которой вы хотите сохранить.
 * Функция может быть вызвана в коде несколько раз, при каждом вызове путь, по которому
 * будут сохраняться картинки, будет меняться и шаги последующих функций 
 * будут записаны в новом месте.
 * Функция сохраняет только промежуточные шаги, для сохранения исходного изображения 
 * и результата вызовите также SetLogMain
 * Чтобы прекратить логирование вызовите функцию с аргументом -- пустой строкой.
 *
 * @param path Путь к директории
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