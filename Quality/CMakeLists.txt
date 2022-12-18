cmake_minimum_required(VERSION 3.1)

project(shadow_removing)

find_package(OpenCV REQUIRED)

include_directories(../Shadow_Removing)

add_executable(make_stat quality.cpp ../Shadow_Removing/remove_shadow.cpp ../Shadow_Removing/remove_shadow.h)
add_executable(make_rate convergence_rate.cpp ../Shadow_Removing/remove_shadow.cpp ../Shadow_Removing/remove_shadow.h)

target_link_libraries(make_stat PRIVATE ${OpenCV_LIBS})
target_link_libraries(make_rate PRIVATE ${OpenCV_LIBS})