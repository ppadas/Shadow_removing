cmake_minimum_required(VERSION 3.1)

project(shadow_removing)

find_package(OpenCV REQUIRED)

include_directories(./)

add_executable(test test.cpp remove_shadow.h remove_shadow.cpp)

target_link_libraries(test PRIVATE ${OpenCV_LIBS})