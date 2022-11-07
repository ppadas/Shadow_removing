cmake_minimum_required(VERSION 3.1)

project(shadow_removing)

find_package(OpenCV REQUIRED)

add_executable(tmp main.cpp)

target_link_libraries(tmp PRIVATE ${OpenCV_LIBS})