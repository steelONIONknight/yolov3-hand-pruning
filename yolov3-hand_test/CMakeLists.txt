cmake_minimum_required(VERSION 3.10)
project(yolov3_hand_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
list(APPEND CMAKE_PREFIX_PATH "ncnn lib path")
find_package(ncnn REQUIRED)
add_executable(yolov3_hand_test main.cpp)
target_link_libraries(yolov3_hand_test ncnn)