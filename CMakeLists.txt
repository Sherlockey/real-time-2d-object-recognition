cmake_minimum_required(VERSION 3.15...3.31)
project(real-time-2d-object-recognition)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(main ./source/main.cpp)
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
