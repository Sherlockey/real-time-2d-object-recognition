cmake_minimum_required(VERSION 3.15...3.31)
project(object_recognition)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(object_recognition
    source/object_recognition.cpp
    source/k_means.cpp
    source/utilities.cpp
)
target_link_libraries(object_recognition PRIVATE ${OpenCV_LIBS})

target_include_directories(object_recognition PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
