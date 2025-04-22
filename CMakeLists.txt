cmake_minimum_required(VERSION 3.11)

project(YOLOv5CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-g -O2 -Wall -Werror -Wno-unused)

find_package(OpenCV REQUIRED)

include_directories(./ ${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp)

target_link_libraries(main ${OpenCV_LIBS})
