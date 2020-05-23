cmake_minimum_required(VERSION 2.6)
project(slam_evaluate)

set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE( PCL REQUIRED )
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include/ ${OPenCV_DIRS} ${PCL_INCLUDE_DIRS})
file(GLOB SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)

ADD_DEFINITIONS( ${PCL_DEFINITIONS} )

add_executable(baseORB ${SRC} ${PROJECT_SOURCE_DIR}/main/baseORB.cpp)
target_link_libraries(baseORB ${OpenCV_LIBS} ${PCL_LIBRARIES})
