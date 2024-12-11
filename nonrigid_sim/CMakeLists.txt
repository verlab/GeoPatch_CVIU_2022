cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(nonrigid_sim)

#find_package(PCL 1.7 REQUIRED)

find_package(OpenCV QUIET)
if (NOT OpenCV_FOUND)
	set(OpenCV_DIR /homeLocal2/guipotje/opencv34/install/share/OpenCV)
	find_package(OpenCV REQUIRED)
endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
#set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -std=c++11")
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

include_directories(${EIGEN3_INCLUDE_DIR})

#add_executable (nonrigid_sim cloth_nonrigid_sim.cpp)
#target_link_libraries (nonrigid_sim ${PCL_LIBRARIES} ${OpenCV_LIBS} -lglut -lGLU -lGL)

add_executable (nonrigid_sim_auto nonrigid_sim_auto.cpp)
#target_link_libraries (nonrigid_sim_auto ${PCL_LIBRARIES} ${OpenCV_LIBS} -lglut -lGLU -lGL)
target_link_libraries (nonrigid_sim_auto ${OpenCV_LIBS} -lglut -lGLU -lGL)
