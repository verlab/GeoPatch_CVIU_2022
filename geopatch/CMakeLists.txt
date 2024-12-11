#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%   This file is part of https://github.com/verlab/GeoPatch_CVIU_2022
#
#    geopatch-descriptor is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    geopatch-descriptor is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with geopatch-descriptor.  If not, see <http://www.gnu.org/licenses/>.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cmake_minimum_required(VERSION 2.9 FATAL_ERROR)

project(geodesic_patch)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-w -std=c++11") #"-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV QUIET)
if (NOT OpenCV_FOUND)
  set(OpenCV_DIR ../../OpenCV3/install/share/OpenCV)
	find_package(OpenCV REQUIRED)
endif()

#find_package(Eigen3 REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#include_directories(${EIGEN3_INCLUDE_DIR})


file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/ArgParser.hpp   
                    ${PROJECT_SOURCE_DIR}/CSVParser.hpp      
                    ${PROJECT_SOURCE_DIR}/DepthProcessor.hpp  
                    ${PROJECT_SOURCE_DIR}/Mesh.cpp  
                    ${PROJECT_SOURCE_DIR}/Utils.hpp
                    ${PROJECT_SOURCE_DIR}/DepthProcesor.cpp  
                    ${PROJECT_SOURCE_DIR}/Descriptors.hpp     
                    ${PROJECT_SOURCE_DIR}/Mesh.hpp  
                    ${PROJECT_SOURCE_DIR}/Vec3.hpp   
                    ${PROJECT_SOURCE_DIR}/main.cpp  )

file(GLOB SIMPLE_DESC_SRC_FILES ${PROJECT_SOURCE_DIR}/ArgParser.hpp   
                    ${PROJECT_SOURCE_DIR}/CSVParser.hpp      
                    ${PROJECT_SOURCE_DIR}/DepthProcessor.hpp  
                    ${PROJECT_SOURCE_DIR}/Mesh.cpp  
                    ${PROJECT_SOURCE_DIR}/Utils.hpp
                    ${PROJECT_SOURCE_DIR}/DepthProcesor.cpp  
                    ${PROJECT_SOURCE_DIR}/Descriptors.hpp     
                    ${PROJECT_SOURCE_DIR}/Mesh.hpp  
                    ${PROJECT_SOURCE_DIR}/Vec3.hpp   
                    ${PROJECT_SOURCE_DIR}/simple_desc.cpp )


add_executable (geodesic_patch ${SRC_FILES})
target_link_libraries (geodesic_patch ${OpenCV_LIBS})

#######################

add_executable (simple_desc ${SIMPLE_DESC_SRC_FILES})
target_link_libraries (simple_desc ${OpenCV_LIBS})

