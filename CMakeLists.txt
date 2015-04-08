project(LLF)

cmake_minimum_required (VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${LLF_SOURCE_DIR}/cmake)

set(LIB_MACROS "")

# Find external libraries
find_package(OpenCV)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11
                  support. Please use a different C++ compiler.")
endif()

set(CMAKE_BUILD_TYPE "Debug")
if (NOT (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "Debug"))
  set(CMAKE_BUILD_TYPE "Release")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if(    NOT cmake_build_type_tolower STREQUAL "debug"
   AND NOT cmake_build_type_tolower STREQUAL "release")
  message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug, Release (case-insensitive).")
endif()

include_directories(${LLF_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if (cmake_build_type_tolower STREQUAL "release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

set(hdrs gaussian_pyramid.h
         laplacian_pyramid.h
         opencv_utils.h
         remapping_function.h)
set(srcs gaussian_pyramid.cpp
         laplacian_pyramid.cpp
         opencv_utils.cpp
         remapping_function.cpp)

add_executable(main ${srcs} ${hdrs} main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
