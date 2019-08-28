cmake_minimum_required(VERSION 3.14.6)
project(fasttensor)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 17)

include(clang-format.cmake)

option(BUILD_TESTS "Set to ON to build tests" ON)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BOOST_ROOT "" CACHE PATH "Path to Boost libraries")

add_subdirectory(fasttensor)

if(BUILD_TESTS)
	add_subdirectory(ext/googletest)
	add_subdirectory(tests)
endif()
