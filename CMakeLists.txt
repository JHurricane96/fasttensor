cmake_minimum_required(VERSION 3.14.6)
project(fasttensor)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(USE_LIBCPP ON CACHE BOOL "Use libc++ if compiling with clang")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND USE_LIBCPP)
	set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -stdlib=libc++")
endif()

option(FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." TRUE)
if(${FORCE_COLORED_OUTPUT})
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
		add_compile_options(-fdiagnostics-color=always)
	elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		add_compile_options(-fcolor-diagnostics)
	endif()
endif()

set(CMAKE_CXX_STANDARD 17)

include(clang-format.cmake)

option(BUILD_TESTS "Set to ON to build tests" ON)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory(fasttensor)

if(BUILD_TESTS)
	add_subdirectory(ext/googletest)
	add_subdirectory(tests)
endif()
