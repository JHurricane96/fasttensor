cmake_minimum_required(VERSION 3.14.6)
project(tests)

SET(SOURCE_FILES
	cwiseops.cpp
	test.cpp)

add_executable(tests ${SOURCE_FILES})
target_link_libraries(tests fasttensor gtest)

set(DEVICE_TYPE "SIMD" CACHE STRING "Device type")
message("** Device type is set to ${DEVICE_TYPE}")

if(DEVICE_TYPE STREQUAL "SIMD")
	target_compile_definitions(tests PRIVATE FASTTENSOR_SIMD)
endif()

target_compile_options(tests PRIVATE "-march=native")

install(TARGETS tests DESTINATION bin)