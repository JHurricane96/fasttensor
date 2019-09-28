cmake_minimum_required(VERSION 3.14.6)

SET(SOURCE_FILES
	cwiseops.cpp
	test.cpp)

add_executable(tests ${SOURCE_FILES})
target_link_libraries(tests fasttensor gtest)

set(DEVICE_TYPE "SIMD" CACHE STRING "Device type")
message("** Device type is set to ${DEVICE_TYPE}")

set(CUDA_PATH "" CACHE PATH "Path to CUDA toolkit")
set(GPU_ARCH "" CACHE STRING "CUDA GPU compute architecture")

if(DEVICE_TYPE STREQUAL "SIMD")
	target_compile_definitions(tests PRIVATE FASTTENSOR_SIMD)
elseif(DEVICE_TYPE STREQUAL "GPU")
	target_compile_definitions(tests PRIVATE FASTTENSOR_GPU)
	target_compile_options(tests PRIVATE -x cuda -pthread --cuda-gpu-arch=sm_${GPU_ARCH} --cuda-path=${CUDA_PATH})
	target_link_options(tests PRIVATE -lcudart -ldl -lrt -L${CUDA_PATH}/lib64)
endif()

target_compile_features(tests PUBLIC cxx_std_17)

target_compile_options(tests PRIVATE -march=native)

install(TARGETS tests DESTINATION bin)