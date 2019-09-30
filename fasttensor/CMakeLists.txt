cmake_minimum_required(VERSION 3.14.6)

add_library(fasttensor INTERFACE)

target_include_directories(fasttensor INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	$<INSTALL_INTERFACE:.>
)

set(DEVICE_TYPE "SIMD" CACHE STRING "Device type")
message("** Device type is set to ${DEVICE_TYPE}")

set(CUDA_PATH "" CACHE PATH "Path to CUDA toolkit")
set(GPU_ARCH "" CACHE STRING "CUDA GPU compute architecture")

if(DEVICE_TYPE STREQUAL "SIMD")
	target_compile_definitions(fasttensor INTERFACE FASTTENSOR_SIMD)
elseif(DEVICE_TYPE STREQUAL "GPU")
	target_compile_definitions(fasttensor INTERFACE FASTTENSOR_GPU)
	target_compile_options(fasttensor INTERFACE -x cuda -pthread --cuda-gpu-arch=sm_${GPU_ARCH} --cuda-path=${CUDA_PATH})
	target_link_options(fasttensor INTERFACE -lcudart -ldl -lrt -L${CUDA_PATH}/lib64)
endif()

target_compile_features(fasttensor INTERFACE cxx_std_17)

install(TARGETS fasttensor EXPORT fasttensor_config
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
)

install(EXPORT fasttensor_config DESTINATION lib)
install(DIRECTORY ./ DESTINATION fasttensor)
