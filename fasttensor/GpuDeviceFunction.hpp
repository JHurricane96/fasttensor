#pragma once

#include "Device.hpp"

namespace fasttensor {

#ifdef FASTTENSOR_GPU
#  define GPU_DEVICE_FUNC __device__ __host__
#else
#  define GPU_DEVICE_FUNC
#endif

} // namespace fasttensor
