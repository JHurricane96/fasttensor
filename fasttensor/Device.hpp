#pragma once

namespace fasttensor {

enum class DeviceType { Normal, Simd, Gpu };

#if defined FASTTENSOR_GPU
constexpr DeviceType device_type = DeviceType::Gpu;
#elif defined FASTTENSOR_SIMD
constexpr DeviceType device_type = DeviceType::Simd;
#else
constexpr DeviceType device_type = DeviceType::Normal;
#endif

} // namespace fasttensor
