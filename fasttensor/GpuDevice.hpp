#pragma once

#include "DeviceProperties.hpp"
#include <vector>

namespace fasttensor {

struct GpuDevice {
  GpuDevice() {
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    for (int i = 0; i < num_devices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      auto block_size = prop.maxThreadsPerBlock;
      auto max_blocks = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor / block_size;
      device_props.emplace_back(block_size, max_blocks);
    }
  }

  inline auto &deviceProps() const { return device_props; }

private:
  std::vector<DeviceProperties> device_props;
};

} // namespace fasttensor
