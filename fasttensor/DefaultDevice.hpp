#pragma once

#include "DeviceProperties.hpp"
#include <vector>

namespace fasttensor {

struct DefaultDevice {
  DefaultDevice() { device_props.emplace_back(1, 1); }

private:
  std::vector<DeviceProperties> device_props;
};

} // namespace fasttensor
