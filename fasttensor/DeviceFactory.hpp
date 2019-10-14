#pragma once

#include <optional>

namespace fasttensor {

template <typename DeviceType>
class DeviceFactory {
private:
  static std::optional<DeviceType> device;

public:
  static DeviceType GetDevice() {
    if (!device) {
      device = std::make_optional<DeviceType>();
    }
    return *device;
  }
};

template <typename DeviceType>
std::optional<DeviceType> DeviceFactory<DeviceType>::device = std::nullopt;

} // namespace fasttensor
