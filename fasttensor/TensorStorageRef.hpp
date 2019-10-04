#pragma once

#include "GpuDeviceFunction.hpp"
#include "TensorExpression.hpp"

namespace fasttensor {

template <typename ElementType>
class TensorStorageRef : TensorExpression {
public:
  TensorStorageRef(ElementType *elements) : _elements(elements) {}

  GPU_DEVICE_FUNC inline const auto &getCoeff(std::ptrdiff_t index) const {
    return _elements[index];
  }

  inline auto elements() { return _elements; }

  inline const auto elements() const { return _elements; }

private:
  ElementType *_elements;
};

} // namespace fasttensor
