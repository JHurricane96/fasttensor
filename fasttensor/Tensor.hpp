#pragma once

#include "DefaultDevice.hpp"
#include "Device.hpp"
#include "DeviceFactory.hpp"
#include "GpuDeviceFunction.hpp"
#include "RefSelector.hpp"
#include "Simd/Simd.hpp"
#include "TensorExpression.hpp"
#include "TensorStorage.hpp"
#include "UnrollUtils.hpp"

#if defined FASTTENSOR_GPU
#  include "GpuDevice.hpp"
#endif

namespace fasttensor {

#if defined FASTTENSOR_GPU
using DefaultDeviceType = GpuDevice;
#else
using DefaultDeviceType = DefaultDevice;
#endif

template <typename ElementType, int Rank, typename DeviceType = DefaultDeviceType>
class Tensor;

template <typename T>
constexpr bool is_tensor = false;

template <typename ElementType, int Rank>
constexpr bool is_tensor<Tensor<ElementType, Rank>> = true;

template <typename T>
struct ref_selector<T, typename std::enable_if_t<is_tensor<std::remove_cv_t<T>>>> {
  using type = T &;
};

template <typename ElementType, int Rank, typename DeviceType>
class Tensor : public TensorExpression {
public:
  using TStorage = TensorStorage<ElementType, Rank>;
  using Self = Tensor<ElementType, Rank, DeviceType>;

  Tensor(std::array<std::ptrdiff_t, Rank> dimensions)
      : _storage(dimensions), _device(DeviceFactory<DeviceType>::GetDevice()) {}

  Tensor(const Self &other)
      : _storage(other._storage), _device(DeviceFactory<DeviceType>::GetDevice()) {}

  inline auto &storage() { return _storage; }

  inline const auto &storage() const { return _storage; }

  template <typename... Index>
  inline ElementType &operator()(Index... indices) {
    return _storage(std::array<std::ptrdiff_t, Rank>{indices...});
  }

  inline auto num_elements() { return _storage.num_elements(); }

  inline const auto &dimensions() { return _storage.dimensions(); }

  inline auto getPacket(std::ptrdiff_t n) const { return _storage.getPacket(n); }

  GPU_DEVICE_FUNC inline auto getCoeff(std::ptrdiff_t n) const { return _storage.getCoeff(n); }

  inline auto &device() const { return _device; }

  template <typename OtherExpr, typename = enable_if_tensor_exprs<OtherExpr>>
  inline Tensor &operator=(OtherExpr const &);

private:
  TStorage _storage;
  DeviceType _device;
};

} // namespace fasttensor
