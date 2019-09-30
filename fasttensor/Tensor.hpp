#pragma once

#include "Device.hpp"
#include "GpuDeviceFunction.hpp"
#include "RefSelector.hpp"
#include "Simd/Simd.hpp"
#include "TensorExpression.hpp"
#include "TensorStorage.hpp"
#include "UnrollUtils.hpp"

namespace fasttensor {

template <typename ElementType, int Rank>
class Tensor;

template <typename ElementType, int Rank>
struct ref_selector<Tensor<ElementType, Rank>> {
  using type = Tensor<ElementType, Rank> &;
};

template <typename ElementType, int Rank>
struct ref_selector<const Tensor<ElementType, Rank>> {
  using type = const Tensor<ElementType, Rank> &;
};

template <typename ElementType, int Rank>
class Tensor : public TensorExpression {
public:
  using TStorage = TensorStorage<ElementType, Rank>;
  using Self = Tensor<ElementType, Rank>;

  Tensor(std::array<std::ptrdiff_t, Rank> dimensions) : _storage(dimensions) {}

  Tensor(const Self &other) : _storage(other._storage) {}

  inline auto &storage() { return _storage; }

  template <typename... Index>
  inline ElementType &operator()(Index... indices) {
    return _storage(std::array<std::ptrdiff_t, Rank>{indices...});
  }

  inline auto num_elements() { return _storage.num_elements(); }

  inline const auto &dimensions() { return _storage.dimensions(); }

  inline auto getPacket(std::ptrdiff_t n) const { return _storage.getPacket(n); }

  GPU_DEVICE_FUNC inline auto getCoeff(std::ptrdiff_t n) const { return _storage.getCoeff(n); }

  template <typename OtherExpr, typename = enable_if_tensor_exprs<OtherExpr>>
  inline Tensor &operator=(OtherExpr const &);

private:
  TStorage _storage;
};

} // namespace fasttensor
