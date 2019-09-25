#pragma once

#include "Device.hpp"
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
class Tensor : public TensorExpression {
public:
  using TStorage = TensorStorage<ElementType, Rank>;
  using Self = Tensor<ElementType, Rank>;

  Tensor(std::array<std::ptrdiff_t, Rank> dimensions) : _storage(dimensions) {}

  Tensor(const Self &other) : _storage(other._storage) {}

  auto &storage() { return _storage; }

  template <typename... Index>
  ElementType &operator()(Index... indices) {
    return _storage(std::array<std::ptrdiff_t, Rank>{indices...});
  }

  auto num_elements() { return _storage.num_elements(); }

  const auto &dimensions() { return _storage.dimensions(); }

  auto getPacket(std::ptrdiff_t n) const { return _storage.getPacket(n); }

  auto getCoeff(std::ptrdiff_t n) const { return _storage.getCoeff(n); }

  template <typename OtherExpr, typename = enable_if_tensor_exprs<OtherExpr>>
  Tensor &operator=(OtherExpr const &other) {
    if constexpr (device_type == DeviceType::Simd &&
                  simd::PacketTraits<ElementType>::is_vectorizable) {
      auto packet_size = simd::PacketTraits<ElementType>::size;
      auto num_packets = _storage.num_elements() / packet_size;
      for (std::ptrdiff_t i = 0; i < num_packets; ++i) {
        _storage.storePacket(i, other.getPacket(i));
      }
      for (std::ptrdiff_t i = num_packets * packet_size; i < _storage.num_elements(); ++i) {
        _storage.storeCoeff(other.getCoeff(i), i);
      }
    } else {
      for (std::ptrdiff_t i = 0; i < _storage.num_elements(); ++i) {
        _storage.storeCoeff(other.getCoeff(i), i);
      }
    }
    return *this;
  }

private:
  TStorage _storage;
};

} // namespace fasttensor
