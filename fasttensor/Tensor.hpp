#pragma once

#include "Device.hpp"
#include "Simd/Simd.hpp"
#include "TensorExpression.hpp"
#include "TensorStorage.hpp"
#include "Transforms/GetCoeff.hpp"
#include "Transforms/GetPacket.hpp"
#include "UnrollUtils.hpp"
#include "boost/yap/algorithm.hpp"

namespace yap = boost::yap;
namespace hana = boost::hana;

namespace fasttensor {

template <typename ElementType, int Rank>
class Tensor : public TensorExpression<yap::expr_kind::terminal,
                                       hana::tuple<TensorStorage<ElementType, Rank>>> {
public:
  using TStorage = TensorStorage<ElementType, Rank>;

  Tensor(std::array<std::ptrdiff_t, Rank> dimensions)
      : TensorExpression<yap::expr_kind::terminal, hana::tuple<TStorage>>{
            hana::tuple<TStorage>(TStorage(dimensions))} {}

  auto &storage() { return yap::value(*this); }

  template <typename... Index>
  ElementType &operator()(Index... indices) {
    return storage().getCoeff(std::array<std::ptrdiff_t, Rank>{indices...});
  }

  auto num_elements() { return storage().num_elements(); }

  const auto &dimensions() { return storage().dimensions(); }

  template <yap::expr_kind OtherExprKind, typename OtherTuple>
  Tensor &operator=(TensorExpression<OtherExprKind, OtherTuple> const &other) {
    if constexpr (device_type == DeviceType::Simd) {
      auto &_storage = storage();
      auto num_packets = _storage.num_elements() / simd::PacketTraits<ElementType>::size;
      for (std::ptrdiff_t i = 0; i < num_packets; ++i) {
        _storage.storePacket(i, yap::transform(other, GetPacket{i}));
      }
    }
    return *this;
  }
};

} // namespace fasttensor
