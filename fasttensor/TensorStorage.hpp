#pragma once

#include "Simd/Simd.hpp"
#include "UnrollUtils.hpp"
#include "boost/hana.hpp"
#include <algorithm>
#include <execution>
#include <functional>

namespace fasttensor {

template <typename ElementType, int Rank>
class TensorStorage {
public:
  using PacketType = typename simd::PacketTraits<ElementType>::type;
  int PacketSize = simd::PacketTraits<ElementType>::size;

  TensorStorage() : _dimensions(), _num_elements(0), _elements(nullptr) {}

  TensorStorage(std::array<std::ptrdiff_t, Rank> dimensions) : _dimensions(dimensions) {
    _num_elements = utils::fold<Rank - 1, std::ptrdiff_t>(_dimensions, std::multiplies());
    _elements = reinterpret_cast<ElementType *>(operator new[](sizeof(ElementType) * _num_elements,
                                                               std::align_val_t(simd::PacketSize)));
  }

  TensorStorage(const TensorStorage &other) : TensorStorage(other._dimensions) {
    std::copy(other._elements, other._elements + _num_elements, _elements);
  }

  friend void swap(TensorStorage &first, TensorStorage &second) noexcept {
    using std::swap;

    swap(first._dimensions, second._dimensions);
    swap(first._num_elements, second._num_elements);
    swap(first._elements, second._elements);
  }

  TensorStorage(TensorStorage &&other) noexcept : TensorStorage() { swap(*this, other); }

  TensorStorage &operator=(TensorStorage other) {
    swap(*this, other);
    return *this;
  }

  auto num_elements() { return _num_elements; }

  const auto &dimensions() { return _dimensions; }

  PacketType getPacket(std::ptrdiff_t index) { return simd::Load(&_elements[index * PacketSize]); }

  void storePacket(std::ptrdiff_t index, PacketType packet) {
    simd::Store(&_elements[index * PacketSize], packet);
  }

  ElementType &getCoeff(std::ptrdiff_t index) { return _elements[index]; }

  ElementType &getCoeff(std::array<std::ptrdiff_t, Rank> indices) {
    return _elements[utils::getIndex<Rank - 1>(_dimensions, indices)];
  }

  ~TensorStorage() { operator delete[](_elements, std::align_val_t(simd::PacketSize)); }

private:
  std::array<std::ptrdiff_t, Rank> _dimensions;
  std::ptrdiff_t _num_elements;
  ElementType *_elements;
};

} // namespace fasttensor
