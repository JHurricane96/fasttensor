#pragma once

#include "GpuDeviceFunction.hpp"
#include "Memory.hpp"
#include "Simd/Simd.hpp"
#include "UnrollUtils.hpp"
#include <algorithm>
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
    _elements = AllocateMemory<ElementType>(_num_elements);
  }

  TensorStorage(const TensorStorage &other) : TensorStorage(other._dimensions) {
    std::copy(other._elements, other._elements + _num_elements, _elements);
  }

  inline friend void swap(TensorStorage &first, TensorStorage &second) noexcept {
    using std::swap;

    swap(first._dimensions, second._dimensions);
    swap(first._num_elements, second._num_elements);
    swap(first._elements, second._elements);
  }

  TensorStorage(TensorStorage &&other) noexcept : TensorStorage() { swap(*this, other); }

  inline TensorStorage &operator=(TensorStorage other) {
    swap(*this, other);
    return *this;
  }

  inline auto elements() { return _elements; }

  inline auto num_elements() { return _num_elements; }

  inline const auto &dimensions() { return _dimensions; }

  inline PacketType getPacket(std::ptrdiff_t index) const {
    return simd::Load(&_elements[index * PacketSize]);
  }

  inline void storePacket(std::ptrdiff_t index, PacketType packet) {
    simd::Store(&_elements[index * PacketSize], packet);
  }

  GPU_DEVICE_FUNC inline const ElementType &getCoeff(std::ptrdiff_t index) const {
    return _elements[index];
  }

  inline const ElementType &getCoeff(std::array<std::ptrdiff_t, Rank> indices) const {
    return _elements[utils::getIndex<Rank - 1>(_dimensions, indices)];
  }

  inline ElementType &operator()(std::array<std::ptrdiff_t, Rank> indices) {
    return _elements[utils::getIndex<Rank - 1>(_dimensions, indices)];
  }

  inline void storeCoeff(ElementType element, std::ptrdiff_t index) { _elements[index] = element; }

  ~TensorStorage() { DeallocateMemory(_elements); }

private:
  std::array<std::ptrdiff_t, Rank> _dimensions;
  std::ptrdiff_t _num_elements;
  ElementType *_elements;
};

} // namespace fasttensor
