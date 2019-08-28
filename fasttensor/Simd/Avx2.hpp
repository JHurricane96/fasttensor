#pragma once

#include "SimdMacros.hpp"
#include <type_traits>

namespace fasttensor {

namespace simd {

constexpr int PacketSize = 32;

template <typename T>
constexpr int NumElementsInPacket = PacketSize / sizeof(T);

template <typename T>
struct PacketTraits {
  using type = T;
  static constexpr int size = 1;
};

template <>
struct PacketTraits<int> {
  using type = __m256i;
  static constexpr int size = NumElementsInPacket<int>;
};

template <typename ScalarType>
inline typename PacketTraits<ScalarType>::type Load(ScalarType *source) {
  if constexpr (std::is_same_v<ScalarType, int>) {
    return _mm256_load_si256(reinterpret_cast<const PacketTraits<int>::type *>(source));
  } else {
    return source;
  }
}

template <typename ScalarType>
inline void Store(ScalarType *dest, typename PacketTraits<ScalarType>::type source) {
  if constexpr (std::is_same_v<ScalarType, int>) {
    _mm256_store_si256(reinterpret_cast<PacketTraits<int>::type *>(dest), source);
  } else {
    dest = source;
  }
}

template <typename PacketType>
inline PacketType Add(PacketType left, PacketType right) {
  if constexpr (std::is_same_v<PacketType, __m256i>) {
    return _mm256_add_epi32(left, right);
  } else {
    return left + right;
  }
}

} // namespace simd

} // namespace fasttensor
