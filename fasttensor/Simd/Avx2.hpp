#pragma once

#include <type_traits>

namespace fasttensor::simd {

constexpr int PacketSize = 32;

template <typename T>
constexpr int NumElementsInPacket = PacketSize / sizeof(T);

template <typename T>
struct PacketTraits {
  using type = T;
  static constexpr bool is_vectorizable = false;
  static constexpr int size = 1;
};

template <>
struct PacketTraits<int> {
  using type = __m256i;
  static constexpr bool is_vectorizable = true;
  static constexpr int size = NumElementsInPacket<int>;
};

template <>
struct PacketTraits<float> {
  using type = __m256;
  static constexpr bool is_vectorizable = true;
  static constexpr int size = NumElementsInPacket<float>;
};

template <>
struct PacketTraits<double> {
  using type = __m256d;
  static constexpr bool is_vectorizable = true;
  static constexpr int size = NumElementsInPacket<double>;
};

template <typename ScalarType>
inline typename PacketTraits<ScalarType>::type Load(ScalarType *source) {
  if constexpr (std::is_same_v<ScalarType, int>) {
    return _mm256_load_si256(reinterpret_cast<const PacketTraits<int>::type *>(source));
  } else if constexpr (std::is_same_v<ScalarType, float>) {
    return _mm256_load_ps(reinterpret_cast<const float *>(source));
  } else if constexpr (std::is_same_v<ScalarType, double>) {
    return _mm256_load_pd(reinterpret_cast<const double *>(source));
  }
}

template <typename ScalarType>
inline void Store(ScalarType *dest, typename PacketTraits<ScalarType>::type source) {
  if constexpr (std::is_same_v<ScalarType, int>) {
    _mm256_store_si256(reinterpret_cast<PacketTraits<int>::type *>(dest), source);
  } else if constexpr (std::is_same_v<ScalarType, float>) {
    _mm256_store_ps(dest, source);
  } else if constexpr (std::is_same_v<ScalarType, double>) {
    _mm256_store_pd(dest, source);
  }
}

template <typename PacketType>
inline PacketType Add(PacketType left, PacketType right) {
  if constexpr (std::is_same_v<PacketType, __m256i>) {
    return _mm256_add_epi32(left, right);
  } else if constexpr (std::is_same_v<PacketType, __m256>) {
    return _mm256_add_ps(left, right);
  } else if constexpr (std::is_same_v<PacketType, __m256d>) {
    return _mm256_add_pd(left, right);
  }
}

template <typename PacketType>
inline PacketType Sub(PacketType left, PacketType right) {
  if constexpr (std::is_same_v<PacketType, __m256i>) {
    return _mm256_sub_epi32(left, right);
  } else if constexpr (std::is_same_v<PacketType, __m256>) {
    return _mm256_sub_ps(left, right);
  } else if constexpr (std::is_same_v<PacketType, __m256d>) {
    return _mm256_sub_pd(left, right);
  }
}

template <typename PacketType>
inline PacketType Mult(PacketType left, PacketType right) {
  if constexpr (std::is_same_v<PacketType, __m256i>) {
    return _mm256_mullo_epi32(left, right);
  } else if constexpr (std::is_same_v<PacketType, __m256>) {
    return _mm256_mul_ps(left, right);
  } else if constexpr (std::is_same_v<PacketType, __m256d>) {
    return _mm256_mul_pd(left, right);
  }
}

template <typename PacketType>
inline PacketType Div(PacketType dividend, PacketType divisor) {
  if constexpr (std::is_same_v<PacketType, __m256>) {
    return _mm256_div_ps(dividend, divisor);
  } else if constexpr (std::is_same_v<PacketType, __m256d>) {
    return _mm256_div_pd(dividend, divisor);
  }
}

} // namespace fasttensor::simd
