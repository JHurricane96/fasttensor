#pragma once

namespace fasttensor::simd {

constexpr int PacketSize = 1;

template <typename T>
constexpr int NumElementsInPacket = PacketSize / sizeof(T);

template <typename T>
struct PacketTraits {
  using type = T;
  static constexpr bool is_vectorizable = false;
  static constexpr int size = 1;
};

template <typename T>
inline T Load(T *source) {
  return source;
}

template <typename T>
inline void Store(T *dest, T *source) {
  dest = source;
}

template <typename T>
inline T Add(T *left, T *right) {
  return *left + *right;
}

template <typename T>
inline T Sub(T *left, T *right) {
  return *left - *right;
}

template <typename T>
inline T Mult(T *left, T *right) {
  return *left * *right;
}

template <typename T>
inline T Div(T *left, T *right) {
  return *left / *right;
}

} // namespace fasttensor::simd
