#pragma once

namespace fasttensor::utils {

template <std::size_t iter, typename Scalar, std::size_t N, typename Op>
inline Scalar fold(const std::array<Scalar, N> &list, const Op &op) {
  if constexpr (iter == 0) {
    return list[0];
  } else {
    return op(list[iter], fold<iter - 1>(list, op));
  }
}

template <std::size_t iter, std::size_t N>
inline std::ptrdiff_t getIndex(const std::array<std::ptrdiff_t, N> &dimensions,
                               const std::array<std::ptrdiff_t, N> &indices) {
  if constexpr (iter == 0) {
    return indices[0];
  } else {
    return indices[iter] + dimensions[iter] * getIndex<iter - 1>(dimensions, indices);
  }
}

} // namespace fasttensor::utils
