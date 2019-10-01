#pragma once

namespace fasttensor {

template <typename T, typename = void>
struct ref_selector {
  using type = T;
};

template <typename T>
using ref_selector_t = typename ref_selector<T>::type;

} // namespace fasttensor
