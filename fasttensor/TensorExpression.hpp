#pragma once

#include "TensorExpression.fwd.hpp"
#include <type_traits>

namespace fasttensor {

class TensorExpression {};

template <typename T>
constexpr bool is_tensor_expr = std::is_base_of_v<TensorExpression, T>;

template <typename... T>
constexpr bool are_tensor_exprs = (... && is_tensor_expr<T>);

template <typename... T>
using enable_if_tensor_exprs = std::enable_if_t<are_tensor_exprs<T...>>;

} // namespace fasttensor
