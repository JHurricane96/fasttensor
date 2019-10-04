#pragma once

#include "CWiseBinaryOp.hpp"
#include "Tensor.hpp"
#include "TensorStorageRef.hpp"

namespace fasttensor {

template <typename LeftExpr, typename RightExpr, BinaryOp Op>
auto inline UnwrapStorage(CWiseBinaryOp<LeftExpr, RightExpr, Op> const &cwise_binary_op) {
  auto left_expr = UnwrapStorage(cwise_binary_op.leftExpr());
  auto right_expr = UnwrapStorage(cwise_binary_op.rightExpr());
  return CWiseBinaryOp<const typeof(left_expr), const typeof(right_expr), Op>(left_expr,
                                                                              right_expr);
}

template <typename ElementType, int Rank>
auto inline UnwrapStorage(Tensor<ElementType, Rank> const &tensor) {
  return TensorStorageRef(tensor.storage().elements());
}

} // namespace fasttensor
