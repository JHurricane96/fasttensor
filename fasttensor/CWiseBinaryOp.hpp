#pragma once

#include "GpuDeviceFunction.hpp"
#include "RefSelector.hpp"
#include "Simd/Simd.hpp"
#include "Tensor.hpp"
#include "TensorExpression.hpp"

namespace fasttensor {

enum class BinaryOp { Plus, Minus, Multiplies, Divides };

template <typename LeftExpr, typename RightExpr, BinaryOp Op>
class CWiseBinaryOp;

template <typename LeftExpr, typename RightExpr, BinaryOp Op>
class CWiseBinaryOp : public TensorExpression {
  static_assert(are_tensor_exprs<LeftExpr, RightExpr>,
                "Expressions in CWiseBinaryOp must inherit from TensorExpression");

public:
  using LeftExprRefType = ref_selector_t<LeftExpr>;
  using RightExprRefType = ref_selector_t<RightExpr>;

  CWiseBinaryOp(const LeftExpr &left_expr, const RightExpr &right_expr)
      : _left_expr(left_expr), _right_expr(right_expr) {}

  inline auto getPacket(std::ptrdiff_t n) const {
    if constexpr (Op == BinaryOp::Plus) {
      return simd::Add(_left_expr.getPacket(n), _right_expr.getPacket(n));
    } else if constexpr (Op == BinaryOp::Minus) {
      return simd::Sub(_left_expr.getPacket(n), _right_expr.getPacket(n));
    } else if constexpr (Op == BinaryOp::Multiplies) {
      return simd::Mult(_left_expr.getPacket(n), _right_expr.getPacket(n));
    } else if constexpr (Op == BinaryOp::Divides) {
      return simd::Div(_left_expr.getPacket(n), _right_expr.getPacket(n));
    }
  }

  GPU_DEVICE_FUNC inline auto getCoeff(std::ptrdiff_t n) const {
    if constexpr (Op == BinaryOp::Plus) {
      return _left_expr.getCoeff(n) + _right_expr.getCoeff(n);
    } else if constexpr (Op == BinaryOp::Minus) {
      return _left_expr.getCoeff(n) - _right_expr.getCoeff(n);
    } else if constexpr (Op == BinaryOp::Multiplies) {
      return _left_expr.getCoeff(n) * _right_expr.getCoeff(n);
    } else if constexpr (Op == BinaryOp::Divides) {
      return _left_expr.getCoeff(n) / _right_expr.getCoeff(n);
    }
  }

  inline LeftExprRefType leftExpr() const { return _left_expr; }

  inline RightExprRefType rightExpr() const { return _right_expr; }

private:
  LeftExprRefType _left_expr;
  RightExprRefType _right_expr;
};

template <typename LeftExpr, typename RightExpr,
          typename = enable_if_tensor_exprs<LeftExpr, RightExpr>>
inline auto operator+(LeftExpr const &left_expr, RightExpr const &right_expr) {
  return CWiseBinaryOp<const LeftExpr, const RightExpr, BinaryOp::Plus>(left_expr, right_expr);
}

template <typename LeftExpr, typename RightExpr,
          typename = enable_if_tensor_exprs<LeftExpr, RightExpr>>
inline auto operator-(LeftExpr const &left_expr, RightExpr const &right_expr) {
  return CWiseBinaryOp<const LeftExpr, const RightExpr, BinaryOp::Minus>(left_expr, right_expr);
}

template <typename LeftExpr, typename RightExpr,
          typename = enable_if_tensor_exprs<LeftExpr, RightExpr>>
inline auto operator*(LeftExpr const &left_expr, RightExpr const &right_expr) {
  return CWiseBinaryOp<const LeftExpr, const RightExpr, BinaryOp::Multiplies>(left_expr,
                                                                              right_expr);
}

template <typename LeftExpr, typename RightExpr,
          typename = enable_if_tensor_exprs<LeftExpr, RightExpr>>
inline auto operator/(LeftExpr const &left_expr, RightExpr const &right_expr) {
  return CWiseBinaryOp<const LeftExpr, const RightExpr, BinaryOp::Divides>(left_expr, right_expr);
}

} // namespace fasttensor
