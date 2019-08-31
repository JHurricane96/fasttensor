#pragma once

#include "TensorExpression.fwd.hpp"
#include "TensorStorage.hpp"
#include "boost/yap/algorithm.hpp"

namespace yap = boost::yap;

namespace fasttensor {

struct GetPacket {
  std::ptrdiff_t n;

  template <typename Operand1, typename Operand2>
  auto operator()(yap::expr_tag<yap::expr_kind::plus>, Operand1 left, Operand2 right) {
    return simd::Add(left.getPacket(n), right.getPacket(n));
  }

  template <typename Operand1, typename Operand2>
  auto operator()(yap::expr_tag<yap::expr_kind::minus>, Operand1 left, Operand2 right) {
    return simd::Sub(left.getPacket(n), right.getPacket(n));
  }

  template <typename Operand1, typename Operand2>
  auto operator()(yap::expr_tag<yap::expr_kind::multiplies>, Operand1 left, Operand2 right) {
    return simd::Mult(left.getPacket(n), right.getPacket(n));
  }

  template <typename Operand1, typename Operand2>
  auto operator()(yap::expr_tag<yap::expr_kind::divides>, Operand1 left, Operand2 right) {
    return simd::Div(left.getPacket(n), right.getPacket(n));
  }

  template <typename TerminalExpression>
  auto operator()(yap::expr_tag<yap::expr_kind::terminal>, TerminalExpression terminal) {
    return terminal.getPacket(n);
  }
};

} // namespace fasttensor
