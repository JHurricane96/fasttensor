#pragma once

#include <TensorExpression.hpp>

namespace fasttensor {

template <yap::expr_kind ExpressionKind, typename Tuple>
class CWiseBinaryOp : public TensorExpression<ExpressionKind, Tuple> {
public:
  auto getPacket(std::ptrdiff_t index) { return; }
};

} // namespace fasttensor
