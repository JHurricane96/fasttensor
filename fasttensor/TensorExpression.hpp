#pragma once

#include "TensorExpression.fwd.hpp"
#include "Transforms/GetPacket.hpp"
#include "boost/yap/algorithm.hpp"

namespace yap = boost::yap;

namespace fasttensor {

template <boost::yap::expr_kind ExpressionKind, typename Tuple>
class TensorExpression {
public:
  static const boost::yap::expr_kind kind = ExpressionKind;
  Tuple elements;

  auto getPacket(std::ptrdiff_t index) { return yap::transform(*this, GetPacket{index}); }
};

BOOST_YAP_USER_BINARY_OPERATOR(plus, TensorExpression, TensorExpression);
BOOST_YAP_USER_BINARY_OPERATOR(minus, TensorExpression, TensorExpression);
BOOST_YAP_USER_BINARY_OPERATOR(multiplies, TensorExpression, TensorExpression);
BOOST_YAP_USER_BINARY_OPERATOR(divides, TensorExpression, TensorExpression);

} // namespace fasttensor
