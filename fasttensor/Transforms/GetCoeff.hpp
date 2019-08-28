#pragma once

namespace fasttensor {

struct GetCoeff {
  std::ptrdiff_t index;

  template <typename TerminalExpression>
  auto operator()(yap::expr_tag<yap::expr_kind::terminal>, TerminalExpression terminal) {
    return terminal.getCoeff(index);
  }
};

} // namespace fasttensor
