#include "Tensor.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace fasttensor;

template <typename ElementType, int Rank>
Tensor<ElementType, Rank> CreateTensor(array<ptrdiff_t, Rank> dimensions) {
  Tensor<ElementType, Rank> t(dimensions);
  auto num_elts = t.num_elements();
  auto &_storage = t.storage();
  for (int i = 0; i < num_elts; ++i) {
    _storage.getCoeff(i) = i;
  }
  return t;
}

TEST(CWiseOps, Add) {
  int n_i = 10, n_j = 11, n_k = 12;
  array<ptrdiff_t, 3> dimensions{n_i, n_j, n_k};
  auto a = CreateTensor<int, 3>(dimensions);
  auto b = CreateTensor<int, 3>(dimensions);
  Tensor<int, 3> result(dimensions);
  result = a + b;
  for (int i = 0; i < n_i; ++i) {
    for (int j = 0; j < n_j; ++j) {
      for (int k = 0; k < n_k; ++k) {
        EXPECT_EQ(result(i, j, k), 2 * (k + n_k * (j + (n_j * i))));
      }
    }
  }
}