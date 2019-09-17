#include "Tensor.hpp"
#include "gtest/gtest.h"
#include <array>
#include <cstddef>

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

TEST(CWiseOps, AddInt) {
  int n_i = 10, n_j = 11, n_k = 13;
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

TEST(CWiseOps, SubInt) {
  int n_i = 10, n_j = 11, n_k = 13;
  array<ptrdiff_t, 3> dimensions{n_i, n_j, n_k};
  auto a = CreateTensor<int, 3>(dimensions);
  auto b = CreateTensor<int, 3>(dimensions);
  auto result = CreateTensor<int, 3>(dimensions);
  result = a - b;
  for (int i = 0; i < n_i; ++i) {
    for (int j = 0; j < n_j; ++j) {
      for (int k = 0; k < n_k; ++k) {
        EXPECT_EQ(result(i, j, k), 0);
      }
    }
  }
}

TEST(CWiseOps, MultInt) {
  int n_i = 10, n_j = 11, n_k = 13;
  array<ptrdiff_t, 3> dimensions{n_i, n_j, n_k};
  auto a = CreateTensor<int, 3>(dimensions);
  auto b = CreateTensor<int, 3>(dimensions);
  Tensor<int, 3> result(dimensions);
  result = a * b;
  for (int i = 0; i < n_i; ++i) {
    for (int j = 0; j < n_j; ++j) {
      for (int k = 0; k < n_k; ++k) {
        auto elt = (k + n_k * (j + (n_j * i)));
        EXPECT_EQ(result(i, j, k), elt * elt);
      }
    }
  }
}

TEST(CWiseOps, AddFloat) {
  int n_i = 10, n_j = 11, n_k = 13;
  array<ptrdiff_t, 3> dimensions{n_i, n_j, n_k};
  auto a = CreateTensor<float, 3>(dimensions);
  auto b = CreateTensor<float, 3>(dimensions);
  Tensor<float, 3> result(dimensions);
  result = a + b;
  for (int i = 0; i < n_i; ++i) {
    for (int j = 0; j < n_j; ++j) {
      for (int k = 0; k < n_k; ++k) {
        EXPECT_FLOAT_EQ(result(i, j, k), 2 * (k + n_k * (j + (n_j * i))));
      }
    }
  }
}

TEST(CWiseOps, SubFloat) {
  int n_i = 10, n_j = 11, n_k = 13;
  array<ptrdiff_t, 3> dimensions{n_i, n_j, n_k};
  auto a = CreateTensor<float, 3>(dimensions);
  auto b = CreateTensor<float, 3>(dimensions);
  auto result = CreateTensor<float, 3>(dimensions);
  result = a - b;
  for (int i = 0; i < n_i; ++i) {
    for (int j = 0; j < n_j; ++j) {
      for (int k = 0; k < n_k; ++k) {
        EXPECT_FLOAT_EQ(result(i, j, k), 0);
      }
    }
  }
}

TEST(CWiseOps, MultFloat) {
  int n_i = 10, n_j = 11, n_k = 13;
  array<ptrdiff_t, 3> dimensions{n_i, n_j, n_k};
  auto a = CreateTensor<float, 3>(dimensions);
  auto b = CreateTensor<float, 3>(dimensions);
  Tensor<float, 3> result(dimensions);
  result = a * b;
  for (int i = 0; i < n_i; ++i) {
    for (int j = 0; j < n_j; ++j) {
      for (int k = 0; k < n_k; ++k) {
        auto elt = (k + n_k * (j + (n_j * i)));
        EXPECT_FLOAT_EQ(result(i, j, k), elt * elt);
      }
    }
  }
}

TEST(CWiseOps, DivFloat) {
  int n_i = 10, n_j = 11, n_k = 13;
  array<ptrdiff_t, 3> dimensions{n_i, n_j, n_k};
  auto a = CreateTensor<float, 3>(dimensions);
  auto b = CreateTensor<float, 3>(dimensions);
  a(0, 0, 0) = 1;
  b(0, 0, 0) = 1;
  Tensor<float, 3> result(dimensions);
  result = a / b;
  for (int i = 0; i < n_i; ++i) {
    for (int j = 0; j < n_j; ++j) {
      for (int k = 0; k < n_k; ++k) {
        EXPECT_FLOAT_EQ(result(i, j, k), 1);
      }
    }
  }
}

TEST(CWiseOps, AddDouble) {
  int n_i = 10, n_j = 11, n_k = 13;
  array<ptrdiff_t, 3> dimensions{n_i, n_j, n_k};
  auto a = CreateTensor<double, 3>(dimensions);
  auto b = CreateTensor<double, 3>(dimensions);
  Tensor<double, 3> result(dimensions);
  result = a + b;
  for (int i = 0; i < n_i; ++i) {
    for (int j = 0; j < n_j; ++j) {
      for (int k = 0; k < n_k; ++k) {
        EXPECT_DOUBLE_EQ(result(i, j, k), 2 * (k + n_k * (j + (n_j * i))));
      }
    }
  }
}

TEST(CWiseOps, SubDouble) {
  int n_i = 10, n_j = 11, n_k = 13;
  array<ptrdiff_t, 3> dimensions{n_i, n_j, n_k};
  auto a = CreateTensor<double, 3>(dimensions);
  auto b = CreateTensor<double, 3>(dimensions);
  auto result = CreateTensor<double, 3>(dimensions);
  result = a - b;
  for (int i = 0; i < n_i; ++i) {
    for (int j = 0; j < n_j; ++j) {
      for (int k = 0; k < n_k; ++k) {
        EXPECT_DOUBLE_EQ(result(i, j, k), 0);
      }
    }
  }
}

TEST(CWiseOps, MultDouble) {
  int n_i = 10, n_j = 11, n_k = 13;
  array<ptrdiff_t, 3> dimensions{n_i, n_j, n_k};
  auto a = CreateTensor<double, 3>(dimensions);
  auto b = CreateTensor<double, 3>(dimensions);
  Tensor<double, 3> result(dimensions);
  result = a * b;
  for (int i = 0; i < n_i; ++i) {
    for (int j = 0; j < n_j; ++j) {
      for (int k = 0; k < n_k; ++k) {
        auto elt = (k + n_k * (j + (n_j * i)));
        EXPECT_DOUBLE_EQ(result(i, j, k), elt * elt);
      }
    }
  }
}

TEST(CWiseOps, DivDouble) {
  int n_i = 10, n_j = 11, n_k = 13;
  array<ptrdiff_t, 3> dimensions{n_i, n_j, n_k};
  auto a = CreateTensor<double, 3>(dimensions);
  auto b = CreateTensor<double, 3>(dimensions);
  a(0, 0, 0) = 1;
  b(0, 0, 0) = 1;
  Tensor<double, 3> result(dimensions);
  result = a / b;
  for (int i = 0; i < n_i; ++i) {
    for (int j = 0; j < n_j; ++j) {
      for (int k = 0; k < n_k; ++k) {
        EXPECT_DOUBLE_EQ(result(i, j, k), 1);
      }
    }
  }
}
