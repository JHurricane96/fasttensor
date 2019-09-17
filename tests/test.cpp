#include "Simd/Simd.hpp"
#include "Tensor.hpp"
#include "TensorExpression.hpp"
#include "boost/yap/print.hpp"
#include "gtest/gtest.h"
#include <array>
#include <iostream>

using namespace boost;
using namespace std;
using namespace fasttensor;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  cout << simd::PacketTraits<int>::size << endl;
  cout << simd::PacketTraits<char>::size << endl;

  cout << "Device Type: " << static_cast<int>(device_type) << endl;

  Tensor<int, 2> t(array<ptrdiff_t, 2>{4, 2});
  Tensor<int, 2> q(array<ptrdiff_t, 2>{4, 2});
  auto result = q + t;
  yap::print(cout, result);
  return RUN_ALL_TESTS();
}