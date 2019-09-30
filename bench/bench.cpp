#include "Assign.hpp"
#include "CWiseBinaryOp.hpp"
#include "Tensor.hpp"
#include "timer.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

using namespace std;
using namespace fasttensor;

const int TRIES = 2;
const int REPEAT = 10;

template <typename ElementType, int Rank>
Tensor<ElementType, Rank> make_rand_tensor(array<ptrdiff_t, Rank> dimensions) {
  static_assert(is_floating_point<ElementType>::value);
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<ElementType> dis(-100.0, 100.0);

  Tensor<ElementType, Rank> t(dimensions);
  auto num_elts = t.num_elements();
  auto &_storage = t.storage();
  for (int i = 0; i < num_elts; ++i) {
    _storage.storeCoeff(dis(gen), i);
  }
  return t;
}

double square(const double num) {
  return num * num;
}

double timer_mean(const BenchTimer &timer) {
  return timer.total() / TRIES;
}

double timer_sd(const BenchTimer &timer) {
  double mean_squares = timer.squared_total() / TRIES;
  double squared_mean = square(timer_mean(timer));
  return sqrt(mean_squares - squared_mean);
}

void print_results(string test_name, const BenchTimer &timer, double flops_factor) {
  auto mean = timer_mean(timer);
  auto sd = timer_sd(timer);
  auto flops = flops_factor * REPEAT * TRIES / (pow(1024., 3) * timer.total());
  std::cout << test_name << " :: Mean: " << mean << "s, SD: " << sd << "s; " << flops << " GFlops\n";
}

int main() {
  ptrdiff_t row = 1E4, col = 1E3, dep = 10;
  array<ptrdiff_t, 3> dimensions{row, col, dep};
  auto a = make_rand_tensor<float, 3>(dimensions);
  auto b = make_rand_tensor<float, 3>(dimensions);
  auto c = make_rand_tensor<float, 3>(dimensions);
  auto d = make_rand_tensor<float, 3>(dimensions);
  Tensor<float, 3> result(dimensions);

  BenchTimer timer;

  BENCH(timer, TRIES, REPEAT, result = a + b + c + d);
  print_results("a+b+c+d (Lazy)", timer, double(row * col * dep * 3));
  std::cout << result.getCoeff(0) << "\n";

  return 0;
}
