# fasttensor

C++ library for tensor arithmetic.

Uses SIMD for CPU acceleration and CUDA for GPU acceleration. Supports multi-GPU if more than 1 is available. [Kernel fusion](https://stackoverflow.com/a/53311373) with [expression templates](https://en.wikipedia.org/wiki/Expression_templates) allows efficient computation of long arithmetic expressions.

## Usage

fasttensor is header-only, simply add the location of the header files to your include path while compiling.

Example code:

```cpp
using namespace fasttensor;

int main() {
	int num_rows = 4;
	int num_cols = 2;
	// Create integer tensor of rank 2
	// Dimensions: 4 rows, 2 columns (4x2)
	Tensor<int, 2> a(array<ptrdiff_t, 2>{num_rows, num_cols});
	Tensor<int, 2> b(array<ptrdiff_t, 2>{num_rows, num_cols});

	for (int i = 0; i < num_rows; ++i) {
		for (int j = 0; j < num_cols; ++j) {
			// This is how you set/get elements
			a(i, j) = j + num_cols * i;
			b(i, j) = j + num_cols * i;
		}
	}

	Tensor<int, 2> results(array<ptrdiff_t, 2>{num_rows, num_cols});

	// Element-wise addition of the two tensors
	// This will auto-magically use GPU/SIMD instructions
	// Need to compile with appropriate compiler flags and hardware
	results = a + b;

	for (int i = 0; i < num_rows; ++i) {
		for (int j = 0; j < num_cols; ++j) {
			// Just checking if we got the right answer
			assert(results(i, j) == 2 * (j + num_cols * i));
		}
	}

	return 0;
}
```

## Benchmarks

**Eager mode** is equivalent to a naive implementation of arithmetic expressions, creating a temporary variable after each operation. This behaviour was simulated with a helper function that forces eager evaluation of a given arithmetic expression.

**Lazy mode** constructs an expression at compile time using expression templates and only evaluates the expression when assigned to a matrix.

### Config:

**CPU**: Intel Xeon E5-2690 v3 @ 2.60 GHz  
**GPU**: NVidia Tesla P4  
**Compiler**: Clang 9.0.1  
**CUDA Toolkit Version**: 10.0  

### Results:

- The variables are 3-dimensional float tensors of size 10<sup>4</sup> &times; 10<sup>2</sup> &times; 10<sup>2</sup> filled with random values.
- The results were obtained by running 10 trials.
- Each trial consisted of evaluating the expression 100 times.


#### X = A + B + C + D

<table>
  <tr>
    <th rowspan="2">Devices</th>
    <th colspan="2">Eager</th>
    <th colspan="2">Lazy</th>
  </tr>
  <tr>
    <th>Time</th>
    <th>GFlops</th>
    <th>Time</th>
    <th>GFlops</th>
  </tr>
  <tr>
    <td>AVX2 on CPU</td>
    <td>28.26 &plusmn; 0.21s</td>
    <td>0.99</td>
    <td>17.73 &plusmn; 0.05s</td>
    <td>1.58</td>
  </tr>
  <tr>
    <td>1 Tesla P4 GPU</td>
    <td>2.65 &plusmn; 0.00s</td>
    <td>10.56</td>
    <td>1.51 &plusmn; 0.12s</td>
    <td>18.52</td>
  </tr>
  <tr>
    <td>2 Tesla P4 GPUs</td>
    <td>1.56 &plusmn; 0.20s</td>
    <td>17.92</td>
    <td>0.89 &plusmn; 0.08s</td>
    <td>31.25</td>
  </tr>
</table>

## Development

To run the tests and benchmarks on Linux:

(Dependencies: CMake >= 3.14.6, clang++ >= 8, CUDA >= 9)

1. Clone [this repo](https://github.com/JHurricane96/fasttensor)

2. `mkdir build && cd build`

3. Run CMake to generate build files (detailed instructions below). Add `-DBUILD_TESTS=OFF` to not build tests, `-DBUILD_BENCHMARKS=OFF` to not build benchmarks.

4. `cmake --build .`

5. `./tests` to run the tests and `./bench/bench` to run the benchmarks

### Running CMake

The build can be configured with various build options. The full command to run is:

```
CXX=<clang++ location> CC=<clang location> cmake.. \
-DDEVICE_TYPE=<NORMAL|SIMD|GPU> -DCMAKE_BUILD_TYPE=<Release|Debug> \
-DCUDA_PATH=<CUDA toolkit path> -DGPU_ARCH=<GPU arch>
```

- Use `CXX` and `CC` to set the C and C++ compiler to clang.
- Set `DEVICE_TYPE` to `NORMAL` for normal CPU mode, `SIMD` to use SIMD vectorized instructions, and `GPU` to use the GPU.
- Set `CMAKE_BUILD_TYPE` to `Release` or `Debug` depending on your need.
- Set `CUDA_PATH` to the location of the CUDA toolkit, and `GPU_ARCH` to the GPU's CUDA compute capability (3.7 means you should set it to 37, that is, simply remove the decimal). These options are only required if `DEVICE_TYPE` is `GPU`.
