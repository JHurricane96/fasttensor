#pragma once

#include "Simd/Simd.hpp"
#include <new>

#if defined FASTTENSOR_GPU
#  include <cuda_runtime.h>
#endif

namespace fasttensor {

#if defined FASTTENSOR_SIMD || defined FASTTENSOR_NORMAL

template <typename ElementType>
ElementType *AllocateMemory(std::ptrdiff_t num_elements) {
  if constexpr (simd::PacketTraits<ElementType>::is_vectorizable) {
    return reinterpret_cast<ElementType *>(operator new[](sizeof(ElementType) * num_elements,
                                                          std::align_val_t(simd::PacketSize)));
  } else {
    return new ElementType[num_elements];
  }
}

template <typename ElementType>
void DeallocateMemory(ElementType *memory) {
  if constexpr (simd::PacketTraits<ElementType>::is_vectorizable) {
    operator delete[](memory, std::align_val_t(simd::PacketSize));
  } else {
    delete[] memory;
  }
}

#elif defined FASTTENSOR_GPU

template <typename ElementType>
ElementType *AllocateMemory(std::ptrdiff_t num_elements) {
  ElementType *memory;
  cudaMallocManaged(&memory, num_elements * sizeof(ElementType));
  return memory;
}

template <typename ElementType>
void DeallocateMemory(ElementType *memory) {
  cudaFree(memory);
}

#endif

} // namespace fasttensor
