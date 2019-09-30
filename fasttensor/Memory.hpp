#pragma once

#include "Simd/Simd.hpp"
#include <cstdlib>

namespace fasttensor {

#if defined FASTTENSOR_SIMD || defined FASTTENSOR_NORMAL

template <typename ElementType>
inline ElementType *AllocateMemory(std::ptrdiff_t num_elements) {
  if constexpr (simd::PacketTraits<ElementType>::is_vectorizable) {
    return reinterpret_cast<ElementType *>(
        std::aligned_alloc(simd::PacketSize, sizeof(ElementType) * num_elements));
  } else {
    return new ElementType[num_elements];
  }
}

template <typename ElementType>
inline void DeallocateMemory(ElementType *memory) {
  if constexpr (simd::PacketTraits<ElementType>::is_vectorizable) {
    std::free(memory);
  } else {
    delete[] memory;
  }
}

#elif defined FASTTENSOR_GPU

template <typename ElementType>
inline ElementType *AllocateMemory(std::ptrdiff_t num_elements) {
  ElementType *memory;
  cudaMallocManaged(&memory, num_elements * sizeof(ElementType));
  return memory;
}

template <typename ElementType>
inline void DeallocateMemory(ElementType *memory) {
  cudaFree(memory);
}

#endif

} // namespace fasttensor
