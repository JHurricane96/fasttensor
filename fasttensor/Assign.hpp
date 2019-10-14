#pragma once

#include "Device.hpp"
#include "StorageUnwrapper.hpp"
#include "Tensor.hpp"

#if defined FASTTENSOR_GPU
#  include "GpuDevice.hpp"
#endif

namespace fasttensor {

#if defined FASTTENSOR_SIMD || defined FASTTENSOR_NORMAL

template <typename ElementType, int Rank, typename OtherExpr,
          typename = enable_if_tensor_exprs<OtherExpr>>
inline void Assign(Tensor<ElementType, Rank, DefaultDevice> &lhs, OtherExpr const &rhs) {
  auto &storage = lhs.storage();
  if constexpr (device_type == DeviceType::Simd &&
                simd::PacketTraits<ElementType>::is_vectorizable) {
    auto packet_size = simd::PacketTraits<ElementType>::size;
    auto num_packets = storage.num_elements() / packet_size;
    for (std::ptrdiff_t i = 0; i < num_packets; ++i) {
      storage.storePacket(i, rhs.getPacket(i));
    }
    for (std::ptrdiff_t i = num_packets * packet_size; i < storage.num_elements(); ++i) {
      storage.storeCoeff(rhs.getCoeff(i), i);
    }
  } else {
    for (std::ptrdiff_t i = 0; i < storage.num_elements(); ++i) {
      storage.storeCoeff(rhs.getCoeff(i), i);
    }
  }
}

#elif defined FASTTENSOR_GPU

template <typename ElementType, typename OtherExpr, typename = enable_if_tensor_exprs<OtherExpr>>
__global__ void Kernel(ElementType *lhs_storage, OtherExpr rhs, int num_elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < num_elements; i += stride) {
    lhs_storage[i] = rhs.getCoeff(i);
  }
}

template <typename ElementType, int Rank, typename OtherExpr,
          typename = enable_if_tensor_exprs<OtherExpr>>
inline void Assign(Tensor<ElementType, Rank, GpuDevice> &lhs, OtherExpr const &rhs) {
  auto unwrapped_rhs = UnwrapStorage(rhs);

  auto &device_props = lhs.device().deviceProps()[0];
  auto block_size = device_props.blockSize();
  auto max_blocks = device_props.maxBlocks();

  auto num_elements = lhs.num_elements();
  int num_blocks = std::min((long)max_blocks, (num_elements + block_size - 1) / block_size);

  Kernel<<<num_blocks, block_size>>>(lhs.storage().elements(), unwrapped_rhs, num_elements);
  cudaDeviceSynchronize();
}

#endif

template <typename ElementType, int Rank, typename DeviceType>
template <typename OtherExpr, typename>
inline Tensor<ElementType, Rank, DeviceType> &Tensor<ElementType, Rank, DeviceType>::
operator=(OtherExpr const &other) {
  Assign(*this, other);
  return *this;
}

} // namespace fasttensor
