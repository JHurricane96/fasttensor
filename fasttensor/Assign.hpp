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
__global__ void Kernel(ElementType *lhs_storage, OtherExpr rhs, int start_offset, int end_offset) {
  int index = blockIdx.x * blockDim.x + threadIdx.x + start_offset;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < end_offset; i += stride) {
    lhs_storage[i] = rhs.getCoeff(i);
  }
}

template <typename ElementType, int Rank, typename OtherExpr,
          typename = enable_if_tensor_exprs<OtherExpr>>
inline void Assign(Tensor<ElementType, Rank, GpuDevice> &lhs, OtherExpr const &rhs) {
  auto unwrapped_rhs = UnwrapStorage(rhs);

  auto &device_props = lhs.device().deviceProps();
  auto num_devices = device_props.size();
  auto num_elements = lhs.num_elements();
  auto num_elts_per_device_floored = num_elements / num_devices;
  auto partition_point = num_devices - (num_elements % num_devices);

  auto num_calculted_elts = 0;
  for (int i = 0; i < num_devices; ++i) {
    cudaSetDevice(i);

    auto &device = device_props[i];
    auto block_size = device.blockSize();
    auto max_blocks = device.maxBlocks();
    decltype(num_elements) num_elts_current_device = 0;

    if (i >= partition_point) {
      num_elts_current_device = num_elts_per_device_floored + 1;
    } else {
      num_elts_current_device = num_elts_per_device_floored;
    }

    int num_blocks = std::min((decltype(num_elements))max_blocks,
                              (num_elts_current_device + block_size - 1) / block_size);
    auto end_offset = num_calculted_elts + num_elts_current_device;

    Kernel<<<num_blocks, block_size>>>(lhs.storage().elements(), unwrapped_rhs, num_calculted_elts,
                                       end_offset);

    num_calculted_elts = end_offset;
  }

  for (int i = 0; i < device_props.size(); ++i) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
}

#endif

template <typename ElementType, int Rank, typename DeviceType>
template <typename OtherExpr, typename>
inline Tensor<ElementType, Rank, DeviceType> &
Tensor<ElementType, Rank, DeviceType>::operator=(OtherExpr const &other) {
  Assign(*this, other);
  return *this;
}

} // namespace fasttensor
