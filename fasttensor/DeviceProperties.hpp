#pragma once

namespace fasttensor {

struct DeviceProperties {
  DeviceProperties(int block_size, int max_blocks)
      : block_size(block_size), max_blocks(max_blocks) {}
  inline int blockSize() const { return block_size; }

  inline int maxBlocks() const { return max_blocks; }

private:
  int block_size;
  int max_blocks;
};

} // namespace fasttensor
