cmake_minimum_required(VERSION 3.14.6)

add_executable(bench bench.cpp)
target_link_libraries(bench fasttensor rt)
target_compile_options(bench PRIVATE -O3)
target_compile_options(bench PRIVATE -DNDEBUG)
target_compile_options(bench PRIVATE -march=native)
