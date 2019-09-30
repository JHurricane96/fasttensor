cmake_minimum_required(VERSION 3.14.6)

SET(SOURCE_FILES
	cwiseops.cpp
	test.cpp)

add_executable(tests ${SOURCE_FILES})
target_link_libraries(tests fasttensor gtest)

target_compile_features(tests PRIVATE cxx_std_17)

target_compile_options(tests PRIVATE -march=native)

install(TARGETS tests DESTINATION bin)