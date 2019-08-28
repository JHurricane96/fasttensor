cmake_minimum_required(VERSION 3.14.6)

find_package(Boost 1.70.0 EXACT REQUIRED)

add_library(fasttensor INTERFACE)

target_include_directories(fasttensor INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
	$<INSTALL_INTERFACE:.>
)

install(TARGETS fasttensor EXPORT fasttensor_config
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
)

install(EXPORT fasttensor_config DESTINATION lib)
install(DIRECTORY ./ DESTINATION fasttensor)
