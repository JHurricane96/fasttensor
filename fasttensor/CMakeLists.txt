cmake_minimum_required(VERSION 3.14.6)

add_library(fasttensor INTERFACE)

target_include_directories(fasttensor INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	$<INSTALL_INTERFACE:.>
)

install(TARGETS fasttensor EXPORT fasttensor_config
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
)

install(EXPORT fasttensor_config DESTINATION lib)
install(DIRECTORY ./ DESTINATION fasttensor)
