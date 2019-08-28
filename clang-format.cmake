# Creates additional target to perform clang-format run, requires clang-format

# Find clang format
find_program(CLANG_FORMAT_EXECUTABLE "clang-format")
if(NOT CLANG_FORMAT_EXECUTABLE)
	return()
endif()

# Get all project files
file(GLOB_RECURSE ALL_SOURCE_FILES fasttensor/*.hpp tests/*.cpp)

# Add target to build
add_custom_target(
	clangformat
	COMMAND ${CLANG_FORMAT_EXECUTABLE}
	-style=file
	-i
	${ALL_SOURCE_FILES}
)