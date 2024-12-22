cmake_minimum_required(VERSION 3.10)
project(TestCMake LANGUAGES C CXX) # Enable both C and C++

# Set C and C++ standards
set(CMAKE_C_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)

# Include gtest headers and src libraries
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    /usr/local/include
)

# Link path to libraries
link_directories(/usr/local/lib)

# Recursively collect all .c source files in src directory
file(GLOB_RECURSE SOURCE_FILES src/*.c) 

# Recursively collect all .cpp test files in tests directory 
file(GLOB_RECURSE TEST_FILES tests/*.cpp)

# Create an executable for each test file
foreach(TEST_FILE ${TEST_FILES})
    # Get the test name from the filename
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)

    # Create the test executable linked with all source files
    add_executable(${TEST_NAME} ${TEST_FILE} ${SOURCE_FILES})
    target_link_libraries(${TEST_NAME} PRIVATE gtest gtest_main pthread)

    # Set executables to be within the build directory
    set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endforeach()