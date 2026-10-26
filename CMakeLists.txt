cmake_minimum_required(VERSION 3.26)
project(a11_optimizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(a11_optimizer_lib STATIC src/memory/nvme_loader.cpp)
target_include_directories(a11_optimizer_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(WIN32)
    target_compile_definitions(a11_optimizer_lib PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

add_executable(a11_tests tests/test_nvme_loader.cpp)
target_link_libraries(a11_tests PRIVATE a11_optimizer_lib)

if(WIN32)
    target_compile_definitions(a11_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

enable_testing()
add_test(NAME a11_tests COMMAND a11_tests)
