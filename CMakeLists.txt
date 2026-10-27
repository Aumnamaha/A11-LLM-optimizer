cmake_minimum_required(VERSION 3.26)
project(a11_optimizer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- pybind11 configuration (manual, from pip-installed package) ----
execute_process(
    COMMAND "C:\\Users\\Aum Namaha\\Documents\\finalenv\\Scripts\\python.exe" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get Python include path from finalenv's Python interpreter
execute_process(
    COMMAND "C:\\Users\\Aum Namaha\\Documents\\finalenv\\Scripts\\python.exe" -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Fallback: if INCLUDEPY is empty, use the system Python include directory
if(NOT PYTHON_INCLUDE_DIR OR PYTHON_INCLUDE_DIR STREQUAL "")
    set(PYTHON_INCLUDE_DIR "C:/Users/Aum Namaha/AppData/Local/Python/pythoncore-3.14-64/Include")
endif()
message(STATUS "Using Python include dir: ${PYTHON_INCLUDE_DIR}")

message(STATUS "pybind11 include: ${PYBIND11_INCLUDE_DIR}")
message(STATUS "Python include: ${PYTHON_INCLUDE_DIR}")

# ---- Library target: a11_optimizer_lib ----
add_library(a11_optimizer_lib STATIC
    src/memory/nvme_loader.cpp
    src/memory/hot_expert_cache.cpp
    src/cuda/paged_kv_cache.cu
    src/cuda/fp8_gemv.cu
    src/cuda/tree_verification.cu
    src/engine/speculative_verifier.cpp
)

target_include_directories(a11_optimizer_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(WIN32)
    target_compile_definitions(a11_optimizer_lib PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# RTX 5070 Ti = Blackwell CC 9.0 (CUDA 13.x supports it)
set_target_properties(a11_optimizer_lib PROPERTIES
    CUDA_ARCHITECTURES "90"
)

# ---- Test executable: a11_tests ----
add_executable(a11_tests
    tests/test_nvme_loader.cpp
    tests/test_paged_kv_cache.cpp
    tests/test_speculative_verifier.cpp
)

target_link_libraries(a11_tests PRIVATE a11_optimizer_lib)

if(WIN32)
    target_compile_definitions(a11_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ---- Testing enablement ----
enable_testing()
add_test(NAME a11_tests COMMAND a11_tests)

# ===================================================================
# Python extension module: a11_optimizer (pybind11 binding)
# ===================================================================
add_library(a11_optimizer MODULE src/bindings/python_bindings.cpp)

target_include_directories(a11_optimizer PRIVATE ${PYTHON_INCLUDE_DIR} ${PYBIND11_INCLUDE_DIR})

# Link against Python import library for Windows MSVC builds
if(WIN32)
    set(PYTHON_LIB_DIR "C:/Users/Aum Namaha/AppData/Local/Python/pythoncore-3.14-64/libs")
    target_link_libraries(a11_optimizer PRIVATE ${PYTHON_LIB_DIR}/python314.lib)
endif()
target_link_libraries(a11_optimizer PRIVATE a11_optimizer_lib)

if(WIN32)
    target_compile_definitions(a11_optimizer PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# On Windows, pybind11 modules need .pyd extension (not .dll)
set_target_properties(a11_optimizer PROPERTIES
    SUFFIX ".pyd"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/a11_optimizer"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/a11_optimizer"
    # Force flat output (no Release/ subdirectory)
    CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/python/a11_optimizer"
    CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/python/a11_optimizer"
)

# Copy __init__.py alongside the extension module
file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/python/a11_optimizer/__init__.py"
"# a11_optimizer Python package\nfrom .a11_optimizer import *\n__all__ = []\n"
)

# Post-build: copy .pyd from Release/ subdirectory to flat package directory (MSVC VS generator quirk)
add_custom_command(TARGET a11_optimizer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/python/a11_optimizer"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:a11_optimizer>"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/a11_optimizer/$<TARGET_FILE_NAME:a11_optimizer>"
    COMMENT "Copying a11_optimizer.pyd to package directory"
)
