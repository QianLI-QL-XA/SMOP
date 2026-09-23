#======================================================================
# CMake build for the smop package (C++ core + tests)
#
#   cmake -S . -B build
#   cmake --build build --config Release
#   ./build/test_cpp
#======================================================================
cmake_minimum_required(VERSION 3.16)
project(smop LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- header-only interface library -------------------------------------
add_library(smop INTERFACE)
target_include_directories(smop INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# locate Eigen (bundled under third_party/ if present)
find_path(EIGEN3_INCLUDE_DIR Eigen/Core
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen-3.4.0
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen headers not found. Download Eigen 3.4.x into smop/third_party/.")
endif()
target_include_directories(smop INTERFACE ${EIGEN3_INCLUDE_DIR})

# ---- tests --------------------------------------------------------------
enable_testing()
add_executable(test_cpp tests/test_cpp.cpp)
target_link_libraries(test_cpp PRIVATE smop)
add_test(NAME test_cpp COMMAND test_cpp)

# ---- Python bindings (optional; preferred path: pip install ./python) ---
option(BMOP_BUILD_PYTHON "Build the Python module via pybind11" OFF)
if(BMOP_BUILD_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        pybind11_add_module(_core python/src/smop_py.cpp)
        target_include_directories(_core PRIVATE include ${EIGEN3_INCLUDE_DIR})
        target_link_libraries(_core PRIVATE smop)
    else()
        message(WARNING "pybind11 not found; skipping the Python module (use pip install ./python)")
    endif()
endif()
