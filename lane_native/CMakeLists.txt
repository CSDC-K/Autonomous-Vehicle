cmake_minimum_required(VERSION 3.14)
project(lane_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Release optimizations for Raspberry Pi ───────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# AArch64 (Raspberry Pi 4/5) specific tuning
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a72 -ftree-vectorize")
endif()

# ── Dependencies ─────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(pybind11 REQUIRED)

# ── Module ───────────────────────────────────────────────────────
pybind11_add_module(lane_native lane_native.cpp)
target_link_libraries(lane_native PRIVATE ${OpenCV_LIBS})
target_include_directories(lane_native PRIVATE ${OpenCV_INCLUDE_DIRS})

# install into site-packages (pip install handles this)
install(TARGETS lane_native LIBRARY DESTINATION .)
