# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MeshMatching)

# Set the C++ standard to 17 for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Point Cloud Library (PCL) package and its components
# PCL is used for loading, processing, and visualizing the meshes.
find_package(PCL 1.10 REQUIRED COMPONENTS common io features keypoints registration visualization)

# Include the PCL directories so the compiler can find the header files
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Fetch and configure yaml-cpp
include(FetchContent)
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.7.0 # Use a specific tag for stability
)
FetchContent_MakeAvailable(yaml-cpp)

# Add the executable to be built from our source file
add_executable(mesh_matching mesh_matching.cpp)

# Link the executable against the required PCL libraries
# This ensures our program has access to PCL's functions at runtime.
target_link_libraries(mesh_matching ${PCL_LIBRARIES} yaml-cpp)
