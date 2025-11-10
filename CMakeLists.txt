cmake_minimum_required(VERSION 3.10)

# Set extension name here
set(TARGET_NAME mlpack)

# Set C++ standard (MLPack requires C++17 or later)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Consider using ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package. Feel free to remove or replace with other dependencies.
# Note that it should also be removed from vcpkg.json to prevent needlessly installing it..

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

# Find OpenMP (and link below)
find_package(OpenMP REQUIRED)

#find_package(Armadillo CONFIG REQUIRED)

# Find mlpack (and armadillo, ensmallen, cereal, ...)
include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/mlpack.cmake)
fetch_mlpack(COMPILE_OPENBLAS FALSE)

# We need to patch cereal, a mlpack dependency (cf https://github.com/USCiLab/cereal/pull/835)
file(READ ${CMAKE_BINARY_DIR}/deps/cereal-1.3.2/include/cereal/types/tuple.hpp CEREAL_TUPLE)
string(REPLACE
    "::template apply" "::apply"
    CEREAL_TUPLE "${CEREAL_TUPLE}")
file(WRITE ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-linux/include/cereal/types/tuple.hpp "${CEREAL_TUPLE}")
#message(STATUS "${CEREAL_TUPLE}")


set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)
include_directories(${MLPACK_INCLUDE_DIRS})

set(EXTENSION_SOURCES
    src/duckdb_utilities.cpp
    src/mlpack_adaboost.cpp
    src/mlpack_linear_regression.cpp
    src/mlpack_logistic_regression.cpp
    src/mlpack_random_forest.cpp
    src/mlpack_extension.cpp
    src/mlpack_utilities.cpp
)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Link OpenSSL in both the static library as the loadable extension
target_link_libraries(${EXTENSION_NAME} ${ARMADILLO_LIBRARIES} OpenMP::OpenMP_CXX)
target_link_libraries(${LOADABLE_EXTENSION_NAME} ${ARMADILLO_LIBRARIES} OpenMP::OpenMP_CXX)

#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")
#find_path(MLPACK_INCLUDE_DIRS "mlpack.hpp")
#target_include_directories(main PRIVATE ${MLPACK_INCLUDE_DIRS})
target_include_directories(${EXTENSION_NAME} PRIVATE ${MLPACK_INCLUDE_DIRS})
message(STATUS ${MLPACK_INCLUDE_DIRS})

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
