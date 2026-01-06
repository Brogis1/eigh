cmake_minimum_required(VERSION 3.18)
project(eigh_standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Find nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Find BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# XLA headers (from jaxlib)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import jaxlib; import os; print(os.path.join(jaxlib.__path__[0], 'include'))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_INCLUDE_DIR
    RESULT_VARIABLE XLA_RESULT
)

if(NOT XLA_RESULT EQUAL 0 OR NOT EXISTS "${XLA_INCLUDE_DIR}")
    message(FATAL_ERROR "Could not find XLA headers. Make sure jaxlib is installed: pip install jaxlib")
endif()

message(STATUS "XLA headers found at: ${XLA_INCLUDE_DIR}")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/cpu
    ${CMAKE_SOURCE_DIR}/src/cuda
    ${XLA_INCLUDE_DIR}
)

# CPU LAPACK module
nanobind_add_module(
    eigh_lapack
    STABLE_ABI
    src/cpu/lapack.cc
    src/cpu/lapack_kernels.cc
)
target_link_libraries(eigh_lapack PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
set_target_properties(eigh_lapack PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/python
)

# CUDA module (optional - only if CUDA is available)
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit REQUIRED)

    nanobind_add_module(
        eigh_cuda
        STABLE_ABI
        src/cuda/solver.cc
        src/cuda/solver_kernels.cc
    )

    set_source_files_properties(
        src/cuda/solver_kernels.cc
        PROPERTIES LANGUAGE CUDA
    )

    target_link_libraries(eigh_cuda PRIVATE
        CUDA::cudart
        CUDA::cusolver
        CUDA::cublas
    )

    set_target_properties(eigh_cuda PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/python
        CUDA_SEPARABLE_COMPILATION ON
    )

    message(STATUS "CUDA support enabled")
else()
    message(STATUS "CUDA not found - GPU support will be disabled")
endif()

# Install target
install(TARGETS eigh_lapack
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/src/python
)

if(TARGET eigh_cuda)
    install(TARGETS eigh_cuda
        LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/src/python
    )
endif()
