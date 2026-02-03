cmake_minimum_required(VERSION 3.18)
project(eigh_standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Fix for NVHPC compiler compatibility with nanobind
if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
    add_compile_definitions(__GXX_ABI_VERSION=1016)
endif()

# Find required packages
# Use Development.Module instead of Development - manylinux containers
# don't have full Python development files (libpython), but extension
# modules don't need them anyway
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Find nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Find BLAS and LAPACK
# If NVHPC SDK paths are in environment, set hints for CMake
set(NVHPC_LIB_PATH "/softs/nvidia/hpc_sdk/Linux_x86_64/22.1/compilers/lib")
if(EXISTS "${NVHPC_LIB_PATH}/libblas.so")
    set(BLA_VENDOR "NVHPC")
    set(BLAS_LIBRARIES "${NVHPC_LIB_PATH}/libblas.so")
    set(LAPACK_LIBRARIES "${NVHPC_LIB_PATH}/liblapack.so")
    set(BLAS_FOUND TRUE)
    set(LAPACK_FOUND TRUE)
    message(STATUS "Using NVHPC BLAS: ${BLAS_LIBRARIES}")
    message(STATUS "Using NVHPC LAPACK: ${LAPACK_LIBRARIES}")
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

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

# Add NVHPC Fortran runtime when linking against NVHPC BLAS/LAPACK
if(EXISTS "${NVHPC_LIB_PATH}/libnvf.so")
    find_library(NVF_LIB NAMES nvf PATHS "${NVHPC_LIB_PATH}" NO_DEFAULT_PATH)
    if(NVF_LIB)
        target_link_libraries(eigh_lapack PRIVATE ${NVF_LIB} rt)
        message(STATUS "Found NVHPC Fortran runtime: ${NVF_LIB}")
    endif()
endif()

set_target_properties(eigh_lapack PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/python/eigh
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

    # Add CUDA include directories
    target_include_directories(eigh_cuda PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    set_source_files_properties(
        src/cuda/solver_kernels.cc
        PROPERTIES LANGUAGE CUDA
    )

    # Link CUDA libraries - handle both imported targets and direct paths
    if(TARGET CUDA::cudart AND TARGET CUDA::cusolver AND TARGET CUDA::cublas)
        target_link_libraries(eigh_cuda PRIVATE
            CUDA::cudart
            CUDA::cusolver
            CUDA::cublas
        )
    else()
        # Fallback for NVHPC SDK where imported targets may not be available
        target_link_libraries(eigh_cuda PRIVATE
            ${CUDAToolkit_LIBRARY_DIR}/libcudart.so
            ${CUDAToolkit_LIBRARY_DIR}/libcusolver.so
            ${CUDAToolkit_LIBRARY_DIR}/libcublas.so
        )
    endif()

    set_target_properties(eigh_cuda PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/python/eigh
        CUDA_SEPARABLE_COMPILATION ON
    )

    message(STATUS "CUDA support enabled")
else()
    message(STATUS "CUDA not found - GPU support will be disabled")
endif()

# Install target - scikit-build-core will handle the final install location
install(TARGETS eigh_lapack
    LIBRARY DESTINATION eigh
)

if(TARGET eigh_cuda)
    install(TARGETS eigh_cuda
        LIBRARY DESTINATION eigh
    )
endif()
