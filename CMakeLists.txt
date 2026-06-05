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
# modules don't need them anyway.
# Development.SABIModule is additionally required for nanobind's STABLE_ABI
# (abi3) build: without it the Python::SABIModule target is absent and nanobind
# silently falls back to a version-locked cpython-3XX module, which defeats the
# single-abi3-wheel-per-platform goal. Building must use CPython >= 3.12.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)

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

# CUDA module. By default it is optional and built only when a CUDA compiler is
# found (CPU-only wheels skip it). Setting EIGH_REQUIRE_CUDA=ON makes a missing
# nvcc a hard error, so the eigh-cuda12 package build can never silently produce
# a CPU-only wheel under a GPU name.
option(EIGH_REQUIRE_CUDA "Fail the build if CUDA toolchain is unavailable" OFF)

# Resolve the CUDA compiler robustly. If the caller passed CMAKE_CUDA_COMPILER
# explicitly (e.g. the eigh-cuda12 CI passes /usr/local/cuda/bin/nvcc), trust it
# and skip check_language(CUDA) — that probe runs its own detection and can fail
# to honor a pre-set compiler inside cibuildwheel's environment. Otherwise fall
# back to autodetection (CUDACXX env, then check_language) for source builds.
if(NOT CMAKE_CUDA_COMPILER AND DEFINED ENV{CUDACXX})
    set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "CUDA compiler")
endif()

if(NOT CMAKE_CUDA_COMPILER)
    include(CheckLanguage)
    check_language(CUDA)
endif()

if(CMAKE_CUDA_COMPILER AND NOT EXISTS "${CMAKE_CUDA_COMPILER}")
    message(WARNING "CMAKE_CUDA_COMPILER set to '${CMAKE_CUDA_COMPILER}' but that "
                    "path does not exist; treating CUDA as unavailable.")
    unset(CMAKE_CUDA_COMPILER)
    unset(CMAKE_CUDA_COMPILER CACHE)
endif()

if(EIGH_REQUIRE_CUDA AND NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
        "EIGH_REQUIRE_CUDA=ON but no CUDA compiler (nvcc) was found. "
        "The eigh-cuda12 wheel must be built with the CUDA toolkit available.")
endif()

if(CMAKE_CUDA_COMPILER)
    # nvcc enforces a host-compiler version check and refuses GCC newer than the
    # CUDA toolkit officially supports. The CI build image pairs CUDA 12.8 with
    # GCC 14 (a supported combination), so this is OFF by default. Turn it ON
    # (-DEIGH_CUDA_ALLOW_UNSUPPORTED=ON) only if you must build with an
    # officially-unsupported toolkit/GCC pairing (e.g. CUDA 12.3 + GCC 14); it
    # bypasses the check but may then hit header incompatibilities. Must be set
    # BEFORE enable_language(CUDA), since the compiler-id test trips the check.
    option(EIGH_CUDA_ALLOW_UNSUPPORTED
        "Pass -allow-unsupported-compiler to nvcc (for GCC > toolkit max)" OFF)
    if(EIGH_CUDA_ALLOW_UNSUPPORTED)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler"
            CACHE STRING "CUDA flags" FORCE)
    endif()

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

    # When the eigh-cudaXXX wheel depends on NVIDIA's pip CUDA wheels, the CUDA
    # runtime libraries live in site-packages/nvidia/*/lib, NOT next to this .so.
    # The module installs to site-packages/eigh/, so add RPATH entries that climb
    # to site-packages (../) and into each nvidia/<lib>/lib directory.
    #
    # CRITICAL: this must cover EVERY library in the .so's NEEDED list, including
    # the ones cuSOLVER pulls in TRANSITIVELY. The .so links cudart/cusolver/
    # cublas directly, but cusolver also needs cublasLt, cusparse, nvJitLink (and
    # nvrtc). Missing any one -> `ImportError: libXXX.so.N: cannot open shared
    # object file` at import on a real machine. Verify after a build with:
    #   readelf -d eigh_cuda*.so | grep NEEDED   # every libcu*/libnv* must have
    #                                            # a matching RPATH dir below.
    # ${ORIGIN} is escaped so it is written literally into the ELF RPATH.
    if(EIGH_REQUIRE_CUDA)
        set_target_properties(eigh_cuda PROPERTIES
            BUILD_RPATH_USE_ORIGIN ON
            INSTALL_RPATH "\$ORIGIN/../nvidia/cuda_runtime/lib;\$ORIGIN/../nvidia/cusolver/lib;\$ORIGIN/../nvidia/cublas/lib;\$ORIGIN/../nvidia/cusparse/lib;\$ORIGIN/../nvidia/cuda_nvrtc/lib;\$ORIGIN/../nvidia/nvjitlink/lib"
        )
    endif()

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
