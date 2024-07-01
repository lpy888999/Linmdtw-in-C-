cmake_minimum_required(VERSION 3.22.1)
project(dtwC LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 指定CUDA编译器路径（可选）
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# 找到CUDA
find_package(CUDAToolkit REQUIRED)
# 找到 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 找到 FFTW 和 RubberBand
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3f)
pkg_check_modules(RUBBERBAND REQUIRED rubberband)
pkg_check_modules(SNDFILE REQUIRED sndfile)
include_directories(${FFTW_INCLUDE_DIRS} ${RUBBERBAND_INCLUDE_DIRS} ${SNDFILE_INCLUDE_DIRS})

# 包含头文件目录
include_directories(.)

# 设置源文件列表
set(SOURCES
        dtw.cpp
        alignmenttools.cpp
        linmdtw.cpp
        main.cpp
        DTWGPU.cu
        dtw_diag_gpu.cu
        dynseqalign.cpp
        auditools.cpp
)

# 添加可执行文件
add_executable(dtwC ${SOURCES})

# 链接CUDA库、OpenCV库、FFTW库、RubberBand库和libsndfile库
target_link_libraries(dtwC
        CUDA::cudart
        ${OpenCV_LIBS}
        ${FFTW_LIBRARIES}
        ${RUBBERBAND_LIBRARIES}
        ${SNDFILE_LIBRARIES}
)

# 设置CUDA编译选项（根据需要进行调整）
set_target_properties(dtwC PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "50"
)
