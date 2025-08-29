cmake_minimum_required(VERSION 3.19)
project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 90)

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# add_compile_options("--ptxas-options=-v")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src_padded)
# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/src_padded SRC_PADDED)

# generate executable
add_executable(conv2d conv2d.cu ${SRC})
target_include_directories(conv2d PRIVATE ${CUDA_CUDNN_DIR}/include)
set_target_properties(conv2d PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_directories(conv2d PRIVATE ${CUDA_CUDNN_DIR}/lib)
target_link_libraries(conv2d ${CUDA_LIBRARIES} cudnn)
target_compile_options(conv2d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>)
target_compile_options(conv2d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# generate executable
add_executable(conv2d_padded conv2d_padded.cu ${SRC_PADDED})
target_include_directories(conv2d_padded PRIVATE ${CUDA_CUDNN_DIR}/include)
set_target_properties(conv2d_padded PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_directories(conv2d_padded PRIVATE ${CUDA_CUDNN_DIR}/lib)
target_link_libraries(conv2d_padded ${CUDA_LIBRARIES} cudnn)
target_compile_options(conv2d_padded PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>)
target_compile_options(conv2d_padded PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_subdirectory(playbox)
