add_executable(conv2d_padded conv2d_padded.cu)
set_target_properties(conv2d_padded PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(conv2d_padded ${CUDA_LIBRARIES})

add_executable(tma tma_try.cu)
set_target_properties(tma PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(tma ${CUDA_LIBRARIES})

