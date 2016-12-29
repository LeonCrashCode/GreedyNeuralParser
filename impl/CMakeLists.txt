CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

foreach(TARGET
	greedy
	greedy_attention
	greedy_attention_s
	greedy_attention_b
	greedy_attention_sb
	greedy_attention_sb_cond
	greedy_attention_sb_noC

	greedy_attention_sb_tree
	greedy_attention_sb_tree_label
	greedy_attention_sbd_tree_label
	greedy_attention_sb_tree2)
  ADD_EXECUTABLE(${TARGET} ${TARGET}.cc)
  if (WITH_CUDA_BACKEND)
    target_link_libraries(${TARGET} gdynet ${LIBS})
    add_dependencies(${TARGET} dynetcuda)
    target_link_libraries(${TARGET} dynetcuda)
    CUDA_ADD_CUBLAS_TO_TARGET(${TARGET})
  else()
    target_link_libraries(${TARGET} dynet ${LIBS})
  endif (WITH_CUDA_BACKEND)
  if(UNIX AND NOT APPLE)
    target_link_libraries(${TARGET} rt)
  endif()
endforeach()

