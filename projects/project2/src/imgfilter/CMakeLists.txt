add_executable(imgfilter main.c single.c parallel.c)
target_link_libraries(imgfilter
  ppp_pnm
  ${MPI_C_LIBRARIES}
  m
  )
