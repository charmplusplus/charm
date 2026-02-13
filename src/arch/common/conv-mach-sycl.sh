BUILD_SYCL=1
CMK_INCDIR="-I/opt/intel/oneapi/level-zero/latest/include $CMK_INCDIR "
CMK_LIBDIR="-L/opt/intel/oneapi/level-zero/latest/lib $CMK_LIBDIR "
CMK_LIBS="-lhybridapi -lze_loader -lrt -fsycl $CMK_LIBS "
