BUILD_HIP=1
CMK_INCDIR="-I/opt/rocm/include $CMK_INCDIR "
CMK_LIBDIR="-L/opt/rocm/lib $CMK_LIBDIR "
CMK_LIBS="-lhybridapi -lamdhip64 $CMK_LIBS "
