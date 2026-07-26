BUILD_CUDA=1
CMK_INCDIR="-I$CUDA_DIR/include -I$CUDA_DIR/extras/CUPTI/include $CMK_INCDIR "
# lib64/stubs provides libcuda.so for linking on nodes without a driver
# (the real driver library is loaded at runtime via its soname).
CMK_LIBDIR="-L$CUDA_DIR/lib64 -L$CUDA_DIR/extras/CUPTI/lib64 -L$CUDA_DIR/lib64/stubs $CMK_LIBDIR "
# -lcuda: hybridapi's +gpuflagpoll uses the driver API (cuStreamWriteValue32)
CMK_LIBS="-lhybridapi -lcudart -lcuda -lcupti -lrt $CMK_LIBS "
