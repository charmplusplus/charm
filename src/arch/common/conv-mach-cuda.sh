BUILD_CUDA=1
CMK_INCDIR="-I$CUDA_DIR/include $CMK_INCDIR "
CMK_LIBDIR="-L$CUDA_DIR/lib64 -L$CUDA_DIR/extras/CUPTI/lib64 $CMK_LIBDIR "
CMK_LIBS="-lcudahybridapi -lcudart -lcupti $CMK_LIBS "
