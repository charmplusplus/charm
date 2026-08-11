BUILD_CUDA=1
CMK_INCDIR="-I$CUDA_DIR/include -I$CUDA_DIR/extras/CUPTI/include $CMK_INCDIR "
CMK_LIBDIR="-L$CUDA_DIR/lib64 -L$CUDA_DIR/extras/CUPTI/lib64 $CMK_LIBDIR "
CMK_LIBS="-lhybridapi -lcudart -lcupti -lrt $CMK_LIBS "
