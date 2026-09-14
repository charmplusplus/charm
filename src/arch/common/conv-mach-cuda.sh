BUILD_CUDA=1
CMK_INCDIR="-I$CUDA_DIR/include -I$CUDA_DIR/extras/CUPTI/include $CMK_INCDIR "
CMK_LIBDIR="-L$CUDA_DIR/lib64 -L$CUDA_DIR/extras/CUPTI/lib64 $CMK_LIBDIR "
CMK_LIBS="-lhybridapi -lcudart -lcupti -lrt $CMK_LIBS "
# HAPI_CUPTI_LB turns on per-object GPU load measurement. It has to be a global
# define rather than one confined to hybridapi: the fields it guards live in
# LDObjData, which ck-core and ck-ldb see too, and a partial definition would
# give them different layouts for the same struct.
CMK_DEFS="$CMK_DEFS -DHAPI_CUPTI_LB "
