BUILD_CUDA=1
CMK_INCDIR="-I$CUDA_DIR/include -I$CUDA_DIR/extras/CUPTI/include $CMK_INCDIR "
# -lcuda: the direct CUDA IPC transport needs cuMemGetAddressRange to find the
# base of the allocation holding an interior pointer, which only the driver API
# exposes. libcuda ships with the driver, so it is already on the system linker
# path wherever a CUDA build can actually run. The stubs directory is listed
# LAST, so it only supplies libcuda on a driverless build host -- put it any
# earlier and the stub, which has no working implementation, wins at link time
# and the binary fails on a machine that does have a driver.
CMK_LIBDIR="-L$CUDA_DIR/lib64 -L$CUDA_DIR/extras/CUPTI/lib64 $CMK_LIBDIR -L$CUDA_DIR/lib64/stubs "
CMK_LIBS="-lhybridapi -lcudart -lcuda -lcupti -lrt $CMK_LIBS "
