BUILD_HIP=1
ROCM_DIR="${ROCM_PATH:-/opt/rocm}"
CMK_INCDIR="-I$ROCM_DIR/include $CMK_INCDIR "
CMK_LIBDIR="-L$ROCM_DIR/lib $CMK_LIBDIR "
# amdhip64: the HIP runtime (streams, memory, IPC, the virtual-memory pool).
# rocprofiler-sdk: the balancer's GPU load tracing (hapi_trace_rocprof.cpp);
# rocprofiler-sdk-roctx: the +nvtx ranges on this vendor.
CMK_LIBS="-lhybridapi -lamdhip64 -lrocprofiler-sdk -lrocprofiler-sdk-roctx -lrt $CMK_LIBS "
