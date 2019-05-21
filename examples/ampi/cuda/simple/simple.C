#include "ampi.h"
#include "hapi.h"

#if USE_WR
extern hapiWorkRequest* setupWorkRequest(cudaStream_t stream);
#else
extern void invokeKernel(cudaStream_t stream);
#endif

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Request req;
    MPI_Status sts;

    cudaStream_t stream;

#if USE_WR
    // Use deprecated workRequest API
    hapiWorkRequest* wr = setupWorkRequest(stream);
    AMPI_GPU_Iinvoke_wr(wr, &req);
    MPI_Wait(&req, &sts);
#else
    // Use HAPI
    invokeKernel(stream);
    AMPI_GPU_Iinvoke(stream, &req);
    MPI_Wait(&req, &sts);
#endif

    MPI_Finalize();

    return 0;
}
