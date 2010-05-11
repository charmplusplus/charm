#include "ampi.h"

void *kernelSetup();

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Request req;
    MPI_Status sts;
    void *wr = kernelSetup();
    AMPI_GPU_invoke(wr, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &sts);

    MPI_Finalize();
    return 0;
}
