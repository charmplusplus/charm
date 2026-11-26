// main() of a Charm++ program linked with charmc -mpi-main: an ordinary
// Charm++ run inside an MPI job (see CharmMpiMain in mpi-interoperate.h).
// charmc compiles this file at link time with the MPI compiler given to it.
#include <mpi.h>
#include "mpi-interoperate.h"

int main(int argc, char **argv) { return CharmMpiMain(argc, argv); }
