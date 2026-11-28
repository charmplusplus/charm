// main() of a Charm++ program with a mainchare (exitcode.C) run in main mode:
// an ordinary Charm++ run inside an MPI job. CharmMpiMain initializes MPI,
// starts Charm++ and does not return; CkExit(code) ends each process with
// code, after MPI_Finalize. charmc -mpi-main supplies the same main.
#include <mpi.h>
#include "mpi-interoperate.h"

int main(int argc, char **argv) { return CharmMpiMain(argc, argv); }
