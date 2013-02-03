#ifndef _MPI_INTEROPERATE_
#define _MPI_INTEROPERATE_

#define CMK_HAS_INTEROP 1

#include "converse.h"
#include "ck.h"
#include "trace.h"

#if CMK_CONVERSE_MPI
#include <mpi.h>
extern "C" void CharmLibInit(MPI_Comm userComm, int argc, char **argv);
#else
extern "C" void CharmLibInit(int userComm, int argc, char **argv);
#endif

extern "C" void CharmLibExit();

extern "C" void LibCkExit(void);

extern "C" void StartCharmScheduler();
#define CkExit LibCkExit

#endif //_MPI_INTEROPERATE_
