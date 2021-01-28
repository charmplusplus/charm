#ifndef _MPI_INTEROPERATE_
#define _MPI_INTEROPERATE_

#include "converse.h"
#include "ck.h"
#include "trace.h"

#if CMK_CONVERSE_MPI
#include <mpi.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if CMK_CONVERSE_MPI
void CharmLibInit(MPI_Comm userComm, int argc, char **argv);
#else
void CharmLibInit(int userComm, int argc, char **argv);
#endif

void CharmBeginInit(int argc, char** argv);
void CharmFinishInit();
void CharmInit(int argc, char** argv);

void CharmLibExit();

void LibCkExit(void);

void StartCharmScheduler();
void StopCharmScheduler();

#ifdef __cplusplus
}
#endif

#undef CkExit
#define CkExit LibCkExit

#endif //_MPI_INTEROPERATE_
