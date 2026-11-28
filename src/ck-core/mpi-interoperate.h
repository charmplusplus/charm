#ifndef _MPI_INTEROPERATE_
#define _MPI_INTEROPERATE_

#include "converse.h"
#include "ck.h"
#include "trace.h"

// CharmLibInit takes an MPI communicator, so it is declared only in a
// translation unit that includes mpi.h before this header (or defines
// CMK_INTEROP_USE_MPI). Merely finding mpi.h on the include path is not
// enough: Open MPI's mpi.h pulls its C++ bindings into every C++ file, which
// then needs libmpi_cxx at link time. libck itself needs no MPI.
#if CMK_CONVERSE_MPI || defined(MPI_VERSION) || defined(CMK_INTEROP_USE_MPI)
#define CMK_INTEROP_HAVE_MPI_H 1
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

void CharmLibInitWorld(int argc, char **argv);
int CharmInteropPickRendezvous(char *host, int hostlen);
void CharmLibInitWorldTcp(int rank, int size, int local_rank, int local_size,
                          const char *host, int port, int argc, char **argv);
void CharmInteropSetTcpBootstrap(int rank, int size, int local_rank,
                                 int local_size, const char *host, int port);
int CharmMpiMainStart(int rank, int size, int serialized, const char *userPmi,
                      int argc, char **argv);

void CharmBeginInit(int argc, char** argv);
void CharmFinishInit();
void CharmInit(int argc, char** argv);

// Returns the exit code the application gave to CkExit, the same on every
// process; 0 if the exit was started by CharmLibExit itself. After
// CharmLibInit, call it once per process: a second CharmLibExit, or a
// CharmLibInit after it, prints a message and aborts.
int CharmLibExit();

void LibCkExit(void);

void StartCharmScheduler();
void StopCharmScheduler();

#ifdef __cplusplus
}
#endif

#if CMK_INTEROP_HAVE_MPI_H
// The abort hook of CharmLibInit and CharmMpiMain (CmiSetAbortHook): a
// CkAbort or CmiAbort in any
// process ends the whole MPI job. Without it an abort ends only its own
// process, because LCI is bootstrapped over LCT's tcp backend, whose
// process-manager abort reaches no other. CmiAbort has already printed msg;
// the hook says itself that it ends the job, since the MPI library's own
// report can be lost (Open MPI's mpirun drops it at times). It runs on
// whichever PE thread aborted, which with several PEs per process is not the
// thread that initialized MPI (see CharmLibInit). If the program has already
// finalized MPI, it does nothing and CmiAbort goes on as usual.
static inline void CharmInteropAbortHook(int code, const char *msg) {
  (void)msg;
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized) return;
  fprintf(stderr, "MPI interop: ending the MPI job with MPI_Abort(MPI_COMM_WORLD, %d)\n",
          code);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, code);
}

// The part of an MPI interop start that needs MPI, shared by CharmLibInit and
// CharmMpiMain (who names the caller in messages): checks that userComm is
// MPI_COMM_WORLD or congruent to it, selects LCT's tcp bootstrap for LCI with
// the rendezvous at rank 0, so the launcher's PMI is left to MPI, and
// registers the abort hook. Returns this process's rank and the number of
// ranks, and in userPmi the LCT_PMI_BACKEND the user had set ("" if none).
static inline void CharmInteropBootstrap(MPI_Comm userComm, const char *who,
                                         int *rankOut, int *sizeOut,
                                         char *userPmi, int userPmiLen) {
#if defined(_WIN32)
  fprintf(stderr, "%s: MPI interop is not supported on Windows\n", who);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  int cmp = MPI_UNEQUAL, rank = -1, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_compare(userComm, MPI_COMM_WORLD, &cmp);
  if (cmp != MPI_IDENT && cmp != MPI_CONGRUENT) {
    fprintf(stderr,
            "%s (rank %d): the communicator must be MPI_COMM_WORLD "
            "or congruent to it. Charm++ on reconverse runs on every rank of "
            "the job; running it on a subset of ranks (a split communicator) "
            "is not supported.\n", who, rank);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  // A launcher that does not match the MPI library makes every process a
  // 1-rank MPI job; the launcher's own task count still says how many it
  // started. Checked before Charm++ starts, so the warning comes first.
  if (size == 1 && rank == 0) {
    const char *vars[] = {"OMPI_COMM_WORLD_SIZE", "PMI_SIZE",
                          "SLURM_STEP_NUM_TASKS"};
    for (const char *var : vars) {
      const char *val = getenv(var);
      if (val == NULL) continue;
      if (atoi(val) > 1) {
        fprintf(stderr,
                "Warning: %s: MPI_COMM_WORLD has 1 rank, but %s=%s: "
                "the launcher may have started %s processes, each its own MPI "
                "job. Launch with the launcher your MPI requires (for Cray "
                "MPICH under Slurm, srun --mpi=cray_shasta).\n", who, var, val,
                val);
        fflush(stderr);
      }
      break;
    }
  }
  char host[256] = {0};
  int port = -1;
  if (rank == 0) port = CharmInteropPickRendezvous(host, sizeof(host));
  MPI_Bcast(&port, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(host, sizeof(host), MPI_CHAR, 0, MPI_COMM_WORLD);
  if (port < 0) {
    if (rank == 0)
      fprintf(stderr, "%s: no TCP port for the LCI bootstrap\n", who);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Comm local;
  int local_rank = 0, local_size = 1;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                      MPI_INFO_NULL, &local);
  MPI_Comm_rank(local, &local_rank);
  MPI_Comm_size(local, &local_size);
  MPI_Comm_free(&local);
  // Read before CharmInteropSetTcpBootstrap, which sets the variable when it
  // is unset.
  userPmi[0] = '\0';
  if (const char *p = getenv("LCT_PMI_BACKEND"))
    snprintf(userPmi, userPmiLen, "%s", p);
  // Before Charm++ starts, so that an abort during startup ends the job too.
  CmiSetAbortHook(CharmInteropAbortHook);
  CharmInteropSetTcpBootstrap(rank, size, local_rank, local_size, host, port);
  *rankOut = rank;
  *sizeOut = size;
}

// Library mode: starts Charm++ inside an MPI program. Charm++ runs on every
// rank of the job: userComm must be MPI_COMM_WORLD or congruent to it, and
// Charm++ process i is MPI_COMM_WORLD rank i. After CharmLibExit returns the
// program may keep using MPI and must call MPI_Finalize itself. With several
// PEs per process, initialize MPI with
// MPI_Init_thread(..., MPI_THREAD_SERIALIZED, ...): an abort on a PE calls
// MPI_Abort from a thread MPI did not create, while the main thread, inside
// Charm++, makes no MPI call. CharmLibExit removes the abort hook.
static inline void CharmLibInit(MPI_Comm userComm, int argc, char **argv) {
  int rank = -1, size = 0;
  char userPmi[32] = "";
  CharmInteropBootstrap(userComm, "CharmLibInit", &rank, &size, userPmi,
                        sizeof(userPmi));
  CharmLibInitWorld(argc, argv);
  if (CmiMyNode() != rank || CmiNumNodes() != size) {
    fprintf(stderr,
            "CharmLibInit: Charm++ process %d of %d is MPI_COMM_WORLD rank %d "
            "of %d; the launcher gave Charm++ and MPI different process "
            "numbering, which MPI interop requires to match.\n",
            CmiMyNode(), CmiNumNodes(), rank, size);
    if (userPmi[0])
      fprintf(stderr,
              "CharmLibInit: LCT_PMI_BACKEND=%s was set in the environment, "
              "so LCI did not use the tcp bootstrap that CharmLibInit sets "
              "up; unset it for MPI interop.\n", userPmi);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

// CharmMpiMain's exit hook (CmiSetExitHook): the runtime calls it once per
// process, on the thread that called CharmMpiMain, after the exit sequence
// and immediately before exit(code).
static inline void CharmInteropExitHook(int code) {
  (void)code;
  int finalized = 0, isMain = 1;
  MPI_Finalized(&finalized);
  if (finalized) return;
  MPI_Is_thread_main(&isMain);
  if (!isMain) {
    fprintf(stderr, "Warning: CharmMpiMain: MPI_Finalize is called on a thread "
            "other than the one that initialized MPI\n");
    fflush(stderr);
  }
  MPI_Finalize();
}

// Main mode: runs a Charm++ program with a mainchare inside an MPI job as an
// ordinary Charm++ run. Call it from main, on every rank, as
// "return CharmMpiMain(argc, argv);" (charmc -mpi-main supplies such a main).
// It initializes MPI with MPI_THREAD_SERIALIZED unless MPI is already
// initialized, starts Charm++ on every rank of MPI_COMM_WORLD with Charm++
// process i being rank i, and does not return: CkExit(code) ends the program
// as in any Charm++ run, MPI_Finalize is called on this thread just before
// each process exits with code, and a CkAbort ends the job through
// MPI_Abort. With several PEs per process the run aborts at startup if MPI
// provides less than MPI_THREAD_SERIALIZED.
static inline int CharmMpiMain(int argc, char **argv) {
  int initialized = 0, provided = MPI_THREAD_SINGLE;
  MPI_Initialized(&initialized);
  if (initialized)
    MPI_Query_thread(&provided);
  else
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  int rank = -1, size = 0;
  char userPmi[32] = "";
  CharmInteropBootstrap(MPI_COMM_WORLD, "CharmMpiMain", &rank, &size, userPmi,
                        sizeof(userPmi));
  CmiSetExitHook(CharmInteropExitHook);
  return CharmMpiMainStart(rank, size, provided >= MPI_THREAD_SERIALIZED,
                           userPmi, argc, argv);
}
#endif

#undef CkExit
#define CkExit LibCkExit

#endif //_MPI_INTEROPERATE_
