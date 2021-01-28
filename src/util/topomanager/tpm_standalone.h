#ifdef __TPM_STANDALONE__

#ifndef _TPM_STANDALONE_H_
#define _TPM_STANDALONE_H_

// The TopoManager library depends on some charm++ functions/variables.
// To be able to work standalone most of these are made no-ops here to minimize changes
// to the code

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __cplusplus
# define CMI_EXTERN extern "C"
#else
# define CMI_EXTERN
#endif

extern int _tpm_numpes;
extern int _tpm_numthreads;

#define CmiAssert(c)
#define CmiAbort(m) fprintf(stderr, "%s", m); exit(1)
#define CmiNumPes() _tpm_numpes
#define CmiNumPesGlobal() _tpm_numpes
#define CmiMyNodeSize() _tpm_numthreads
#define CmiNodeOf(i) i
#define CmiNodeFirst(i) i
#define CmiNumPartitions() 1
#define CmiMyPartition() 0
#define CmiGetPeGlobal(pe,part) pe
#define CmiGetNodeGlobal(node,part) node
#define CmiNumCores() sysconf(_SC_NPROCESSORS_ONLN)

typedef int CmiNodeLock;
#define CmiCreateLock() 1
#define CmiLock(l)
#define CmiUnlock(l)

#define _MEMCHECK(ptr)

#endif

#endif
