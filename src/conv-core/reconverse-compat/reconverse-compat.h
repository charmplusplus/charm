/*
  Classic Converse API that TCharm and AMPI use and reconverse does not
  provide, for reconverse builds only.

  Reconverse has user-level threads but no thread migration: no isomalloc,
  no migratable stacks, and no per-thread heap, globals or TLS switching
  ("interceptions"). The functions declared here and in memory-isomalloc.h
  are defined in reconverse-compat.C so that TCharm and AMPI build and run
  without migration:
    - CthMigratable() returns 0, so TCharm's migrateTo and TCHARM_Migrate
      print "thread migration is not supported" and return.
    - CthCreateMigratable creates an ordinary thread with CthCreate.
    - CthPup aborts: a thread cannot be packed.
    - CmiIsomallocContextCreate and CmiIsomallocGetThreadContext return a
      context whose opaque pointer is null; allocations through a context
      come from malloc and are not migratable.
    - The interception and globals hooks (CthInterceptions*, CtgInit) do
      nothing.
    - CthStackOffset and CthPointer are the identity on addresses, which is
      correct for a thread whose stack never moves.

  The four declarations at the end are functions libreconverse defines (with
  C++ linkage) but its converse.h does not declare.

  Include after converse.h.
*/
#ifndef RECONVERSE_COMPAT_H
#define RECONVERSE_COMPAT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Thread implementation flags, as in classic converse.h. CmiThreadIs
   returns 0 for every flag: reconverse threads are fcontext threads with
   their own malloc'd stack (no alias, stack-copy or TLS-switching flavor). */
#define CMI_THREAD_IS_QT         (1<<1)
#define CMI_THREAD_IS_CONTEXT    (1<<2)
#define CMI_THREAD_IS_UJCONTEXT  (1<<3)
#define CMI_THREAD_IS_PTHREADS   (1<<4)
#define CMI_THREAD_IS_FIBERS     (1<<5)
#define CMI_THREAD_IS_ALIAS      (1<<6)
#define CMI_THREAD_IS_STACKCOPY  (1<<7)
#define CMI_THREAD_IS_TLS        (1<<8)
int CmiThreadIs(int flag);

void CmiPrintStackTrace(int nSkip);

size_t CthStackOffset(CthThread t, char *p);
char  *CthPointer(CthThread t, size_t pos);

void CtgInit(void);

void CthInterceptionsDeactivatePush(CthThread th);
void CthInterceptionsDeactivatePop(CthThread th);
int  CthInterceptionsTemporarilyActivateStart(CthThread th);
void CthInterceptionsTemporarilyActivateEnd(CthThread th, int old);

#ifdef __cplusplus
}

/* Defined in libreconverse with C++ linkage. */
int       CthIsMainThread(CthThread t);
void      CthSetStrategy(CthThread t, CthAwkFn awkfn, CthThFn chsfn);
void      CthSetNext(CthThread t, CthThread next);
CthThread CthGetNext(CthThread t);
#endif

#endif
