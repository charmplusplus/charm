/*
  Classic Converse API that TCharm and AMPI use and reconverse does not
  provide, for reconverse builds only.

  Thread migration itself is not here: reconverse provides migratable
  threads (CthCreateMigratable, CthPupThread, CthStackOffset, CthPointer,
  the CthInterceptions* calls, CmiThreadIs, CtgInit), and Charm++'s
  isomalloc library (src/conv-core/isomalloc.C, isomalloc-reconverse.C)
  provides isomalloc and CthPup on top of it. What is left is:
    - CmmPup aborts (see below).
    - Four functions libreconverse defines (with C++ linkage) but its
      converse.h does not declare.

  Include after converse.h.
*/
#ifndef RECONVERSE_COMPAT_H
#define RECONVERSE_COMPAT_H

#include <stddef.h>
#include "pup_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Reconverse's message manager (CmmTable) has no pup routine. CmmPup aborts:
   its only caller is mblock's MBlockChunk::pup, which packs a chunk's pending
   messages, so an mblock chunk cannot migrate or be checkpointed on
   reconverse. */
typedef void (*CmmPupMessageFn)(pup_er p, void **msg);
CmmTable CmmPup(pup_er p, CmmTable t, CmmPupMessageFn msgpup);

#ifdef __cplusplus
}

/* Defined in libreconverse with C++ linkage. */
int       CthIsMainThread(CthThread t);
void      CthSetStrategy(CthThread t, CthAwkFn awkfn, CthThFn chsfn);
void      CthSetNext(CthThread t, CthThread next);
CthThread CthGetNext(CthThread t);
#endif

#endif
