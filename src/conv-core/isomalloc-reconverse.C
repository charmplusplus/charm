/**************************************************************************
Isomalloc <-> Reconverse glue.

Reconverse owns user-level threads but not Isomalloc: Isomalloc's context
serialization is written against PUP, which belongs to the layer above
Reconverse, so the two cannot simply be merged. Instead Reconverse declares
the handful of operations a migratable thread needs (CthIsomallocOps) and
this file registers Charm++'s Isomalloc as the provider.

Two directions of adaptation live here:

  - Down: the CthIsomallocOps table, plus registerIsomallocInit so that
    CmiIsomallocInit runs at the right point in Reconverse's per-PE startup.

  - Up: CthPup(pup_er, ...), the signature TCharm and the rest of Charm++
    call, wrapping Reconverse's serializer-agnostic CthPupThread.

Only built for Reconverse targets; the classic Converse build has all of this
inside threads.C and isomalloc.C already.
 *************************************************************************/

/* Angle brackets, not quotes: this file sits next to Charm++'s own
   converse.h, and under Reconverse the converse.h that must win is the one on
   the include path, not the sibling. */
#include <converse.h>
#include "conv-autoconfig.h"

#if CMK_RECONVERSE

#include "memory-isomalloc.h"
#include "pup.h"
#include "pup_c.h"

/****************** Down: Isomalloc as a thread-stack provider **************/

/* Defined in isomalloc.C. Declared here rather than in memory-isomalloc.h
   because that header is extern "C" and isomalloc.C's definition is not. */
void CmiIsomallocInit(char ** argv);

/* Isomalloc calls into the memory module to keep its own bookkeeping
   allocations out of the context it is servicing, and the glue below activates
   a context for malloc interception. Both live in memory-isomalloc.C, which is
   only linked when the program asked for `charmc -memory isomalloc`. Weak
   no-ops stand in otherwise, so that Isomalloc is usable for thread stacks
   even in a program whose heap is not intercepted; the real definitions
   override these whenever the memory module is present. */
CLINKAGE __attribute__((weak)) void CmiMemoryIsomallocDisablePush(void) {}
CLINKAGE __attribute__((weak)) void CmiMemoryIsomallocDisablePop(void) {}
CLINKAGE __attribute__((weak)) void
CmiMemoryIsomallocContextActivate(CmiIsomallocContext ctx) {}

static int iso_enabled(void) { return CmiIsomallocEnabled(); }

/* Isomalloc puts a migratable thread's stack and heap at addresses every
   process agrees on, but it has no say over where the program's own code
   lands. A suspended thread's stack holds return addresses into that code, so
   if the loader placed it somewhere else in the destination process, resuming
   the thread jumps into whatever is there now -- a segmentation fault with no
   other symptom and nothing pointing at the cause. Say so once, at the moment
   the first migratable thread is created, and only where it could bite: a
   single-process job never migrates anything between address spaces. */
static void warnOnceAboutAslr(void)
{
  static bool warned = false;
  extern int CmiIsomallocAddressSpaceIsRandomized(void);

  if (warned || CmiMyPe() != 0 || CmiNumNodes() <= 1)
    return;
  warned = true;

  if (!CmiIsomallocAddressSpaceIsRandomized())
    return;

  CmiPrintf("Isomalloc> Warning: address space randomization is enabled, and "
            "this job just created a migratable thread. Moving one between "
            "processes needs the program's code at the same address in both; "
            "run under `setarch -R`, or disable randomize_va_space, if these "
            "threads will migrate.\n");
}

static void * iso_permanentAllocAlign(CmiIsomallocContext ctx, size_t align,
                                      size_t size)
{
  warnOnceAboutAslr();
  return CmiIsomallocContextPermanentAllocAlign(ctx, align, size);
}

static void iso_contextDelete(CmiIsomallocContext ctx)
{
  CmiIsomallocContextDelete(ctx);
}

/* The stream's `user` is the PUP::er that Charm++ is driving; recovering it
   is what lets the context be pupped with the real framework even though
   Reconverse only knows about the byte-stream shape. */
static void iso_contextPup(CmiPupStream * p, CmiIsomallocContext * ctx)
{
  auto & pupper = *(PUP::er *)p->user;
  CmiIsomallocContextPup((pup_er)&pupper, ctx);
}

static void iso_contextActivate(CmiIsomallocContext ctx)
{
  CmiMemoryIsomallocContextActivate(ctx);
}

static const CthIsomallocOps cmiIsomallocOps = {
  iso_enabled,
  iso_permanentAllocAlign,
  iso_contextDelete,
  iso_contextPup,
  iso_contextActivate,
};

/* Present only when the program linked a memory module (charmc -memory ...).
   Without one there is nothing to initialize and nothing intercepts malloc.
   C++ linkage, to match memory.C's definition: Reconverse's converse.h does
   not declare this, so memory.C defines it unmangled-by-extern-"C". */
__attribute__((weak)) void CmiMemoryInit(char ** argv);

static void CmiIsomallocInitForReconverse(char ** argv)
{
  /* Before Isomalloc: the memory module owns the per-PE "which context is
     receiving mallocs" state that Isomalloc pushes and pops around its own
     bookkeeping allocations. */
  if (CmiMemoryInit != NULL)
    CmiMemoryInit(argv);

  CmiIsomallocInit(argv);
  /* After init, so that CthMigratable() answers with a region that has
     actually been negotiated rather than one that is about to be. */
  CthRegisterIsomallocOps(&cmiIsomallocOps);
}

/* Called from charm_main, before ConverseInit. */
void CmiIsomallocHookIntoReconverse(void)
{
  registerIsomallocInit(CmiIsomallocInitForReconverse);
}

/****************** Up: the PUP-shaped thread serializer ********************/

static void charmPupBytes(CmiPupStream * p, void * data, size_t n)
{
  if (n == 0) return;
  auto & pupper = *(PUP::er *)p->user;
  pupper((char *)data, (size_t)n);
}

CthThread CthPup(pup_er cpup, CthThread t)
{
  PUP::er & p = *(PUP::er *)cpup;

  CmiPupStream s;
  s.user = &p;
  s.bytes = charmPupBytes;
  s.mode = 0;
  if (p.isSizing()) s.mode |= CMI_PUP_SIZING;
  if (p.isPacking()) s.mode |= CMI_PUP_PACKING;
  if (p.isUnpacking()) s.mode |= CMI_PUP_UNPACKING;
  if (p.isDeleting()) s.mode |= CMI_PUP_DELETING;

  return CthPupThread(&s, t);
}

CmiIsomallocContext CmiIsomallocGetThreadContext(CthThread th)
{
  return CthGetIsomallocContext(th);
}

#endif /* CMK_RECONVERSE */
