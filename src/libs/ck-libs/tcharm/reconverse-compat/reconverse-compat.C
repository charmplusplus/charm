/*
  Definitions for the classic Converse API that reconverse lacks. See
  reconverse-compat.h for what each one does and why. Linked into
  libmoduletcharm, so every TCharm user (AMPI, collide, mblock, ...) gets
  them.
*/
#include "conv-config.h"

#include "converse.h"
#include "pup_c.h"
#include "reconverse-compat.h"

CmmTable CmmPup(pup_er p, CmmTable t, CmmPupMessageFn msgpup)
{
  CmiAbort("CmmPup: packing a message table (migration or checkpointing of "
           "an mblock chunk) is not supported on reconverse.\n");
  return t;
}
