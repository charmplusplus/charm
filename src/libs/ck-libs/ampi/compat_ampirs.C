#include "charm-api.h"

/* Declared locally so this stub does not have to pull in converse.h -- unless
   something already has, in which case the sentinel says so. */
#ifndef CMI_ISOMALLOC_CONTEXT_DEFINED
#define CMI_ISOMALLOC_CONTEXT_DEFINED 1
typedef struct CmiIsomallocContext {
  void * opaque;
} CmiIsomallocContext;
#endif

CLINKAGE void AMPI_Rank_Setup(int myrank, int numranks, CmiIsomallocContext ctx);
void AMPI_Rank_Setup(int myrank, int numranks, CmiIsomallocContext ctx)
{
}
