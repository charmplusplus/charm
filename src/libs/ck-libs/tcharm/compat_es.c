#include "charm-api.h"

/* Declared locally so this stub does not have to pull in converse.h -- unless
   something already has, in which case the sentinel says so. */
#ifndef CMI_ISOMALLOC_CONTEXT_DEFINED
#define CMI_ISOMALLOC_CONTEXT_DEFINED 1
typedef struct CmiIsomallocContext {
  void * opaque;
} CmiIsomallocContext;
#endif

CLINKAGE void TCHARM_Element_Setup(int myelement, int numelements, CmiIsomallocContext ctx);
void TCHARM_Element_Setup(int myelement, int numelements, CmiIsomallocContext ctx)
{
}
