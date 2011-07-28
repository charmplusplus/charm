/*****************************************************************************
 * $Source$
 * $Author$ 
 * $Date$
 * $Revision$
 *****************************************************************************/

/* Sanjay's pooling allocator adapted for cmialloc usage*/
#ifndef GNIPOOL_H
#define GNIPOOL_H  


#if defined(__cplusplus)
extern "C" {
#endif
#define GNI_POOL_HEADER_SIZE 8
#define GNI_POOL_DEFAULT_BINS 30
#include "converse.h"

#define GNI_POOL_HEADER_SIZE 8

void GniPoolPrintList(char *p); 

void GniPoolAllocInit(int numBins);


void * GniPoolAlloc(unsigned int numBytes);

void  GniPoolFree(void * p);
void  GniPoolAllocStats();

/* theoretically we should put a pool cleanup function in here */

#if defined(__cplusplus)
}
#endif

#endif /* GNIPOOL.H */
