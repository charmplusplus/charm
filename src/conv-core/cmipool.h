/* Sanjay's pooling allocator adapted for cmialloc usage*/
#ifndef CMIPOOL_H
#define CMIPOOL_H  

#define CMI_POOL_HEADER_SIZE 8
#define CMI_POOL_DEFAULT_BINS 30
#include "converse.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define CMI_POOL_HEADER_SIZE 8

void CmiPoolPrintList(char *p); 

void CmiPoolAllocInit(int numBins);


void * CmiPoolAlloc(unsigned int numBytes);

void  CmiPoolFree(void * p);
void  CmiPoolAllocStats(void);

/* theoretically we should put a pool cleanup function in here */

#if defined(__cplusplus)
}
#endif

#endif /* CMIPOOL.H */
