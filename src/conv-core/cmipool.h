/*****************************************************************************
 * $Source$
 * $Author$ 
 * $Date$
 * $Revision$
 *****************************************************************************/

/* Sanjay's pooling allocator adapted for cmialloc usage*/
  

#define CMI_POOL_HEADER_SIZE 8
#define CMI_POOL_DEFAULT_BINS 30
#include "converse.h"

#define CMI_POOL_HEADER_SIZE 8


void CmiPoolPrintList(char *p); 

void CmiPoolAllocInit(int numBins);


void * CmiPoolAlloc(unsigned int numBytes);

void * CmiPoolFree(char * p);
void  CmiPoolAllocStats();

/* theoretically we should put a pool cleanup function in here */



