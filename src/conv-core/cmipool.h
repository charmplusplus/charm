/*****************************************************************************
 * $Source$
 * $Author$ 
 * $Date$
 * $Revision$
 *****************************************************************************/

/* Sanjay's pooling allocator adapted for cmialloc usage*/


void CmiPoolAllocInit(int numBins);

void * CmiPoolAlloc(unsigned int numBytes);

void * CmiPoolFree(char * p);

void  CmiPoolAllocStats();
