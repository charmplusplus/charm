/*****************************************************************************
 * $Source$
 * $Author$ 
 * $Date$
 * $Revision$
 *****************************************************************************/

/* adapted by Eric Bohm from Sanjay Kale's pplKalloc */


/* An extremely simple implementation of memory allocation
   that maintains bins for power-of-two sizes.
   May waste about 33%  memory
   Does not do recombining or buddies. 
   Maintains stats that can be turned off for performance, but seems
   plenty fast.
*/


#include "cmipool.h"

CpvDeclare(char **, bins);
CpvDeclare(int *, binLengths);
CpvDeclare(int, maxBin);
CpvDeclare(int, numKallocs);
CpvDeclare(int, numMallocs);
CpvDeclare(int, numFrees);

/* Each block has a 8 byte header.
   This contains the pointer to the next  block, when 
   the block is in the free list of a particular bin.
   When it is allocated to the app, the header doesn't
  have the pointer, but instead has the bin number to which
  it belongs. I.e. the lg(block size).
*/

/* TODO wrap ifndef CMK_OPTIMIZE around the stats collection bits */

/* TODO figure out where we should apply CmiMemLock in here */

/* Once it all works inline it */

extern void *malloc_nomigrate(size_t size);
extern void free_nomigrate(void *mem);

void CmiPoolAllocInit(int numBins)
{
  int i;
  CpvInitialize(char **, bins);
  CpvInitialize(int *, binLengths);
  CpvInitialize(int, maxBin);
  CpvInitialize(int, numKallocs);
  CpvInitialize(int, numMallocs);
  CpvInitialize(int, numFrees);

  CpvAccess(bins) = (char **) malloc_nomigrate(  numBins*sizeof(char *));
  CpvAccess(binLengths) = (int *) malloc_nomigrate(  numBins*sizeof(int));
  CpvAccess(maxBin) = numBins -1;
  for (i=0; i<numBins; i++) CpvAccess(bins)[i] = NULL;
  for (i=0; i<numBins; i++) CpvAccess(binLengths)[i] = 0;

    CpvAccess(numKallocs) =  CpvAccess(numMallocs) =  CpvAccess(numFrees) = 0;
}

void * CmiPoolAlloc(unsigned int numBytes)
{
  char *p, *next;
  int bin=0;
  int n=numBytes+CMI_POOL_HEADER_SIZE;
  int *header;
  /* get 8 more bytes, so I can store a header to the left*/
  numBytes = n;
  while (n !=0) /* find the bin*/
    {     
      n = n >> 1;
      bin++;
    }
  /* even 0 size messages go in bin 1 leaving 0 bin for oversized */
  if(bin<CpvAccess(maxBin))
    {
      CmiAssert(bin>0);
      if(CpvAccess(bins)[bin] != NULL) 
	{
	  /*	  CmiPrintf("p\n");*/
	  CpvAccess(numKallocs)++;
	  /* store some info in the header*/
	  p = CpvAccess(bins)[bin];
	  /*next pointer from the header*/

	  /* this conditional should not be necessary
	     as the header next pointer should contain NULL
	     for us when there is nothing left in the pool */
	  if(--CpvAccess(binLengths)[bin])
	      CpvAccess(bins)[bin] = (char *) *((char **)(p -CMI_POOL_HEADER_SIZE)); 
	  else  /* there is no next */
	      CpvAccess(bins)[bin] = NULL;
	}
      else
	{
	  /*  CmiPrintf("np %d\n",bin);*/

	  CpvAccess(numMallocs)++;
	  /* Round up the allocation to the max for this bin */
	   p =(char *) malloc_nomigrate(1 << bin) + CMI_POOL_HEADER_SIZE;
	}
    }
  else
    {
      /*  CmiPrintf("u b%d v %d\n",bin,CpvAccess(maxBin));  */
      /* just revert to malloc for big things and set bin 0 */
      p = (char *) malloc_nomigrate(numBytes) + CMI_POOL_HEADER_SIZE;
      bin=0; 
    }
  header = (int *) (p-CMI_POOL_HEADER_SIZE);
  *header = bin; /* stamp the bin number on the header.*/
  return p;
}


 void * CmiPoolFree(char * p) 
{
  char **header = (char **)( p - CMI_POOL_HEADER_SIZE);
  int bin = (int) *header;
  /*  CmiPrintf("f%d\n",bin,CpvAccess(maxBin));  */
  if(bin==0)
    {
      free_nomigrate(header);
    }
  else
    {
      CpvAccess(numFrees)++;
      /* add to the begining of the list at CpvAccess(bins)[bin]*/
      *header =  CpvAccess(bins)[bin]; 
      CpvAccess(bins)[bin] = p;
      CpvAccess(binLengths)[bin]++;
    }
}

 void  CmiPoolAllocStats()
{
  int i;
  CmiPrintf("numKallocs: %d\n", CpvAccess(numKallocs));
  CmiPrintf("numFrees: %d\n", CpvAccess(numFrees));
  /*  for (i=0; i<=CpvAccess(maxbin); i++) */
  /*   { CmiPrintf("binLength[%d] : %d\n", i, CpvAccess(binLengths)[i]);}*/
}

/* theoretically we should have a pool cleanup function in here */
