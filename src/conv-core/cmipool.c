
/* adapted by Eric Bohm from Sanjay Kale's pplKalloc */


/* An extremely simple implementation of memory allocation
   that maintains bins for power-of-two sizes.
   May waste about 33%  memory
   Does not do recombining or buddies. 
   Maintains stats that can be turned off for performance, but seems
   plenty fast.
*/


#include "cmipool.h"

CpvStaticDeclare(char **, bins);
CpvStaticDeclare(int *, binLengths);
CpvStaticDeclare(int, maxBin);
CpvStaticDeclare(int, numKallocs);
CpvStaticDeclare(int, numMallocs);
CpvStaticDeclare(int, numOallocs);
CpvStaticDeclare(int, numFrees);
CpvStaticDeclare(int, numOFrees);

/* Each block has a 8 byte header.
   This contains the pointer to the next  block, when 
   the block is in the free list of a particular bin.
   When it is allocated to the app, the header doesn't
  have the pointer, but instead has the bin number to which
  it belongs. I.e. the lg(block size).
*/

/* TODO figure out where we should apply CmiMemLock in here */

/* Once it all works inline it */

extern void *malloc_nomigrate(size_t size);
extern void free_nomigrate(void *mem);

void CmiPoolAllocInit(int numBins)
{
  int i;
  if (CpvInitialized(bins)) return;
  CpvInitialize(char **, bins);
  CpvInitialize(int *, binLengths);
  CpvInitialize(int, maxBin);
  CpvInitialize(int, numKallocs);
  CpvInitialize(int, numMallocs);
  CpvInitialize(int, numOFrees);
  CpvInitialize(int, numFrees);

  CpvAccess(bins) = (char **) malloc_nomigrate(  numBins*sizeof(char *));
  CpvAccess(binLengths) = (int *) malloc_nomigrate(  numBins*sizeof(int));
  CpvAccess(maxBin) = numBins -1;
  for (i=0; i<numBins; i++) CpvAccess(bins)[i] = NULL;
  for (i=0; i<numBins; i++) CpvAccess(binLengths)[i] = 0;

  CpvAccess(numKallocs) =  CpvAccess(numMallocs) =  CpvAccess(numFrees)=CpvAccess(numOFrees) = 0;
}

void * CmiPoolAlloc(unsigned int numBytes)
{
  char *p;
  int bin=0;
  int n=numBytes+CMI_POOL_HEADER_SIZE;
  CmiInt8 *header;
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
	  /* CmiPrintf("p\n"); */
#if CMK_WITH_STATS
	  CpvAccess(numKallocs)++;
#endif
	  /* store some info in the header*/
	  p = CpvAccess(bins)[bin];
	  /*next pointer from the header*/

	  /* this conditional should not be necessary
	     as the header next pointer should contain NULL
	     for us when there is nothing left in the pool */
#if CMK_WITH_STATS
	  if(--CpvAccess(binLengths)[bin])
	      CpvAccess(bins)[bin] = (char *) *((char **)(p -CMI_POOL_HEADER_SIZE)); 
	  else  /* there is no next */
	      CpvAccess(bins)[bin] = NULL;
#else
	  CpvAccess(bins)[bin] = (char *) *((char **)(p -CMI_POOL_HEADER_SIZE)); 
#endif
	}
      else
	{
	  /* CmiPrintf("np %d\n",bin); */
#if CMK_WITH_STATS
	  CpvAccess(numMallocs)++;
#endif
	  /* Round up the allocation to the max for this bin */
	   p =(char *) malloc_nomigrate(1 << bin) + CMI_POOL_HEADER_SIZE;
	}
    }
  else
    {
      /*  CmiPrintf("u b%d v %d\n",bin,CpvAccess(maxBin));  */
      /* just revert to malloc for big things and set bin 0 */
#if CMK_WITH_STATS
	  CpvAccess(numOallocs)++;
#endif
      p = (char *) malloc_nomigrate(numBytes) + CMI_POOL_HEADER_SIZE;
      bin=0; 

    }
  header = (CmiInt8 *) (p-CMI_POOL_HEADER_SIZE);
  CmiAssert(header !=NULL);
  *header = bin; /* stamp the bin number on the header.*/
  return p;
}

void CmiPoolFree(void * p) 
{
  char **header = (char **)( (char*)p - CMI_POOL_HEADER_SIZE);
  int bin = *(CmiInt8 *)header;
  /*  CmiPrintf("f%d\n",bin,CpvAccess(maxBin));  */
  if(bin==0)
    {
#if CMK_WITH_STATS
      CpvAccess(numOFrees)++;
#endif
      free_nomigrate(header);
    }
  else if(bin<CpvAccess(maxBin))
    {
#if CMK_WITH_STATS
      CpvAccess(numFrees)++;
#endif
      /* add to the begining of the list at CpvAccess(bins)[bin]*/
      *header =  CpvAccess(bins)[bin]; 
      CpvAccess(bins)[bin] = p;
#if CMK_WITH_STATS
      CpvAccess(binLengths)[bin]++;
#endif
    }
  else
    CmiAbort("CmiPoolFree: Invalid Bin");
}

void  CmiPoolAllocStats()
{
  int i;
  CmiPrintf("numKallocs: %d\n", CpvAccess(numKallocs));
  CmiPrintf("numMallocs: %d\n", CpvAccess(numMallocs));
  CmiPrintf("numOallocs: %d\n", CpvAccess(numOallocs));
  CmiPrintf("numOFrees: %d\n", CpvAccess(numOFrees));
  CmiPrintf("numFrees: %d\n", CpvAccess(numFrees));
  CmiPrintf("Bin:");
  for (i=0; i<=CpvAccess(maxBin); i++)
    if(CpvAccess(binLengths)[i])
      CmiPrintf("%d\t", i);
  CmiPrintf("\nVal:");
  for (i=0; i<=CpvAccess(maxBin); i++)
    if(CpvAccess(binLengths)[i])
      CmiPrintf("%d\t", CpvAccess(binLengths)[i]);
  CmiPrintf("\n");
}

void CmiPoolPrintList(char *p)
{
  CmiPrintf("Free list is: -----------\n");
  while (p != 0) {
    char ** header = (char **) p-CMI_POOL_HEADER_SIZE;
    CmiPrintf("next ptr is %p. ", p);
    CmiPrintf("header is at: %p, and contains: %p \n", header, *header);
    p = *header;
  }
  CmiPrintf("End of Free list: -----------\n");
 
}


/* theoretically we should have a pool cleanup function in here */
