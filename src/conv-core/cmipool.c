/*****************************************************************************
 * $Source$
 * $Author$ 
 * $Date$
 * $Revision$
 *****************************************************************************/

/* adapted from Sanjay Kale's pplKalloc by Eric Bohm */


/* An extremely simple implementation of memory allocation
   that maintains bins for power-of-two sizes.
   May waste about 33%  memory
   Does not do recombining or buddies. 
   Maintains stats that can be turned off for performance, but seems
   plenty fast.
*/

#define CMI_POOL_HEADER_SIZE 8

#include "converse.h"
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


void printList(char *p) 
{
  printf("Free list is: -----------\n");
  while (p != 0) {
    printf("next ptr is %d. ", (int) p);
    char ** header = p-CMI_POOL_HEADER_SIZE;
    printf("header is at: %d, and contains: %d \n", (int) header, (int) (*header));
    p = *header;
  }
  printf("End of Free list: -----------\n");

}

inline void CmiPoolAllocInit(int numBins)
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

  CpvAccess(numKallocs) = CpvAccess(numMallocs) = CpvAccess(numFrees) = 0;
}

inline void * CmiPoolAlloc(unsigned int numBytes)
{
  char *p, *next;
  int bin,n;
  bin = 0;

  /* this could be speeded up.. I think*/
  numBytes += CMI_POOL_HEADER_SIZE; 
  n = numBytes;
  /* get 8 more bytes, so I can store a header to the left*/
  while (n !=0) /* find the bin*/
    {     
      n = n >> 1;
      bin++;
    }
  /* even 0 size messages go in bin 1 leaving 0 bin for too bigs */
  if(bin<CpvAccess(maxBin) && CpvAccess(bins)[bin] != NULL) /* message not too big */
    {
      CpvAccess(numKallocs)++;
      /* store some info in the header*/
      p = CpvAccess(bins)[bin];
      next = (char *) *((char **)(p -CMI_POOL_HEADER_SIZE)); /*next pointer from the header*/
      CpvAccess(bins)[bin] = next;
      CpvAccess(binLengths)[bin]--;
    }
  else
    {
      /* just revert to the standard malloc for big things */
      char * header = malloc_nomigrate(numBytes);
      p = header + CMI_POOL_HEADER_SIZE;
      bin=0;
    }
  int *header =  p-CMI_POOL_HEADER_SIZE;
  *header = bin; /* stamp the bin number on the header.*/
  return p;
}


inline void * CmiPoolFree(char * p) 
{
  int bin;
  char ** header;
  if(bin==0)
    {
      free_nomigrate(p-CMI_POOL_HEADER_SIZE);
    }
  else
    {
      CpvAccess(numFrees)++;
      header = p - CMI_POOL_HEADER_SIZE;
      bin = (int) *header;
  
      /* add to the begining of the list at CpvAccess(bins)[bin]*/
      *header =  CpvAccess(bins)[bin]; 
      CpvAccess(bins)[bin] = p;
      CpvAccess(binLengths)[bin]++;
    }
}

inline void  CmiPoolAllocStats()
{
  int i;
  printf("numKallocs: %d\n", CpvAccess(numKallocs));
  printf("numMallocs: %d\n", CpvAccess(numMallocs));
  printf("numFrees: %d\n", CpvAccess(numFrees));
  /*  for (i=0; i<=CpvAccess(maxbin); i++) */
  /*   { printf("binLength[%d] : %d\n", i, CpvAccess(binLengths)[i]);}*/
}

/* theoretically we should put a pool cleanup function in here */
