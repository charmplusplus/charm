/*
  Converse performance counter API (high level)
  currently support origin2000 and PAPI
  PAPI implementation uses high level functions for single thread applications.

  To instrument a code block:

   CmiStartCounters(...)
   code
   CmiStopCounters(...)

   Gengbin Zheng, gzheng@uiuc.edu, 2004/5/10 

TODO:
   implement a low-level interface
*/

#include "converse.h"

#ifdef CMK_ORIGIN2000
int start_counters(int e0, int e1);
int read_counters(int e0, long long *c0, int e1, long long *c1);
#elif CMK_HAS_COUNTER_PAPI
#include <papi.h>
#endif

void CmiInitCounters()
{
#if CMK_HAS_COUNTER_PAPI
  int retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT && retval > 0) { 
    printf("PAPI library version mismatch!\n"); 
  }
#endif
}

void CmiStartCounters(int events[], int numEvents)
{
#ifdef CMK_ORIGIN2000
  CmiAssert(numEvents >= 2);
  if (start_counters(events[0], events[1]) <0) {
      perror("start_counters");;
  }
#elif CMK_HAS_COUNTER_PAPI
  if (PAPI_start_counters(events, numEvents) != PAPI_OK) {
      CmiAbort("Failed to read PAPI counters!!\n");
  }
#else
  /* CmiAbort("CmiStartCounters not implemented!"); */
  /* nop */
#endif
}

/* read */
void CmiStopCounters(int events[], CMK_TYPEDEF_INT8 values[], int numEvents)
{
#ifdef CMK_ORIGIN2000
  CmiAssert(numEvents >= 2);
  if (read_counters(events[0], &values[0], events[1], &values[1]) < 0) perror("read_counters");
#elif CMK_HAS_COUNTER_PAPI
  if (PAPI_stop_counters(values, 2) != PAPI_OK) {
    CmiAbort("Failed to read PAPI counters!\n");
  }
#else
  /* CmiAbort("CmiReadCounters not implemented!"); */
  int i;
  for (i=0; i<numEvents; i++) values[i] = 0;
#endif
}


