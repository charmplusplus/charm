#include "cldb.h"
/* for sqrt() */
#include <math.h>

typedef struct loadmsg_s {
  char header[CmiMsgHeaderSizeBytes];
  int pe, load;
#if ! USE_MULTICAST
  short fromindex, toindex;
  struct loadmsg_s  *next;
#endif
} loadmsg;

/* work request message when idle */
typedef struct requestmsg_s {
  char header[CmiMsgHeaderSizeBytes];
  int from_pe;
  int to_rank;
} requestmsg;

typedef struct CldNeighborData
{
  int pe, load;
#if ! USE_MULTICAST
  int index;                 // my index on this neighbor
#endif
} *CldNeighborData;

CpvDeclare(CldNeighborData, neighbors);
CpvDeclare(CmiGroup, neighborGroup);
CpvDeclare(int, numNeighbors);
