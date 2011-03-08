/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "cldb.h"
/* for sqrt() */
#include <math.h>

typedef struct loadmsg_s {
  char header[CmiMsgHeaderSizeBytes];
  int pe, load;
#if ! USE_MULTICAST
  int fromindex, toindex;
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
  loadmsg *msg;
  int index;                 // my index on this neighbor
#endif
} *CldNeighborData;

CpvDeclare(CldNeighborData, neighbors);
CpvDeclare(CmiGroup, neighborGroup);
CpvDeclare(int, numNeighbors);
