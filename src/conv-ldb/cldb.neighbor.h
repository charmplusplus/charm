/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "cldb.h"
/* for sqrt() */
#include <math.h>

typedef struct CldNeighborData
{
  int pe, load;
} *CldNeighborData;

typedef struct loadmsg_s {
  char header[CmiMsgHeaderSizeBytes];
  int pe, load;
} loadmsg;

/* work request message when idle */
typedef struct requestmsg_s {
  char header[CmiMsgHeaderSizeBytes];
  int from_pe;
  int to_rank;
} requestmsg;

CpvDeclare(CldNeighborData, neighbors);
CpvDeclare(CmiGroup, neighborGroup);
CpvDeclare(int, numNeighbors);
