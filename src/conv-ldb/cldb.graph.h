#include <stdio.h>
#include "converse.h"

#define MAXMSGBFRSIZE 100000

typedef struct CldNeighborData
{
  int pe, load;
} *CldNeighborData;

typedef struct loadmsg_s {
  char header[CmiMsgHeaderSizeBytes];
  int pe, load;
} loadmsg;

typedef struct requestmsg_s {
  char header[CmiMsgHeaderSizeBytes];
  int pe;
} requestmsg;

CpvDeclare(CldNeighborData, neighbors);
CpvDeclare(CmiGroup, neighborGroup);
CpvDeclare(int, numNeighbors);
CpvDeclare(int, CldHandlerIndex);
CpvDeclare(int, CldBalanceHandlerIndex);
CpvDeclare(int, CldRelocatedMessages);
CpvDeclare(int, CldLoadBalanceMessages);
CpvDeclare(int, CldMessageChunks);

