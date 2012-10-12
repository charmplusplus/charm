#include <stdio.h>
#include "converse.h"

#define MAXMSGBFRSIZE 100000

CpvExtern(int, CldHandlerIndex);
CpvExtern(int, CldNodeHandlerIndex);
CpvExtern(int, CldBalanceHandlerIndex);

CpvExtern(int, CldRelocatedMessages);
CpvExtern(int, CldLoadBalanceMessages);
CpvExtern(int, CldMessageChunks);
CpvExtern(int, CldLoadNotify);

CpvExtern(CmiNodeLock, cldLock);

void CldMultipleSend(int pe, int numToSend, int rank, int immed);
void CldSimpleMultipleSend(int pe, int numToSend, int rank);
void CldSetPEBitVector(const char *);

int  CldLoad(void);
int  CldLoadRank(int rank);
int  CldCountTokens(void);
int  CldCountTokensRank(int rank);
void CldPutToken(char *);
void CldPutTokenPrio(char *);
void CldRestoreHandler(char *);
void CldSwitchHandler(char *, int);
void CldModuleGeneralInit();
int  CldPresentPE(int pe);
void seedBalancerExit();
