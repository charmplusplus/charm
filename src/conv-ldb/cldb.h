#include <stdio.h>
#include "converse.h"

#define MAXMSGBFRSIZE 100000

CpvExtern(int, CldHandlerIndex);
CpvExtern(int, CldBalanceHandlerIndex);

CpvExtern(int, CldRelocatedMessages);
CpvExtern(int, CldLoadBalanceMessages);
CpvExtern(int, CldMessageChunks);
CpvExtern(int, CldLoadNotify);

CpvExtern(CmiNodeLock, cldLock);

void CldMultipleSend(int pe, int numToSend, int rank);
void CldSetPEBitVector(const char *);

int  CldLoad(void);
int  CldCountTokens(void);
void CldPutToken(char *);
void CldRestoreHandler(char *);
void CldSwitchHandler(char *, int);
void CldModuleGeneralInit();
int  CldPresentPE(int pe);
