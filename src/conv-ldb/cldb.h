#include <stdio.h>
#include "converse.h"

#define MAXMSGBFRSIZE 100000

CpvExtern(int, CldHandlerIndex);
CpvExtern(int, CldBalanceHandlerIndex);

CpvExtern(int, CldRelocatedMessages);
CpvExtern(int, CldLoadBalanceMessages);
CpvExtern(int, CldMessageChunks);
CpvExtern(int, CldLoadNotify);

void CldMultipleSend(int pe, int numToSend);
void CldSetPEBitVector(unsigned int);
