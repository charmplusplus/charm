#include <stdio.h>
#include "converse.h"

#define MAXMSGBFRSIZE 100000

CpvDeclare(int, CldHandlerIndex);
CpvDeclare(int, CldBalanceHandlerIndex);

CpvDeclare(int, CldRelocatedMessages);
CpvDeclare(int, CldLoadBalanceMessages);
CpvDeclare(int, CldMessageChunks);

void CldMultipleSend(int pe, int numToSend);
