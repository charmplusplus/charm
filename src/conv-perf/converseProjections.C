
#include <stdio.h>
#include <stdlib.h>
#include "converse.h"
#include "converseEvents.h"
#include "traceCoreCommon.h"
#include "converseProjections.h"

extern "C" void converse_msgSent(int destPE, int size)
{
	int iData[2];
	iData[0] = destPE;
	iData[1] = size;
	LogEvent1(_CONVERSE_LANG_ID, _E_MSG_SENT, 2, iData); 
}

//TODO
extern "C" void converse_msgQueued();
//TODO
extern "C" void converse_msgRecvMC();
//TODO
extern "C" void converse_msgRecvSC();

extern "C" void converse_handlerBegin(int handlerIdx)
{		
	int iData[2];
	iData[0] = handlerIdx;
	iData[1] = CmiMyPe();
	LogEvent1(_CONVERSE_LANG_ID, _E_HANDLER_BEGIN, 2, iData); 
}

extern "C" void converse_handlerEnd(int handlerIdx)
{	
	int iData[2];
	iData[0] = handlerIdx;
	iData[1] = CmiMyPe();
	LogEvent1(_CONVERSE_LANG_ID, _E_HANDLER_END, 2, iData); 
}

