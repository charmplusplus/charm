
#include <stdio.h>
#include <stdlib.h>
#include "converse.h"
#include "converseEvents.h"
#include "traceCoreCommon.h"
#include "converseProjections.h"

extern "C" void msgSent(int destPE, int size)
{
	int* iData = (int*)malloc(sizeof(int)*2); 
	iData[0] = destPE;
	iData[1] = size;
	LogEvent1(_CONVERSE_LANG_ID, _E_MSG_SENT, 2, iData); 
}

//TODO
extern "C" void msgQueued();
//TODO
extern "C" void msgRecvMC();
//TODO
extern "C" void msgRecvSC();

extern "C" void handlerBegin(int handlerIdx)
{	
	int* iData = (int*)malloc(sizeof(int)*2); 
	iData[0] = handlerIdx;
	iData[1] = CmiMyPe();
	LogEvent1(_CONVERSE_LANG_ID, _E_HANDLER_BEGIN, 2, iData); 
}

extern "C" void handlerEnd(int handlerIdx)
{	
	int* iData = (int*)malloc(sizeof(int)*2); 
	iData[0] = handlerIdx;
	iData[1] = CmiMyPe();
	LogEvent1(_CONVERSE_LANG_ID, _E_HANDLER_END, 2, iData); 
}

extern "C" void procIdle()
{	
	int* iData = (int*)malloc(sizeof(int)); 
	iData[0] = CmiMyPe();
	LogEvent1(_CONVERSE_LANG_ID, _E_PROC_IDLE, 1, iData); 
}

extern "C" void procBusy()
{	
	int* iData = (int*)malloc(sizeof(int)); 
	iData[0] = CmiMyPe();
	LogEvent1(_CONVERSE_LANG_ID, _E_PROC_BUSY, 1, iData); 
}
