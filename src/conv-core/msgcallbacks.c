/***********************************************************************************************************
Callbacks for converse messages. Sameer Paranjpye 2/24/2000.
This module defines an interface for registering callbacks for converse messages. 
This is meant to be a machine level interface through which additional message processing capabilities can 
be added to converse. The idea is that if a message needs additional processing when it is sent or received
such as encryption/decryption, maintaining quiescence counters etc., then a callback is registered that provides 
this processing capability. Its only current application is keeping quiescence counts.

For now I'm assuming that callbacks are only registered in pairs, for every send callback there is a receice 
callback. But this can be easily changed by making CMsgRegisterCallback non-static.

CMsgCallbacksInit - Initializes the callback mechanism.

CMsgRegisterCallbackPair - Registers a message callback pair, one at the send side one at the recv side.

CMsgInvokeCallbacks - Invokes registered callbacks

************************************************************************************************************/


#include "converse.h"
#define CALLBACKSETSIZE     5
#define MAXCALLBACKSETS     5 
#define SENDCALLBACK        0
#define RECVCALLBACK        1

typedef struct CMsgCallback {
  CMsgProcFn fn;
} CMSGCALLBACK;

struct CMsgCallbackQ {
	CMSGCALLBACK **cbQ;
	int          size;
};

struct CMsgCallbackQ  cMsgSendCbQ;
struct CMsgCallbackQ  cMsgRecvCbQ;   

void CMsgCallbacksInit(void)
{
	cMsgSendCbQ.cbQ  = (CMSGCALLBACK **) malloc(MAXCALLBACKSETS* 
												sizeof(CMSGCALLBACK*));
	cMsgSendCbQ.size = 0;
	cMsgRecvCbQ.cbQ  = (CMSGCALLBACK **) malloc(MAXCALLBACKSETS* 
												sizeof(CMSGCALLBACK*));
	cMsgRecvCbQ.size = 0;
}

static int CMsgRegisterCallback(CMsgProcFn fnp, int type)
{
	int setNum, setIdx;
	struct CMsgCallbackQ* Q;
	
	if (type < 2)
		Q = (type)? (&cMsgRecvCbQ) : (&cMsgSendCbQ);
	else 
	{
		CmiPrintf("Unknown callback type, cannot register");
		return -1;
	}

	setNum = Q->size/CALLBACKSETSIZE;
	setIdx = Q->size%CALLBACKSETSIZE;
	
	if (setNum >= MAXCALLBACKSETS) {
		CmiPrintf("Too many message callbacks, cannot register\n");
		return -1;
	}
	
	if (!setIdx)
		Q->cbQ[setNum] = (CMSGCALLBACK *) malloc(CALLBACKSETSIZE*sizeof(CMSGCALLBACK));
	
	(Q->cbQ[setNum])[setIdx].fn = fnp;
	Q->size++;

	return 0;
}

int CMsgRegisterCallbackPair(CMsgProcFn sendFn, CMsgProcFn recvFn)
{
	if(CMsgRegisterCallback(sendFn, SENDCALLBACK) < 0) 
		return -1;
	if(CMsgRegisterCallback(recvFn, RECVCALLBACK) < 0) 
		return -1;
	return 0;
}

void CMsgInvokeCallbacks(int type, char *msg)
{
	int  i;
	struct CMsgCallbackQ* Q;

	if (type < 2)
		Q = (type)? (&cMsgRecvCbQ) : (&cMsgSendCbQ);
	else 
	{
		CmiPrintf("Unknown callback type, cannot invoke");
		return;
	}

	for(i=0; i < Q->size; i++)
		((Q->cbQ[i/CALLBACKSETSIZE])[i%CALLBACKSETSIZE].fn)(msg);
}
