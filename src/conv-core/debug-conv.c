/*
Converse-level debugger support

Collected from convcore.c, conv-ccs.c, register.c by
Orion Sky Lawlor, olawlor@acm.org, 4/10/2001

This code is years old, poorly designed, and almost
certainly won't (and shouldn't) work.  Beware!
 */
#include "converse.h"
#include "conv-trace.h"
#include "queueing.h"
#include "conv-ccs.h"
#include "ccs-server.h"

CpvStaticDeclare(int, freezeModeFlag);
CpvStaticDeclare(int, continueFlag);
CpvStaticDeclare(int, stepFlag);
CpvDeclare(void *, debugQueue);
unsigned int freezeIP;
int freezePort;
char* breakPointHeader;
char* breakPointContents;

extern int ccs_socket_ready;

/***************************************************
  The CCS interface to the debugger
*/
void dummyF()
{
}

#include <string.h>

static void CpdDebugHandler(char *msg)
{
  char *reply, *temp;
  int index;
  unsigned int ip,ignored_port;
  CcsCallerId(&ip,&ignored_port);
  if(CcsIsRemoteRequest()) {
    char name[128];
    sscanf(msg+CmiMsgHeaderSizeBytes, "%s", name);
    reply = NULL;

    if (strcmp(name, "freeze") == 0) {
      CpdFreeze();
      msgListCleanup();
      msgListCache();
      CmiPrintf("freeze received\n");
    }
    else if (strcmp(name, "unfreeze") == 0) {
      CpdUnFreeze();
      msgListCleanup();
      CmiPrintf("unfreeze received\n");
    }
    else if (strcmp(name, "getObjectList") == 0){
      CmiPrintf("getObjectList received\n");
      reply = getObjectList();
      CmiPrintf("list obtained");
      if(reply == NULL){
	CmiPrintf("list empty");
	CcsSendReply(strlen("$") + 1, "$");
      }
      else{
	CmiPrintf("list : %s\n", reply);
	CcsSendReply(strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if(strncmp(name,"getObjectContents",strlen("getObjectContents"))==0){
      CmiPrintf("getObjectContents received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getObjectContents(index);
      CmiPrintf("Object Contents : %s\n", reply);
      CcsSendReply(strlen(reply) + 1, reply);
      free(reply);
    }
    else if (strcmp(name, "getMsgListSched") == 0){
      CmiPrintf("getMsgListSched received\n");
      reply = getMsgListSched();
      if(reply == NULL)
	CcsSendReply(strlen("$") + 1, "$");
      else{
	CcsSendReply(strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if (strcmp(name, "getMsgListFIFO") == 0){
      CmiPrintf("getMsgListFIFO received\n");
      reply = getMsgListFIFO();
      if(reply == NULL)
	CcsSendReply(strlen("$") + 1, "$");
      else{
	CcsSendReply(strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if (strcmp(name, "getMsgListPCQueue") == 0){
      CmiPrintf("getMsgListPCQueue received\n");
      reply = getMsgListPCQueue();
      if(reply == NULL)
	CcsSendReply(strlen("$") + 1, "$");
      else{
	CcsSendReply(strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if (strcmp(name, "getMsgListDebug") == 0){
      CmiPrintf("getMsgListDebug received\n");
      reply = getMsgListDebug();
      if(reply == NULL)
	CcsSendReply(strlen("$") + 1, "$");
      else{
	CcsSendReply(strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if(strncmp(name,"getMsgContentsSched",strlen("getMsgContentsSched"))==0){
      CmiPrintf("getMsgContentsSched received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getMsgContentsSched(index);
      CmiPrintf("Message Contents : %s\n", reply);
      CcsSendReply(strlen(reply) + 1, reply);
      free(reply);
    }
    else if(strncmp(name,"getMsgContentsFIFO",strlen("getMsgContentsFIFO"))==0){
      CmiPrintf("getMsgContentsFIFO received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getMsgContentsFIFO(index);
      CmiPrintf("Message Contents : %s\n", reply);
      CcsSendReply(strlen(reply) + 1, reply);
      free(reply);
    }
    else if (strncmp(name, "getMsgContentsPCQueue", strlen("getMsgContentsPCQueue")) == 0){
      CmiPrintf("getMsgContentsPCQueue received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getMsgContentsPCQueue(index);
      CmiPrintf("Message Contents : %s\n", reply);
      CcsSendReply(strlen(reply) + 1, reply);
      free(reply);
    }
    else if (strncmp(name, "getMsgContentsDebug", strlen("getMsgContentsDebug")) == 0){
      CmiPrintf("getMsgContentsDebug received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getMsgContentsDebug(index);
      CmiPrintf("Message Contents : %s\n", reply);
      CcsSendReply(strlen(reply) + 1, reply);
      free(reply);
    } 
    else if (strncmp(name, "step", strlen("step")) == 0){
      CmiPrintf("step received\n");
      CpvAccess(stepFlag) = 1;
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &freezePort);
      freezeIP = ip;
      CpdUnFreeze();
    }
    else if (strncmp(name, "continue", strlen("continue")) == 0){
      CmiPrintf("continue received\n");
      CpvAccess(continueFlag) = 1;
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &freezePort);
      freezeIP = ip;
      CpdUnFreeze();
    }
    else if (strcmp(name, "getBreakStepContents") == 0){
      CmiPrintf("getBreakStepContents received\n");
      if(breakPointHeader == 0){
	CcsSendReply(strlen("$") + 1, "$");
      }
      else{
	reply = (char *)malloc(strlen(breakPointHeader) + strlen(breakPointContents) + 1);
	strcpy(reply, breakPointHeader);
	strcat(reply, "@");
	strcat(reply, breakPointContents);
	CcsSendReply(strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if (strcmp(name, "getSymbolTableInfo") == 0){
      CmiPrintf("getSymbolTableInfo received");
      reply = getSymbolTableInfo();
      CcsSendReply(strlen(reply) + 1, reply);
      reply = getBreakPoints();
      CcsSendReply(strlen(reply) + 1, reply);
      free(reply);
    }
    else if (strncmp(name, "setBreakPoint", strlen("setBreakPoint")) == 0){
      CmiPrintf("setBreakPoint received\n");
      temp = strstr(name, "#");
      temp++;
      setBreakPoints(temp);
    }
    else if (strncmp(name, "gdbRequest", strlen("gdbRequest")) == 0){
      CmiPrintf("gdbRequest received\n");
      dummyF();
    }

    else if (strcmp(name, "quit") == 0){
      CpdUnFreeze();
      CsdExitScheduler();
    }
    else{
      CmiPrintf("incorrect command:%s received,len=%ld\n",name,strlen(name));
    }
  }
}

/*
 Start the freeze-- will not return until unfrozen
 via a CCS request.
 */
void CpdFreeze(void)
{
  CpvAccess(freezeModeFlag) = 1;
  CpdFreezeModeScheduler();
}  

void CpdUnFreeze(void)
{
  CpvAccess(freezeModeFlag) = 0;
}


/****************************************************
  A special version of handle message-- checks if
  we're reaching a breakpoint or stepping and handles
  the case specially.

  The right way to do this is probably to reset the call 
  handler, and let the usual CmiHandleMessage call our
  code on a breakpoint.
*/
void Cpd_CmiHandleMessage(void *msg)
{
  extern unsigned int freezeIP;
  extern int freezePort;
  extern char* breakPointHeader;
  extern char* breakPointContents;

  char *freezeReply;
  int fd;

  if(CpvAccess(continueFlag) && (isBreakPoint((char *)msg))) {

    if(breakPointHeader != 0){
      free(breakPointHeader);
      breakPointHeader = 0;
    }
    if(breakPointContents != 0){
      free(breakPointContents);
      breakPointContents = 0;
    }
    
    breakPointHeader = genericViewMsgFunction((char *)msg, 0);
    breakPointContents = genericViewMsgFunction((char *)msg, 1);

    CmiPrintf("BREAKPOINT REACHED :\n");
    CmiPrintf("Header : %s\nContents : %s\n", breakPointHeader, breakPointContents);

    /* Freeze and send a message back */
    freezeReply = (char *)malloc(strlen("freezing@")+strlen(breakPointHeader)+1);
    _MEMCHECK(freezeReply);
    sprintf(freezeReply, "freezing@%s", breakPointHeader);
    fd = skt_connect(freezeIP, freezePort, 120);
    if(fd > 0){
      skt_sendN(fd, freezeReply, strlen(freezeReply) + 1);
      skt_close(fd);
    } else {
      CmiPrintf("unable to connect");
    }
    free(freezeReply);
    CpvAccess(continueFlag) = 0;
    CpdFreeze();
  } else if(CpvAccess(stepFlag) && (isEntryPoint((char *)msg))){
    if(breakPointHeader != 0){
      free(breakPointHeader);
      breakPointHeader = 0;
    }
    if(breakPointContents != 0){
      free(breakPointContents);
      breakPointContents = 0;
    }

    breakPointHeader = genericViewMsgFunction((char *)msg, 0);
    breakPointContents = genericViewMsgFunction((char *)msg, 1);

    CmiPrintf("STEP POINT REACHED :\n");
    CmiPrintf("Header:%s\nContents:%s\n",breakPointHeader,breakPointContents);

    /* Freeze and send a message back */
    freezeReply = (char *)malloc(strlen("freezing@")+strlen(breakPointHeader)+1);
    _MEMCHECK(freezeReply);
    sprintf(freezeReply, "freezing@%s", breakPointHeader);
    fd = skt_connect(freezeIP, freezePort, 120);
    if(fd > 0){
      skt_sendN(fd, freezeReply, strlen(freezeReply) + 1);
      skt_close(fd);
    } else {
      CmiPrintf("unable to connect");
    }
    free(freezeReply);
    CpvAccess(stepFlag) = 0;
    CpdFreeze();
  }
  (CmiGetHandlerFunction(msg))(msg);
}

/* Special scheduler-type loop only executed while in
freeze mode-- only executes CCS requests.
*/
void CpdFreezeModeScheduler(void)
{
    void *msg;
    /* While frozen, queue up messages */
    while (CpvAccess(freezeModeFlag)) {
#if NODE_0_IS_CONVHOST
	if (ccs_socket_ready) CHostProcess();
#endif
	msg = CmiGetNonLocal();
	if(_ccsHandlerIdx == CmiGetHandler(msg))
	/*A CCS request-- handle it immediately*/
	    CmiHandleMessage(msg);
	else
	/*An ordinary message-- queue it up*/
	    CdsFifo_Enqueue(CpvAccess(debugQueue), msg);
    }
    /* Before leaving freeze mode, process the queued messages */
    while (!CpvAccess(freezeModeFlag) && !CdsFifo_Empty(CpvAccess(debugQueue)))
    {
	char *queuedMsg = (char *)CdsFifo_Dequeue(CpvAccess(debugQueue));
        CmiHandleMessage(queuedMsg);
    }
}

void CpdInit(void)
{
  CpvInitialize(int, freezeModeFlag);
  CpvAccess(freezeModeFlag) = 0;

  CpvInitialize(int, continueFlag);
  CpvInitialize(int, stepFlag);
  CpvAccess(continueFlag) = 0;
  CpvAccess(stepFlag) = 0;

  CpvInitialize(void *, debugQueue);
  CpvAccess(debugQueue) = CdsFifo_Create();
    
  CpdInitializeObjectTable();
  CpdInitializeHandlerArray();
  CpdInitializeBreakPoints();

  CcsRegisterHandler("DebugHandler", CpdDebugHandler);

  /* To allow start in freeze state: */
  msgListCleanup();
  msgListCache();  
}

















