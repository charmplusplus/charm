/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#ifndef  WIN32
#include <sys/time.h>
#endif
#include <string.h>

#include "converse.h"
#include "conv-ccs.h"
#include "ccs-server.h"
#include "sockRoutines.h"

#if CMK_CCS_AVAILABLE

/**********************************************
Builtin CCS request handlers:
  "ccs_getinfo"-- no data
    Return the number of parallel nodes, and
      the number of processors per node as an array
      of 4-byte big-endian ints.

  "ccs_killport"-- one 4-byte big-endian port number
    Register a "client kill port".  When this program exits,
    it will connect to this TCP port and write "die\n\0" it.
*/

static void ccs_getinfo(char *msg)
{
  int nNode=CmiNumNodes();
  int len=(1+nNode)*sizeof(ChMessageInt_t);
  ChMessageInt_t *table=(ChMessageInt_t *)malloc(len);
  int n;
  table[0]=ChMessageInt_new(nNode);
  for (n=0;n<nNode;n++)
    table[1+n]=ChMessageInt_new(CmiNodeSize(n));
  CcsSendReply(len,(const char *)table);
  free(table);
  CmiGrabBuffer((void **)&msg);CmiFree(msg);
}

typedef struct killPortStruct{
  int ip,port;
  struct killPortStruct *next;
} killPortStruct;
/*Only 1 kill list per node-- no Cpv needed*/
static killPortStruct *killList=NULL;

static void ccs_killport(char *msg)
{
  killPortStruct *oldList=killList;
  int port=ChMessageInt(*(ChMessageInt_t *)(msg+CmiMsgHeaderSizeBytes));
  unsigned int ip,connPort;
  CcsCallerId(&ip,&connPort);
  killList=(killPortStruct *)malloc(sizeof(killPortStruct));
  killList->ip=ip;
  killList->port=port;
  killList->next=oldList;
  CmiGrabBuffer((void **)&msg);CmiFree(msg);
}
/*Send any registered clients kill messages before we exit*/
static int noMoreErrors(int c,const char *m) {return -1;}
void CcsImpl_kill(void)
{
  skt_set_abort(noMoreErrors);
  while (killList!=NULL)
  {
    SOCKET fd=skt_connect(killList->ip,killList->port,20);
    if (fd!=INVALID_SOCKET) {
      skt_sendN(fd,"die\n",strlen("die\n")+1);
      skt_close(fd);
    }
    killList=killList->next;
  }
}


/* move */

#if CMK_DEBUG_MODE
CpvDeclare(int, freezeModeFlag);
CpvDeclare(int, continueFlag);
CpvDeclare(int, stepFlag);
CpvDeclare(void *, debugQueue);
unsigned int freezeIP;
int freezePort;
char* breakPointHeader;
char* breakPointContents;

void dummyF()
{
}

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
}  

void CpdFreeze(void)
{
  CpvAccess(freezeModeFlag) = 1;
}  

void CpdUnFreeze(void)
{
  CpvAccess(freezeModeFlag) = 0;
}  

#endif


#if CMK_WEB_MODE
/******************************************************
Web performance monitoring interface:
	Clients will register for performance data with 
processor 0.  Every WEB_INTERVAL (few seconds), this code
calls all registered web performance functions on all processors.  
The resulting integers are sent back to the client.  The current
reply format is ASCII and rather nasty. 

	Like the debugger support above, there's no good reason 
for this to be defined here.

The actual call sequence is:
CCS Client->CWebHandler->...  (processor 0)
  ...->CWeb_Collect->... (all processors)
...->CWeb_Reduce->CWeb_Deliver (processor 0 again)
*/


#define WEB_INTERVAL 2000 /*Time, in milliseconds, between performance snapshots*/
#define MAXFNS 20 /*Largest number of performance functions to expect*/

typedef struct {
	char hdr[CmiMsgHeaderSizeBytes];
	int fromPE;/*Source processor*/
	int perfData[MAXFNS];/*Performance numbers*/
} CWeb_CollectedData;

/*This needs to be made into a list of registered clients*/
static int hasApplet=0;
static CcsDelayedReply appletReply;

typedef int (*CWebFunction)(void);
static CWebFunction CWebPerformanceFunctionArray[MAXFNS];
static int CWebNoOfFns;
static int CWeb_ReduceIndex;
static int CWeb_CollectIndex;

/*Deliver the reduced web performance data to the waiting client:
*/
static int collectedCount;
static CWeb_CollectedData **collectedValues;

static void CWeb_Deliver(void)
{
  int i,j;

  if (hasApplet) {
    /*Send the performance data off to the applet*/
    char *reply=(char *)malloc(6+14*CmiNumPes()*CWebNoOfFns);
    sprintf(reply,"perf");
  
    for(i=0; i<CmiNumPes(); i++){
      for (j=0;j<CWebNoOfFns;j++)
      {
        char buf[20];
        sprintf(buf," %d",collectedValues[i]->perfData[j]);
        strcat(reply,buf);
      }
    }
    CcsSendDelayedReply(appletReply,strlen(reply) + 1, reply);
    free(reply);
    hasApplet=0;
  }
  
  /* Free saved performance data */
  for(i = 0; i < CmiNumPes(); i++){
    CmiFree(collectedValues[i]);
    collectedValues[i] = 0;
  }
  collectedCount = 0;
}

/*On PE 0, this handler accumulates all the performace data
*/
static void CWeb_Reduce(void *msg){
  CWeb_CollectedData *cur,*prev;
  int src;

  if(CmiMyPe() != 0){
    CmiAbort("CWeb performance data sent to wrong processor...\n");
  }
  CmiGrabBuffer((void **)&msg);
  cur=(CWeb_CollectedData *)msg;
  src=cur->fromPE;
  prev = collectedValues[src]; /* Previous value, ideally 0 */
  collectedValues[src] = cur;
  if(prev == 0) collectedCount++;
  else CmiFree(prev); /*<- caused by out-of-order perf. data delivery*/

  if(collectedCount == CmiNumPes()){
    CWeb_Deliver();
  }
}

/*On each PE, this handler collects the performance data
and sends it to PE 0.
*/
static void CWeb_Collect(void *dummy)
{
  CWeb_CollectedData *msg;
  int i;

  msg = (CWeb_CollectedData *)CmiAlloc(sizeof(CWeb_CollectedData));
  msg->fromPE = CmiMyPe();
  
  /* Evaluate each performance function*/
  for(i = 0; i < CWebNoOfFns; i++)
    msg->perfData[i] = CWebPerformanceFunctionArray[i] ();

  /* Send result off to node 0 */  
  CmiSetHandler(msg, CWeb_ReduceIndex);
  CmiSyncSendAndFree(0, sizeof(CWeb_CollectedData), msg);

  /* Re-call this function after a delay */
  CcdCallFnAfter(CWeb_Collect, 0, WEB_INTERVAL);
}

void CWebPerformanceRegisterFunction(CWebFunction fn)
{
  CWebPerformanceFunctionArray[CWebNoOfFns] = fn;
  CWebNoOfFns++;
}

/*This is called on PE 0 by clients that wish
to receive performance data.
*/
static void CWebHandler(char *ignoredMsg){
  if(CcsIsRemoteRequest()) {
    static int startedCollection=0;
    
    hasApplet=1;
    appletReply=CcsDelayReply();
    
    if(startedCollection == 0){
      int i;
      startedCollection=1;
      collectedCount=0;
      collectedValues = (CWeb_CollectedData **)malloc(sizeof(void *) * CmiNumPes());
      for(i = 0; i < CmiNumPes(); i++)
        collectedValues[i] = 0;
      
      /*Start collecting data on each processor*/
      for(i = 0; i < CmiNumPes(); i++){
        char *msg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes);
        CmiSetHandler(msg, CWeb_CollectIndex);
        CmiSyncSendAndFree(i, CmiMsgHeaderSizeBytes,msg);
      }
    }
  }
}

/** This "usage" section keeps track of percent of wall clock time
spent actually processing messages on each processor.   
It's a simple performance measure collected by the CWeb framework.

It really aught to be integrated with the tracemode facility.
**/

CpvStaticDeclare(double, startTime);
CpvStaticDeclare(double, beginTime);
CpvStaticDeclare(double, usedTime);
CpvStaticDeclare(int, PROCESSING);

/* Call this when the program is started
 -> Whenever traceModuleInit would be called
 -> -> see conv-core/convcore.c
*/
void initUsage()
{
   CpvInitialize(double, startTime);
   CpvInitialize(double, beginTime);
   CpvInitialize(double, usedTime);
   CpvInitialize(int, PROCESSING);
   CpvAccess(beginTime)  = CmiWallTimer();
   CpvAccess(usedTime)   = 0.;
   CpvAccess(PROCESSING) = 0;
}

/* Call this when a BEGIN_PROCESSING event occurs
 -> Whenever a trace_begin_execute or trace_begin_charminit
    would be called
 -> -> See ck-core/init.c,main.c and conv-core/convcore.c
*/
void usageStart()
{
   if(CpvAccess(PROCESSING)) return;

   CpvAccess(startTime)  = CmiWallTimer();
   CpvAccess(PROCESSING) = 1;
}

/* Call this when an END_PROCESSING event occurs
 -> Whenever a trace_end_execute or trace_end_charminit
    would be called
 -> -> See ck-core/init.c,main.c and conv-core/threads.c
*/
void usageStop()
{
   if(!CpvAccess(PROCESSING)) return;

   CpvAccess(usedTime)   += CmiWallTimer() - CpvAccess(startTime);
   CpvAccess(PROCESSING) = 0;
}


static int getUsage(void)
{
   int usage = 0;
   double time      = CmiWallTimer();
   double totalTime = time - CpvAccess(beginTime);

   if(CpvAccess(PROCESSING))
   {
      CpvAccess(usedTime) += time - CpvAccess(startTime);
      CpvAccess(startTime) = time;
   }
   if(totalTime > 0.)
      usage = (int)((100 * CpvAccess(usedTime))/totalTime);
   CpvAccess(usedTime)  = 0.;
   CpvAccess(beginTime) = time;

   return usage;
}

static int getSchedQlen()
{
  return(CqsLength(CpvAccess(CsdSchedQueue)));
}

void CWebInit(void)
{
  CcsRegisterHandler("perf_monitor", CWebHandler);
  
  CWeb_CollectIndex=CmiRegisterHandler(CWeb_Collect);
  CWeb_ReduceIndex=CmiRegisterHandler(CWeb_Reduce);
  
  CWebPerformanceRegisterFunction(getUsage);
  CWebPerformanceRegisterFunction(getSchedQlen);

}

#endif /*CMK_WEB_MODE*/

/* \move */


/*****************************************************************************
 *
 * Converse Client-Server Functions
 *
 *****************************************************************************/

/*This struct describes a single CCS handler*/
typedef struct CcsListNode {
  char name[CCS_MAXHANDLER]; /*CCS handler name*/
  int hdlr; /*Converse handler index*/
  struct CcsListNode *next;
}CcsListNode;

CpvStaticDeclare(CcsListNode*, ccsList);/*Maps handler name to handler index*/
CpvStaticDeclare(CcsImplHeader,ccsReq);/*CCS requestor*/

void CcsUseHandler(char *name, int hdlr)
{
  CcsListNode *list=(CcsListNode *)malloc(sizeof(CcsListNode));
  CcsListNode *old=CpvAccess(ccsList);
  if (strlen(name)+1>=CCS_MAXHANDLER)
    CmiAbort("CCS Handler name too long to register!\n");
  list->next = old;
  CpvAccess(ccsList)=list;
  strcpy(list->name, name);
  list->hdlr = hdlr;
}

int CcsRegisterHandler(char *name, CmiHandler fn)
{
  int hdlr = CmiRegisterHandlerLocal(fn);
  CcsUseHandler(name, hdlr);
  return hdlr;
}

int CcsEnabled(void)
{
  return 1;
}

int CcsIsRemoteRequest(void)
{
  return (ChMessageInt(CpvAccess(ccsReq).ip) != 0);
}

void CcsCallerId(unsigned int *pip, unsigned int *pport)
{
  *pip = ChMessageInt(CpvAccess(ccsReq).ip);
  *pport = ChMessageInt(CpvAccess(ccsReq).port);
}

CcsDelayedReply CcsDelayReply(void)
{
  ChMessageInt_t fd=CpvAccess(ccsReq).replyFd;
  if (ChMessageInt(fd)==0)
     CmiAbort("CCS: Cannot delay reply to same request twice.\n");
  CpvAccess(ccsReq).replyFd=ChMessageInt_new(0);
  return *(CcsDelayedReply *)&fd;
}

void CcsSendReply(int size, const void *msg)
{
  int fd=ChMessageInt(CpvAccess(ccsReq).replyFd);
  if (fd==0)
      CmiAbort("CCS: Cannot reply to same request twice.\n");
  CcsImpl_reply(fd,size,msg);
  CpvAccess(ccsReq).replyFd=ChMessageInt_new(0);
}

void CcsSendDelayedReply(CcsDelayedReply d,int size, const void *msg)
{
  int fd=ChMessageInt( *(ChMessageInt_t *)&d );
  CcsImpl_reply(fd,size,msg);
}


/**********************************
CCS Implementation Routines:
  These do the request forwarding and
delivery.
***********************************/

/*CCS Bottleneck:
  Deliver the given message data to the given
CCS handler.
*/
static void CcsHandleRequest(CcsImplHeader *hdr,const char *reqData)
{
  char *cmsg;
  int reqLen=ChMessageInt(hdr->len);
/*Look up handler's converse ID*/
  int hdlrID;
  CcsListNode *list = CpvAccess(ccsList);
  /*CmiPrintf("CCS: message for handler %s\n", hdr->handler);*/
  while(list!=0) {
    if(strncmp(hdr->handler, list->name,CCS_MAXHANDLER)==0) {
      hdlrID = list->hdlr;
      break;
    }
    list = list->next;
  }
  if(list==0) {
    CmiPrintf("CCS: Unknown CCS handler name '%s' requested!\n",
	      hdr->handler);
    return;
 /*   CmiAbort("CCS: Unknown CCS handler name.\n");*/
  }

/*Pack user data into a converse message*/
  cmsg = (char *) CmiAlloc(CmiMsgHeaderSizeBytes+reqLen);
  memcpy(cmsg+CmiMsgHeaderSizeBytes, reqData, reqLen);

/* Call the handler */
  CmiSetHandler(cmsg, hdlrID);
  CpvAccess(ccsReq)=*hdr;
  CmiHandleMessage(cmsg);
  
/*Check if a reply was sent*/
  if (ChMessageInt(CpvAccess(ccsReq).replyFd)!=0)
    CcsSendReply(0,NULL);/*Send an empty reply if not*/
  CpvAccess(ccsReq).ip = ChMessageInt_new(0);
}

/*Unpacks request message to call above routine*/
static int req_fw_handler_idx;
static void req_fw_handler(char *msg)
{
  CcsHandleRequest((CcsImplHeader *)(msg+CmiMsgHeaderSizeBytes),
		   msg+CmiMsgHeaderSizeBytes+sizeof(CcsImplHeader));
  CmiGrabBuffer((void **)&msg); CmiFree(msg);  
}


/*Convert CCS header & message data into a converse message 
 addressed to handler*/
char *CcsImpl_ccs2converse(const CcsImplHeader *hdr,const void *data,int *ret_len)
{
  int reqLen=ChMessageInt(hdr->len);
  int len=CmiMsgHeaderSizeBytes+sizeof(CcsImplHeader)+reqLen;
  char *msg=(char *)CmiAlloc(len);
  memcpy(msg+CmiMsgHeaderSizeBytes,hdr,sizeof(CcsImplHeader));
  memcpy(msg+CmiMsgHeaderSizeBytes+sizeof(CcsImplHeader),data,reqLen);
  CmiSetHandler(msg, req_fw_handler_idx);
  if (ret_len!=NULL) *ret_len=len;
  return msg;
}

/*Forward this request to the appropriate PE*/
void CcsImpl_netRequest(CcsImplHeader *hdr,const void *reqData)
{
  int len,repPE=ChMessageInt(hdr->pe);
  char *msg=CcsImpl_ccs2converse(hdr,reqData,&len);
  CmiSyncSendAndFree(repPE,len,msg);
}


#if NODE_0_IS_CONVHOST
/************** NODE_0_IS_CONVHOST ***********
Non net- versions of charm++ are run without a 
(real) conv-host program.  This is fine, except 
CCS clients connect via conv-host; so for CCS
on non-net- versions of charm++, node 0 carries
out the CCS forwarding normally done in conv-host.

CCS works by listening to a TCP connection on a 
port-- the Ccs server socket.  A typical communcation
pattern is:

1.) Random program (CCS client) from the net
connects to the CCS server socket and sends
a CCS request.

2.) Node 0 forwards the request to the proper
PE as a regular converse message (built in CcsImpl_netReq)
for CcsHandleRequest.

3.) CcsHandleRequest looks up the user's pre-registered
CCS handler, and passes the user's handler the request data.

4.) The user's handler calls CcsSendReply with some
reply data; OR finishes without calling CcsSendReply,
in which case CcsHandleRequest does it.

5.) CcsSendReply forwards the reply back to node 0,
which sends the reply back to the original requestor,
on the (still-open) request socket.
 */

/*
Send a Ccs reply back to the requestor, down the given socket.
Since there is no conv-host, node 0 does all the CCS 
communication-- this means all requests come to node 0
and are forwarded out; all replies are forwarded back to node 0.

Note: on Net- versions, CcsImpl_reply is implemented in machine.c
*/
static int rep_fw_handler_idx;

void CcsImpl_reply(SOCKET repFd,int repLen,const void *repData)
{
  const int repPE=0;

  if (CmiMyPe()==repPE) {
    /*Actually deliver reply data*/
    CcsServer_sendReply(repFd,repLen,repData);
  } else {
    /*Forward data & socket # to the replyPE*/
    int len=CmiMsgHeaderSizeBytes+
	       sizeof(SOCKET)+sizeof(int)+repLen;
    char *msg=CmiAlloc(len);
    char *r=msg;
    *(SOCKET *)r=repFd; r+=sizeof(SOCKET);
    *(int *)r=repLen; r+=sizeof(int);
    memcpy(r,repData,repLen);
    CmiSetHandler(msg,rep_fw_handler_idx);
    CmiSyncSendAndFree(repPE,len,msg);
  }
}
/*Receives reply messages passed up from
converse to node 0.*/
static void rep_fw_handler(char *msg)
{
  int len;
  char *r=msg;
  SOCKET fd=*(SOCKET *)r; r+=sizeof(SOCKET);
  len=*(int *)r; r+=sizeof(int);
  CcsImpl_reply(fd,len,r);
  CmiGrabBuffer((void **)&msg); CmiFree(msg);
}

#endif /*NODE_0_IS_CONVHOST*/

#if NODE_0_IS_CONVHOST
/*
We have to run a CCS server socket here on
node 0.  To keep the speed impact minimal,
we only probe for new connections (in CommunicationInterrupt)
occasionally.  Convcore's main scheduler loop
will check our ccs_socket_ready flag, and call
CHostProcess if needed.
 */
#include <signal.h>
#include "ccs-server.c" /*Include implementation here in this case*/

static int inside_comm=0;
int ccs_socket_ready=0;/*Data pending on CCS server socket?*/

static void CommunicationInterrupt(void)
{
  if(inside_comm)
    return;
  if (1==skt_select1(CcsServer_fd(),0))
    ccs_socket_ready=1;
}

void CHostInit(int CCS_server_port)
{
  struct itimerval i;
  CcsServer_new(NULL,&CCS_server_port);
  CmiSignal(SIGALRM, 0, 0, CommunicationInterrupt);
  /*We will receive alarm signals at 10Hz*/
  i.it_interval.tv_sec = 0;
  i.it_interval.tv_usec = 100000;
  i.it_value.tv_sec = 0;
  i.it_value.tv_usec = 100000;
  setitimer(ITIMER_REAL, &i, NULL); 
}

void CHostProcess(void)
{
  CcsImplHeader hdr;
  char *data;
  if (1!=skt_select1(CcsServer_fd(),0)) return;
  inside_comm=1;
  printf("Got CCS connect...\n");
  if (CcsServer_recvRequest(&hdr,&data))
  {/*We got a network request*/
    printf("Got CCS request...\n");
    CcsImpl_netRequest(&hdr,data);
    free(data);
  }
  inside_comm=0;
}

#endif /*NODE_0_IS_CONVHOST*/


void CcsInit(void)
{
  CpvInitialize(CcsListNode*, ccsList);
  CpvAccess(ccsList) = 0;
  CpvInitialize(CcsImplHeader, ccsReq);
  CpvAccess(ccsReq).ip = ChMessageInt_new(0);
  req_fw_handler_idx = CmiRegisterHandler(req_fw_handler);
#if NODE_0_IS_CONVHOST
  rep_fw_handler_idx = CmiRegisterHandler(rep_fw_handler);
#endif
  CcsRegisterHandler("ccs_getinfo",ccs_getinfo);
  CcsRegisterHandler("ccs_killport",ccs_killport);
}

#endif /*CMK_CCS_AVAILABLE*/











