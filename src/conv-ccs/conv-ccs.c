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
static void noMoreErrors(int c,const char *m) {exit(1);}
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
#include "fifo.h"
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
  CpvAccess(debugQueue) = FIFO_Create();
    
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

#define WEB_INTERVAL 2000
#define MAXFNS 20

/* For Web Performance */
typedef int (*CWebFunction)();
unsigned int appletIP;
unsigned int appletPort;
int countMsgs;
char **valueArray;
CWebFunction CWebPerformanceFunctionArray[MAXFNS];
int CWebNoOfFns;
CpvDeclare(int, CWebPerformanceDataCollectionHandlerIndex);
CpvDeclare(int, CWebHandlerIndex);

static void sendDataFunction(void)
{
  char *reply;
  int len = 0, i;

  for(i=0; i<CmiNumPes(); i++){
    len += (strlen((char*)(valueArray[i]+
			   CmiMsgHeaderSizeBytes+sizeof(int)))+1);
    /* for the spaces in between */
  }
  len+=6; /* for 'perf ' and the \0 at the end */

  reply = (char *)malloc(len * sizeof(char));
  strcpy(reply, "perf ");

  for(i=0; i<CmiNumPes(); i++){
    strcat(reply, (valueArray[i] + CmiMsgHeaderSizeBytes + sizeof(int)));
    strcat(reply, " ");
  }

  /* Do the CcsSendReply */
  CcsSendReply(strlen(reply) + 1, reply);

  /*
  CmiPrintf("Reply = %s\n", reply);
  */
  free(reply);

  /* Free valueArray contents */
  for(i = 0; i < CmiNumPes(); i++){
    CmiFree(valueArray[i]);
    valueArray[i] = 0;
  }

  countMsgs = 0;
}

void CWebPerformanceDataCollectionHandler(char *msg){
  int src;
  char *prev;

  if(CmiMyPe() != 0){
    CmiAbort("Wrong processor....\n");
  }
  src = ((int *)(msg + CmiMsgHeaderSizeBytes))[0];
  CmiGrabBuffer((void **)&msg);
  prev = valueArray[src]; /* Previous value, ideally 0 */
  valueArray[src] = (msg);
  if(prev == 0) countMsgs++;
  else CmiFree(prev);

  if(countMsgs == CmiNumPes()){
    sendDataFunction();
  }
}

void CWebPerformanceGetData(void *dummy)
{
  char *msg, data[100];
  int msgSize;
  int i;

  strcpy(data, "");
  /* Evaluate each of the functions and get the values */
  for(i = 0; i < CWebNoOfFns; i++)
    sprintf(data, "%s %d", data, (*(CWebPerformanceFunctionArray[i]))());

  msgSize = (strlen(data)+1) + sizeof(int) + CmiMsgHeaderSizeBytes;
  msg = (char *)CmiAlloc(msgSize);
  ((int *)(msg + CmiMsgHeaderSizeBytes))[0] = CmiMyPe();
  strcpy(msg + CmiMsgHeaderSizeBytes + sizeof(int), data);
  CmiSetHandler(msg, CpvAccess(CWebPerformanceDataCollectionHandlerIndex));
  CmiSyncSendAndFree(0, msgSize, msg);

  CcdCallFnAfter(CWebPerformanceGetData, 0, WEB_INTERVAL);
}

void CWebPerformanceRegisterFunction(CWebFunction fn)
{
  CWebPerformanceFunctionArray[CWebNoOfFns] = fn;
  CWebNoOfFns++;
}

static void CWebHandler(char *msg){
  int msgSize;
  char *getStuffMsg;
  int i;

  if(CcsIsRemoteRequest()) {
    char name[32];
    unsigned int ip, port;

    CcsCallerId(&ip, &port);
    sscanf(msg+CmiMsgHeaderSizeBytes, "%s", name);

    if(strcmp(name, "getStuff") == 0){
      appletIP = ip;
      appletPort = port;

      valueArray = (char **)malloc(sizeof(char *) * CmiNumPes());
      for(i = 0; i < CmiNumPes(); i++)
        valueArray[i] = 0;

      for(i = 0; i < CmiNumPes(); i++){
        msgSize = CmiMsgHeaderSizeBytes + 2*sizeof(int);
        getStuffMsg = (char *)CmiAlloc(msgSize);
        ((int *)(getStuffMsg + CmiMsgHeaderSizeBytes))[0] = appletIP;
        ((int *)(getStuffMsg + CmiMsgHeaderSizeBytes))[1] = appletPort;
        CmiSetHandler(getStuffMsg, CpvAccess(CWebHandlerIndex));
        CmiSyncSendAndFree(i, msgSize, getStuffMsg);

        CcdCallFnAfter(CWebPerformanceGetData, 0, WEB_INTERVAL);
      }
    }
    else{
      CmiPrintf("incorrect command:%s received, len=%ld\n",name,strlen(name));
    }
  }
}

static int f2()
{
  return(CqsLength(CpvAccess(CsdSchedQueue)));
}

static int f3()
{
  struct timeval tmo;

  gettimeofday(&tmo, NULL);
  return(tmo.tv_sec % 10 + CmiMyPe() * 3);
}

/** ADDED 2-14-99 BY MD FOR USAGE TRACKING (TEMPORARY) **/

/* #define CkUTimer()      ((int)(CmiWallTimer() * 1000000.0)) */

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

int getUsage()
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
      usage = (100 * CpvAccess(usedTime))/totalTime;
   CpvAccess(usedTime)  = 0.;
   CpvAccess(beginTime) = time;

   return usage;
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

void CWebInit(void)
{
  CcsRegisterHandler("MonitorHandler", CWebHandler);

  CpvInitialize(int, CWebHandlerIndex);
  CpvAccess(CWebHandlerIndex) = CmiRegisterHandler(CWebHandler);

  CpvInitialize(int, CWebPerformanceDataCollectionHandlerIndex);
  CpvAccess(CWebPerformanceDataCollectionHandlerIndex) =
    CmiRegisterHandler(CWebPerformanceDataCollectionHandler);

  CWebPerformanceRegisterFunction(getUsage);
  CWebPerformanceRegisterFunction(f2);

}

#endif

/* \move */


/* move */

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

void CcsSendReply(int size, const char *msg)
{
  int fd=ChMessageInt(CpvAccess(ccsReq).replyFd);
  if (fd<=0)
      CmiAbort("CCS: Cannot reply to same request twice.\n");
  CcsImpl_reply(fd,size,msg);
  CpvAccess(ccsReq).replyFd=ChMessageInt_new(0);
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
    CmiAbort("CCS: Unknown CCS handler name.\n");
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
char *CcsImpl_ccs2converse(const CcsImplHeader *hdr,const char *data,int *ret_len)
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
void CcsImpl_netRequest(CcsImplHeader *hdr,const char *reqData)
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

void CcsImpl_reply(SOCKET repFd,int repLen,const char *repData)
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
  CmiSignal(SIGALRM, SIGIO, 0, CommunicationInterrupt);
#if !CMI_ASYNC_NOT_NEEDED
  CmiEnableAsyncIO(CcsServer_fd());
#endif
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
  if (ccs_socket_ready==0) return;
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











