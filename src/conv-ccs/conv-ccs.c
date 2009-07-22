/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "converse.h"
#include "conv-ccs.h"
#include "ccs-server.h"
#include "sockRoutines.h"
#include "queueing.h"

#if CMK_CCS_AVAILABLE

/*****************************************************************************
 *
 * Converse Client-Server Functions
 *
 *****************************************************************************/
 
#include "ckhashtable.h"

/* Includes all information stored about a single CCS handler. */
typedef struct CcsHandlerRec {
	const char *name; /*Name passed over socket*/
	CmiHandler fnOld; /*Old converse-style handler, or NULL if new-style*/
	CcsHandlerFn fn; /*New-style handler function, or NULL if old-style*/
	void *userPtr;
	int nCalls; /* Number of times handler has been executed*/
} CcsHandlerRec;

static void initHandlerRec(CcsHandlerRec *c,const char *name) {
  if (strlen(name)>=CCS_MAXHANDLER) 
  	CmiAbort("CCS handler names cannot exceed 32 characters");
  c->name=strdup(name);
  c->fn=NULL;
  c->fnOld=NULL;
  c->userPtr=NULL;
  c->nCalls=0;
}

static void callHandlerRec(CcsHandlerRec *c,int reqLen,const void *reqData) {
	c->nCalls++;
	if (c->fnOld) 
	{ /* Backward compatability version:
	    Pack user data into a converse message (cripes! why bother?);
	    user will delete the message. 
	  */
		char *cmsg = (char *) CmiAlloc(CmiMsgHeaderSizeBytes+reqLen);
		memcpy(cmsg+CmiMsgHeaderSizeBytes, reqData, reqLen);
		(c->fnOld)(cmsg);
	}
	else { /* Pass read-only copy of data straight to user */
		(c->fn)(c->userPtr, reqLen, reqData);
	}
}

/*Table maps handler name to CcsHandler object*/
typedef CkHashtable_c CcsHandlerTable;
CpvStaticDeclare(CcsHandlerTable, ccsTab);

CpvStaticDeclare(CcsImplHeader*,ccsReq);/*Identifies CCS requestor (client)*/

void CcsRegisterHandler(const char *name, CmiHandler fn)
{
  CcsHandlerRec cp;
  initHandlerRec(&cp,name);
  cp.fnOld=fn;
  *(CcsHandlerRec *)CkHashtablePut(CpvAccess(ccsTab),(void *)&cp.name)=cp;
}
void CcsRegisterHandlerFn(const char *name, CcsHandlerFn fn, void *ptr)
{
  CcsHandlerRec cp;
  initHandlerRec(&cp,name);
  cp.fn=fn;
  cp.userPtr=ptr;
  *(CcsHandlerRec *)CkHashtablePut(CpvAccess(ccsTab),(void *)&cp.name)=cp;
}

int CcsEnabled(void)
{
  return 1;
}

int CcsIsRemoteRequest(void)
{
  return CpvAccess(ccsReq)!=NULL;
}

void CcsCallerId(skt_ip_t *pip, unsigned int *pport)
{
  *pip = CpvAccess(ccsReq)->attr.ip;
  *pport = ChMessageInt(CpvAccess(ccsReq)->attr.port);
}

CcsDelayedReply CcsDelayReply(void)
{
  CcsDelayedReply ret;
  ret.attr=CpvAccess(ccsReq)->attr;
  ret.replyFd=CpvAccess(ccsReq)->replyFd;
  CpvAccess(ccsReq)=NULL;
  return ret;
}

void CcsSendReply(int replyLen, const void *replyData)
{
  if (CpvAccess(ccsReq)==NULL)
    CmiAbort("CcsSendReply: reply already sent!\n");
  CpvAccess(ccsReq)->len = ChMessageInt_new(1);
  CcsImpl_reply(CpvAccess(ccsReq),replyLen,replyData);
  CpvAccess(ccsReq) = NULL;
}

void CcsSendDelayedReply(CcsDelayedReply d,int replyLen, const void *replyData)
{
  CcsImplHeader h;
  h.attr=d.attr;
  h.replyFd=d.replyFd;
  h.len=ChMessageInt_new(1);
  CcsImpl_reply(&h,replyLen,replyData);
}

void CcsNoReply()
{
  if (CpvAccess(ccsReq)==NULL) return;
  CpvAccess(ccsReq)->len = ChMessageInt_new(0);
  CcsImpl_reply(CpvAccess(ccsReq),0,NULL);
  CpvAccess(ccsReq) = NULL;
}

void CcsNoDelayedReply(CcsDelayedReply d)
{
  CcsImplHeader h;
  h.attr=d.attr;
  h.replyFd=d.replyFd;
  h.len = ChMessageInt_new(0);
  CcsImpl_reply(&h,0,NULL);
}


/**********************************
_CCS Implementation Routines:
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
  char *handlerStr=hdr->handler;
  CcsHandlerRec *fn=(CcsHandlerRec *)CkHashtableGet(CpvAccess(ccsTab),(void *)&handlerStr);
  if (fn==NULL) {
    CmiPrintf("CCS: Unknown CCS handler name '%s' requested. Ignoring...\n",
	      hdr->handler);
    CpvAccess(ccsReq)=hdr;
    CcsSendReply(0,NULL); /*Send an empty reply to the possibly waiting client*/
    return;
 /*   CmiAbort("CCS: Unknown CCS handler name.\n");*/
  }

/* Call the handler */
  CpvAccess(ccsReq)=hdr;
  callHandlerRec(fn,reqLen,reqData);
  
/*Check if a reply was sent*/
  if (CpvAccess(ccsReq)!=NULL)
    CcsSendReply(0,NULL);/*Send an empty reply if not*/
}

/*Unpacks request message to call above routine*/
int _ccsHandlerIdx;/*Converse handler index of below routine*/
static void req_fw_handler(char *msg)
{
  CcsHandleRequest((CcsImplHeader *)(msg+CmiMsgHeaderSizeBytes),
		   msg+CmiMsgHeaderSizeBytes+sizeof(CcsImplHeader));
  CmiFree(msg);  
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
  CmiSetHandler(msg, _ccsHandlerIdx);
  if (ret_len!=NULL) *ret_len=len;
  return msg;
}

/*Forward this request to the appropriate PE*/
void CcsImpl_netRequest(CcsImplHeader *hdr,const void *reqData)
{
  int len,repPE=ChMessageInt(hdr->pe);
  if (repPE<0 && repPE>=CmiNumPes()) {
	repPE=0;
	hdr->pe=ChMessageInt_new(repPE);
  }

  {
    char *msg=CcsImpl_ccs2converse(hdr,reqData,&len);
    CmiSyncSendAndFree(repPE,len,msg);
  }
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

void CcsImpl_reply(CcsImplHeader *rep,int repLen,const void *repData)
{
  const int repPE=0;
  rep->len=ChMessageInt_new(repLen);
  if (CmiMyPe()==repPE) {
    /*Actually deliver reply data*/
    CcsServer_sendReply(rep,repLen,repData);
  } else {
    /*Forward data & socket # to the replyPE*/
    int len=CmiMsgHeaderSizeBytes+
	       sizeof(CcsImplHeader)+repLen;
    char *msg=CmiAlloc(len);
    char *r=msg+CmiMsgHeaderSizeBytes;
    *(CcsImplHeader *)r=*rep; r+=sizeof(CcsImplHeader);
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
  char *r=msg+CmiMsgHeaderSizeBytes;
  CcsImplHeader *hdr=(CcsImplHeader *)r; 
  r+=sizeof(CcsImplHeader);
  len=ChMessageInt(hdr->len);
  CcsImpl_reply(hdr,len,r);
  CmiFree(msg);
}

/*No request will be sent through this socket.
Closes it.
*/
void CcsImpl_noReply(CcsImplHeader *hdr)
{
  int fd=ChMessageInt(hdr->replyFd);
  skt_close(fd);
}

/*
We have to run a CCS server socket here on
node 0.  To keep the speed impact minimal,
we only probe for new connections (with CcsServerCheck)
occasionally.  
 */
#include <signal.h>
#include "ccs-server.c" /*Include implementation here in this case*/
#include "ccs-auth.c"

/*Check for ready Ccs messages:*/
void CcsServerCheck(void)
{
  while (1==skt_select1(CcsServer_fd(),0)) {
    CcsImplHeader hdr;
    void *data;
    /* printf("Got CCS connect...\n"); */
    if (CcsServer_recvRequest(&hdr,&data))
    {/*We got a network request*/
      /* printf("Got CCS request...\n"); */
      if (! check_stdio_header(&hdr)) {
        CcsImpl_netRequest(&hdr,data);
      }
      free(data);
    }
  }
}

#endif /*NODE_0_IS_CONVHOST*/

int _isCcsHandlerIdx(int hIdx) {
  if (hIdx==_ccsHandlerIdx) return 1;
#if NODE_0_IS_CONVHOST 
  if (hIdx==rep_fw_handler_idx) return 1;
#endif
  return 0;
}

void CcsBuiltinsInit(char **argv);

CpvDeclare(int, cmiArgDebugFlag);
CpvDeclare(char *, displayArgument);

void CcsInit(char **argv)
{
  CpvInitialize(CkHashtable_c, ccsTab);
  CpvAccess(ccsTab) = CkCreateHashtable_string(sizeof(CcsHandlerRec),5);
  CpvInitialize(CcsImplHeader *, ccsReq);
  CpvAccess(ccsReq) = NULL;
  _ccsHandlerIdx = CmiRegisterHandler((CmiHandler)req_fw_handler);
  CpvInitialize(int, cmiArgDebugFlag);
  CpvInitialize(char *, displayArgument);
  CpvAccess(cmiArgDebugFlag) = 0;
  CpvAccess(displayArgument) = NULL;

  CcsBuiltinsInit(argv);

#if NODE_0_IS_CONVHOST
  rep_fw_handler_idx = CmiRegisterHandler((CmiHandler)rep_fw_handler);
#if ! CMK_CMIPRINTF_IS_A_BUILTIN
  print_fw_handler_idx = CmiRegisterHandler((CmiHandler)print_fw_handler);
#endif
  {
   int ccs_serverPort=0;
   char *ccs_serverAuth=NULL;
   
   if (CmiGetArgFlagDesc(argv,"++server", "Create a CCS server port") | 
      CmiGetArgIntDesc(argv,"++server-port",&ccs_serverPort, "Listen on this TCP/IP port number") |
      CmiGetArgStringDesc(argv,"++server-auth",&ccs_serverAuth, "Use this CCS authentication file")) 
    if (CmiMyPe()==0)
    {/*Create and occasionally poll on a CCS server port*/
      CcsServer_new(NULL,&ccs_serverPort,ccs_serverAuth);
      CcdCallOnConditionKeep(CcdPERIODIC,(CcdVoidFn)CcsServerCheck,NULL);
    }
  }
#endif
  /* if in parallel debug mode i.e ++cpd, freeze */
  if (CmiGetArgFlagDesc(argv, "+cpd", "Used *only* in conjunction with parallel debugger"))
  {
     CpvAccess(cmiArgDebugFlag) = 1;
     if (CmiGetArgStringDesc(argv, "+DebugDisplay",&(CpvAccess(displayArgument)), "X display for gdb used only in cpd mode"))
     {
        if (CpvAccess(displayArgument) == NULL)
            CmiPrintf("WARNING> NULL parameter for +DebugDisplay\n***");
     }
     else if (CmiMyPe() == 0)
     {
            /* only one processor prints the warning */
            CmiPrintf("WARNING> x term for gdb needs to be specified as +DebugDisplay by debugger\n***\n");
     }

  }
}

#endif /*CMK_CCS_AVAILABLE*/

