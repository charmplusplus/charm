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

/*Table maps handler name to handler fn*/
typedef CkHashtable_c CcsHandlerTable;
CpvStaticDeclare(CcsHandlerTable, ccsTab);

CpvStaticDeclare(CcsImplHeader*,ccsReq);/*Identifies CCS requestor (client)*/

void CcsRegisterHandler(const char *name, CmiHandler fn)
{
  *(CmiHandler *)CkHashtablePut(CpvAccess(ccsTab),(void *)&name)=fn;
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
  CcsImplHeader *ret=(CcsImplHeader *)malloc(sizeof(CcsImplHeader));
  *ret=*CpvAccess(ccsReq);
  CpvAccess(ccsReq)=NULL;
  return (CcsDelayedReply)ret;
}

void CcsSendReply(int size, const void *msg)
{
  if (CpvAccess(ccsReq)==NULL)
    CmiAbort("CcsSendReply: reply already sent!\n");
  CcsImpl_reply(CpvAccess(ccsReq),size,msg);
  CpvAccess(ccsReq) = NULL;
}

void CcsSendDelayedReply(CcsDelayedReply d,int size, const void *msg)
{
  CcsImpl_reply((CcsImplHeader *)d,size,msg);
  free(d);
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
  char *handlerStr=hdr->handler;
  void *hdlrPtr=CkHashtableGet(CpvAccess(ccsTab),(void *)&handlerStr);
  CmiHandler fn;
  if (hdlrPtr==NULL) {
    CmiPrintf("CCS: Unknown CCS handler name '%s' requested!\n",
	      hdr->handler);
    return;
 /*   CmiAbort("CCS: Unknown CCS handler name.\n");*/
  }
  fn=*(CmiHandler *)hdlrPtr;

/*Pack user data into a converse message (why bother?)*/
  cmsg = (char *) CmiAlloc(CmiMsgHeaderSizeBytes+reqLen);
  memcpy(cmsg+CmiMsgHeaderSizeBytes, reqData, reqLen);

/* Call the handler */
  CpvAccess(ccsReq)=hdr;
  (fn)(cmsg);
  
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
    char *r=msg;
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
  char *r=msg;
  CcsImplHeader *hdr=(CcsImplHeader *)r; 
  r+=sizeof(CcsImplHeader);
  len=ChMessageInt(hdr->len);
  CcsImpl_reply(hdr,len,r);
  CmiFree(msg);
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
      CcsImpl_netRequest(&hdr,data);
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

void CcsInit(char **argv)
{
  CpvInitialize(CkHashtable_c, ccsTab);
  CpvAccess(ccsTab) = CkCreateHashtable_string(sizeof(CmiHandler),5);
  CpvInitialize(CcsImplHeader *, ccsReq);
  CpvAccess(ccsReq) = NULL;
  _ccsHandlerIdx = CmiRegisterHandler(req_fw_handler);

  CcsBuiltinsInit(argv);

#if NODE_0_IS_CONVHOST
  rep_fw_handler_idx = CmiRegisterHandler(rep_fw_handler);
  {
   int ccs_serverPort=0;
   char *ccs_serverAuth=NULL;
   
   if (CmiGetArgFlag(argv,"++server") | 
      CmiGetArgInt(argv,"++server-port",&ccs_serverPort) |
      CmiGetArgString(argv,"++server-auth",&ccs_serverAuth)) 
    if (CmiMyPe()==0)
    {/*Create and occasionally poll on a CCS server port*/
      CcsServer_new(NULL,&ccs_serverPort,ccs_serverAuth);
      CcdCallOnConditionKeep(CcdPERIODIC,(CcdVoidFn)CcsServerCheck,NULL);
    }
  }
#endif
}

#endif /*CMK_CCS_AVAILABLE*/











