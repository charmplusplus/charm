/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/
/*
Converse Client-Server:
  Lets you execute, from an arbitrary program on the network, 
pre-registered "handlers" in a running converse program.  
Also allows you to recv replies from the handlers.  
All requests and replies consist of user-defined binary data.

  This file provides the client interface.

CCS Protocol spec:

 A CCS request message asks a running Converse program to
execute a pre-registered "handler" routine.  You send the
request directly to conv-host's CCS server port.
The request, with header, has the following format on the
network: 
Ccs Message----------------------------------------------
 /--CcsMessageHeader---------------------------       ^
 | 4 bytes  |   Message data length d         ^       |
 | 4 bytes  |   Dest. processor number        |       |
 |          |   (big-endian binary integers)  |   40+d bytes
 +-----------------------------------      40 bytes   |
 |32 bytes  |   CCS Handler name              |       |
 |          |   (ASCII, Null-terminated)      v       |
 \---------------------------------------------       |
    d bytes |   User data (passed to handler)         v
-------------------------------------------------------

 A CCS reply message (if any) comes back on the request socket,
and has only a length header:
CCS Reply ----------------------------------
 | 4 bytes  |   Message data length d        
 |          |   (big-endian binary integer)  
 +----------------------------------------- 
 | d bytes  |   User data                   
--------------------------------------------

 */
#include "ccs.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*Include the socket and message interface routines
  here *whole*, which keeps client linking simple.*/
#include "sockRoutines.c"

#define DEBUGF(x) printf x

/*Parse list of nodes given to us by conv-host.
*/
static void parseInfo(CcsServer *svr,const char *data)
{
  /*Data conv-host sends us is just a big list of integers*/
  const ChMessageInt_t *d=(const ChMessageInt_t *)data;
  int i,index=0; /*Current offset in above array*/

  svr->numNodes=ChMessageInt(d[index++]);
  svr->numProcs = (int *) malloc(svr->numNodes * sizeof(int));
  svr->numPes = 0;
  for(i=0;i<svr->numNodes;i++)
    svr->numPes+=svr->numProcs[i]=ChMessageInt(d[index++]);
}

static void printSvr(CcsServer *svr)
{
  int i;
  DEBUGF(("hostIP: %d\n", svr->hostIP));
  DEBUGF(("hostPort: %d\n", svr->hostPort));
  DEBUGF(("replyFd: %d\n", svr->replyFd));
  DEBUGF(("numNodes: %d\n", svr->numNodes));
  DEBUGF(("numPes: %d\n", svr->numPes));
  for(i=0;i<svr->numNodes;i++) {
    DEBUGF(("Node[%d] has %d processors\n",i, svr->numProcs[i]));
  }
}

/**
 * Converse Client-Server Module: Client Side
 */
void CcsConnect(CcsServer *svr, char *host, int port)
{
  skt_init();
  CcsConnectIp(svr,skt_lookup_ip(host),port);
}
void CcsConnectIp(CcsServer *svr, int ip, int port)
{
  unsigned int msg_len;char *msg_data;/*Reply message*/
  skt_init();
  svr->hostIP = ip;
  svr->hostPort = port;
  svr->replyFd=INVALID_SOCKET;

  /*Request the parallel machine's node info*/
  CcsSendRequest(svr,"ccs_getinfo",0,0,NULL);
  
  /*Wait for conv-host to get back to us*/
  DEBUGF(("Waiting for conv-host to call us back...\n"));
  CcsRecvResponseMsg(svr,&msg_len,&msg_data,60);
  parseInfo(svr,msg_data);
  free(msg_data);
  
  /**/ printSvr(svr);/**/
}

int CcsNumNodes(CcsServer *svr)
{
  return svr->numNodes;
}

int CcsNumPes(CcsServer *svr)
{
  return svr->numPes;
}

int CcsNodeFirst(CcsServer *svr, int node)
{
  int retval=0,i;
  for(i=0;i<node;i++) {
    retval += svr->numProcs[node];
  }
  return retval;
}

int CcsNodeSize(CcsServer *svr,int node)
{
  return svr->numProcs[node];
}

void CcsSendRequest(CcsServer *svr, char *hdlrID, int pe, unsigned int size, const char *msg)
{
  CcsMessageHeader hdr;/*CCS request header*/

  /*Close the old connection (if any)*/
  if (svr->replyFd!=-1) {skt_close(svr->replyFd);svr->replyFd=-1;}

  /*Connect to conv-host, and send the message */
  svr->replyFd=skt_connect(svr->hostIP, svr->hostPort,120);

  hdr.len=ChMessageInt_new(size);
  hdr.pe=ChMessageInt_new(pe);
  strncpy(hdr.handler,hdlrID,CCS_HANDLERLEN);
  skt_sendN(svr->replyFd, (char *)&hdr, sizeof(hdr));
  skt_sendN(svr->replyFd, msg, size);
  /*Leave socket open for reply*/
}

/*Receive data back from the server. (Arbitrary length response)
*/
int CcsRecvResponseMsg(CcsServer *svr, unsigned int *size,char **newBuf, int timeout)
{
  ChMessageInt_t netLen;
  unsigned int len;  
  SOCKET fd=svr->replyFd;
  skt_recvN(fd,(char *)&netLen,sizeof(netLen));
  *size=len=ChMessageInt(netLen);
  *newBuf=(char *)malloc(len);
  skt_recvN(fd,(char *)*newBuf,len);
  return len;
}

/*Receive data from the server. (In-place receive)
*/
int CcsRecvResponse(CcsServer *svr,  unsigned int maxsize, char *recvBuffer,int timeout)
{
  ChMessageInt_t netLen;
  unsigned int len;
  SOCKET fd=svr->replyFd;
  skt_recvN(fd,(char *)&netLen,sizeof(netLen));
  len=ChMessageInt(netLen);
  if (len>maxsize) 
    {skt_close(fd);return -1;/*Buffer too small*/}
  skt_recvN(fd,(char *)recvBuffer,len);
  return len;
}

int CcsProbe(CcsServer *svr)
{
  fprintf(stderr, "CcsProbe not implemented.\n");
  exit(1);
  return 1;
}

void CcsFinalize(CcsServer *svr)
{
  if (svr->replyFd!=-1) skt_close(svr->replyFd);
}





