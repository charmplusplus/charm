/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * Converse Client-Server Module: Client Side
 */

#ifndef _CCS_H_
#define _CCS_H_

#include "sockRoutines.h"

typedef struct CcsServer {
  /*Conv-host:*/
  char hostAddr[128];
  unsigned int hostIP;
  unsigned int hostPort;
  /*Parallel machine:*/
  int numNodes;
  int numPes;
  int *numProcs; /*# of processors for each node*/
  /*Current State:*/
  SOCKET replyFd;/*Socket for replies*/
} CcsServer;

void CcsConnect(CcsServer *svr, char *host, int port);
void CcsConnectIp(CcsServer *svr, int ip, int port);
int CcsNumNodes(CcsServer *svr);
int CcsNumPes(CcsServer *svr);
int CcsNodeFirst(CcsServer *svr, int node);
int CcsNodeSize(CcsServer *svr,int node);
void CcsSendRequest(CcsServer *svr, char *hdlrID, int pe, 
		    unsigned int size, const char *msg);
int CcsRecvResponse(CcsServer *svr, 
		    unsigned int maxsize, char *recvBuffer, int timeout);
int CcsRecvResponseMsg(CcsServer *svr, 
		    unsigned int *retSize,char **newBuf, int timeout);
int CcsProbe(CcsServer *svr);
void CcsFinalize(CcsServer *svr);

#endif
