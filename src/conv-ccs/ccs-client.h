/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * Converse Client-Server Module: Client Side
 */

#ifndef __CCS_CLIENT_H_
#define __CCS_CLIENT_H_

#include "sockRoutines.h"
#include "ccs-auth.h"

typedef struct CcsServer {
  /*Conv-host:*/
  char hostAddr[128];
  unsigned int hostIP;
  unsigned int hostPort;

  /*Authentication*/
  int isAuth;
  int level;/*Security level to ask for*/
  CcsSec_secretKey key;
  int clientID,clientSalt;
  int replySalt;
  CCS_RAND_state rand;

  /*Parallel machine:*/
  int numNodes;
  int numPes;
  int *numProcs; /*# of processors for each node*/

  /*Current State:*/
  SOCKET replyFd;/*Socket for replies*/
} CcsServer;

/*All routines return -1 on failure*/
int CcsConnect(CcsServer *svr, char *host, int port,CcsSec_secretKey *key);
int CcsConnectIp(CcsServer *svr, int ip, int port,CcsSec_secretKey *key);
int CcsSendRequest(CcsServer *svr, char *hdlrID, int pe, 
		    unsigned int size, const char *msg);
int CcsRecvResponse(CcsServer *svr, 
		    unsigned int maxsize, char *recvBuffer, int timeout);
int CcsRecvResponseMsg(CcsServer *svr, 
		    unsigned int *retSize,char **newBuf, int timeout);
int CcsNumNodes(CcsServer *svr);
int CcsNumPes(CcsServer *svr);
int CcsNodeFirst(CcsServer *svr, int node);
int CcsNodeSize(CcsServer *svr,int node);
int CcsProbe(CcsServer *svr);
void CcsFinalize(CcsServer *svr);

#endif
