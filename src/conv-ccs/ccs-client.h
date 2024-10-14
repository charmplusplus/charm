/**
 * Converse Client-Server Module: Client Side
 */

#ifndef __CCS_CLIENT_H_
#define __CCS_CLIENT_H_

#define CMK_NOT_USE_CONVERSE 1
#include "sockRoutines.h"
#include "ccs-auth.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CcsServer {
  /*CCS Server description:*/
  char hostAddr[128];
  skt_ip_t hostIP;
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
int CcsConnect(CcsServer *svr, const char *host, int port,const CcsSec_secretKey *key);
int CcsConnectWithTimeout(CcsServer *svr, const char *host, int port,
	const CcsSec_secretKey *key, int timeout);

int CcsConnectIp(CcsServer *svr,skt_ip_t ip, int port,const CcsSec_secretKey *key);
int CcsConnectIpWithTimeout(CcsServer *svr,skt_ip_t ip, int port,const CcsSec_secretKey *key, int timeout);

int CcsSendRequest(CcsServer *svr, const char *hdlrID, int pe, 
		    int size, const void *msg);
int CcsSendRequestWithTimeout(CcsServer *svr, const char *hdlrID, int pe, 
		    int size, const void *msg, int timeout);
int CcsSendBroadcastRequest(CcsServer *svr, const char *hdlrID,
            int size, const void *msg);
int CcsSendBroadcastRequestWithTimeout(CcsServer *svr, const char *hdlrID, 
            int size, const void *msg, int timeout);
int CcsSendMulticastRequest(CcsServer *svr, const char *hdlrID, int npes, 
            int *pes, int size, const void *msg);
int CcsSendMulticastRequestWithTimeout(CcsServer *svr, const char *hdlrID, int npes, 
            int *pes, int size, const void *msg, int timeout);

int CcsNoResponse(CcsServer *svr);
int CcsRecvResponse(CcsServer *svr, 
		    int maxsize, void *recvBuffer, int timeout);
int CcsRecvResponseMsg(CcsServer *svr, 
		    int *retSize,void **newBuf, int timeout);
int CcsNumNodes(CcsServer *svr);
int CcsNumPes(CcsServer *svr);
int CcsNodeFirst(CcsServer *svr, int node);
int CcsNodeSize(CcsServer *svr,int node);
int CcsProbe(CcsServer *svr);
int CcsProbeTimeout(CcsServer *svr,int timeoutMs);
void CcsFinalize(CcsServer *svr);

#ifdef __cplusplus
};
#endif

#endif
