/**
 * Converse Client-Server Module: Client Side
 */

#ifndef _CCS_H_
#define _CCS_H_

typedef int (*CcsHandlerFn)(int, void *);

typedef struct CcsServer {
  char hostAddr[128];
  unsigned int hostIP;
  unsigned int hostPort;
  unsigned int myIP;
  unsigned int myPort;
  int myFd;
  int persFd;
  int numNodes;
  int numPes;
  int *numProcs;
  int *nodeIPs;
  int *nodePorts;
  CcsHandlerFn callback;
} CcsServer;

int CcsConnect(CcsServer *svr, char *host, int port);
int CcsNumNodes(CcsServer *svr);
int CcsNumPes(CcsServer *svr);
int CcsNodeFirst(CcsServer *svr, int node);
int CcsNodeSize(CcsServer *svr,int node);
int CcsSendRequest(CcsServer *svr, char *hdlrID, int pe, unsigned int size, void *msg);
int CcsRecvResponse(CcsServer *svr, unsigned int maxsize, void *recvBuffer, int timeout);
int CcsProbe(CcsServer *svr);
int CcsResponseHandler(CcsServer *svr, CcsHandlerFn fn);
int CcsFinalize(CcsServer *svr);

#define MAXLINE 1024
#define FIXED_LENGTH 17

#endif
