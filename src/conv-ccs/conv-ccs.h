/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef CONV_CCS_H
#define CONV_CCS_H

#include "converse.h"

/******* Converse Client Server *****/

#define CMK_CCS_VERSION "1"

#if CMK_CCS_AVAILABLE
void CcsUseHandler(char *id, int hdlr);
int CcsRegisterHandler(char *id, CmiHandler fn);
int CcsEnabled(void);
int CcsIsRemoteRequest(void);
void CcsCallerId(unsigned int *pip, unsigned int *pport);
void CcsSendReply(unsigned int ip, unsigned int port, int size, void *reply);
void CcsSendReplyFd(unsigned int ip, unsigned int port, int size, void *reply);
#else
#define CcsInit() do { } while(0)
#define CcsUseHandler(x,y) do { } while(0)
#define CcsRegisterHandler(x,y) 0
#define CcsEnabled() 0
#define CcsIsRemoteRequest() 0
#define CcsCallerId(x,y) do { } while(0)
#define CcsSendReply(i,p,s,r) do { } while(0)
#define CcsSendReplyFd(i,p,s,r) do { } while(0)
#endif

#if NODE_0_IS_CONVHOST
extern int serverFlag;
extern int inside_comm;
CpvExtern(int, strHandlerID);
extern int hostport, hostskt;
extern int hostskt_ready_read;
CpvExtern(int, CHostHandlerIndex);
extern unsigned int *nodeIPs;
extern unsigned int *nodePorts;
extern int numRegistered;
extern void CommunicationServer();
extern unsigned int clientIP, clientPort, clientKillPort;
#endif

#endif
