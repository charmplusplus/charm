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
#define CcsInit()
#define CcsUseHandler(x,y)
#define CcsRegisterHandler(x,y) 0
#define CcsEnabled() 0
#define CcsIsRemoteRequest() 0
#define CcsCallerId(x,y)
#define CcsSendReply(i,p,s,r)
#endif

#endif
