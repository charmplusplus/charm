/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/*This file describes the CCS Server-side handler
interface.  A CCS handler is just a CMI handler,
but it can use the CcsSendReply function.
*/

#ifndef CONV_CCS_H
#define CONV_CCS_H

#include "converse.h"
#include "sockRoutines.h"

#ifdef __cplusplus
extern "C" {
#endif

/******* Converse Client Server *****/

#define CMK_CCS_VERSION "2"

extern int _ccsHandlerIdx;

typedef struct CcsDelayedReply_struct *CcsDelayedReply;

struct CcsDelayedReply_struct {
	unsigned char val;
};

#if CMK_CCS_AVAILABLE
void CcsRegisterHandler(const char *id, CmiHandler fn);

void CcsInit(char **argv);
int CcsEnabled(void);
int CcsIsRemoteRequest(void);
void CcsCallerId(skt_ip_t *pip, unsigned int *pport);
void CcsSendReply(int size, const void *reply);
CcsDelayedReply CcsDelayReply(void);
void CcsSendDelayedReply(CcsDelayedReply d,int size, const void *reply);

#else
#define CcsInit(argv) /*empty*/
#define CcsRegisterHandler(x,y) 0
#define CcsEnabled() 0
#define CcsIsRemoteRequest() 0
#define CcsCallerId(x,y)  /*empty*/
#define CcsDelayReply() 0
#define CcsSendReply(s,r) /*empty*/
#define CcsSendDelayedReply(d,s,r); 
#endif

#ifdef __cplusplus
}
#endif
#endif
