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
#include "ccs-server.h" /*for CcsSecAttr*/

#ifdef __cplusplus
extern "C" {
#endif

/******* Converse Client Server *****/

#define CMK_CCS_VERSION "2"

extern int _ccsHandlerIdx;

#if CMK_CCS_AVAILABLE

typedef struct CcsDelayedReply_struct {
	CcsSecAttr     attr; /*Source information*/
	ChMessageInt_t replyFd;/*Send reply back here*/
} CcsDelayedReply;

/**
 * Backward compatability routine: register a regular converse-style handler
 * to receive CCS requests.  The requests will arrive as a Converse message,
 * with a (useless) converse header.
 */
void CcsRegisterHandler(const char *ccs_handlername, CmiHandler fn);

/**
 * Register a real Ccs handler function to receive these CCS requests. 
 * The requests will arrive as a flat, readonly buffer.
 */
typedef void (*CcsHandlerFn)(void *userPtr,int reqLen,const void *reqData);
void CcsRegisterHandlerFn(const char *ccs_handlername, CcsHandlerFn fn, void *userPtr);

void CcsInit(char **argv);
int CcsEnabled(void);
int CcsIsRemoteRequest(void);
void CcsCallerId(skt_ip_t *pip, unsigned int *pport);
void CcsSendReply(int replyLen, const void *replyData);
CcsDelayedReply CcsDelayReply(void);
void CcsSendDelayedReply(CcsDelayedReply d,int replyLen, const void *replyData);

#else
typedef void *CcsDelayedReply;
#define CcsInit(argv) /*empty*/
#define CcsRegisterHandler(x,y) 0
#define CcsRegisterHandlerFn(x,y,p) 0
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
