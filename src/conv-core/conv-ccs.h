/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/*This file describes the CCS Server-side handler
interface.  A CCS handler is just a CMI handler,
but it can use the functions 
*/

#ifndef CONV_CCS_H
#define CONV_CCS_H

#include "converse.h"

#ifdef __cplusplus
extern "C" {
#endif

/******* Converse Client Server *****/

#define CMK_CCS_VERSION "2"

#if CMK_CCS_AVAILABLE
void CcsUseHandler(char *id, int hdlr);
int CcsRegisterHandler(char *id, CmiHandler fn);

int CcsEnabled(void);
int CcsIsRemoteRequest(void);
void CcsCallerId(unsigned int *pip, unsigned int *pport);

void CcsSendReply(int size, const char *reply);

#else
#define CcsInit() /*empty*/
#define CcsUseHandler(x,y) /*empty*/ 
#define CcsRegisterHandler(x,y) 0
#define CcsEnabled() 0
#define CcsIsRemoteRequest() 0
#define CcsCallerId(x,y)  /*empty*/
#define CcsSendReply(s,r) /*empty*/ 
#endif

#ifdef __cplusplus
}
#endif
#endif
