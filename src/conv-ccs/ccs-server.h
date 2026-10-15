/*
Converse Client/Server: Server-side interface
Orion Sky Lawlor, 9/13/2000, olawlor@acm.org

CcsServer routines handle the CCS Server socket,
translate CCS requests and replies into the final
network format, and send/recv the actual requests
and replies.

Depending on the situation, this code is called from
conv-host.c, or conv-core.c/conv-ccs.C (for
NODE_0_IS_CONVHOST).  All the routines in this file
should be called from within only one machine&program-- 
the CCS server.  That is, you can't receive a request
on one machine, send the request to another machine,
and send the response from the new machine (because
the reply socket is for the original machine).
*/

#ifndef CCS_SERVER_H
#define CCS_SERVER_H

#include "sockRoutines.h"

#ifdef __cplusplus
extern "C" {
#endif

#if CMK_CCS_AVAILABLE

/*Security attributes for a CCS request*/
typedef struct {
  skt_ip_t ip;/*Source machine (or firewall)*/
  ChMessageInt_t port;
  ChMessageInt_t replySalt;/*Salt value for reply hash*/
  unsigned char auth;/*1-- message authenticated; 0-- no authentication*/
  unsigned char level;/*Security level-- 0 to 255*/
} CcsSecAttr;

/*Used within CCS implementation to identify requestor*/
#define CCS_MAXHANDLER 32 /*Longest possible handler name*/
typedef struct {
  CcsSecAttr     attr; /*Source information*/
  char handler[CCS_MAXHANDLER];/*Handler name for message to follow*/
  ChMessageInt_t pe;/*Dest. processor # (global numbering)*/
  ChMessageInt_t replyFd;/*Send reply back here*/
  ChMessageInt_t len;/*Bytes of message data to follow*/
} CcsImplHeader;

/********* CCS Implementation (not in ccs-server.C) ********/
/*Deliver this request data to the appropriate PE. */
void CcsImpl_netRequest(CcsImplHeader *hdr,const void *reqData);

/*Deliver this reply data to this reply socket.
  The data will eventually be sent to the CCS server.
*/
void CcsImpl_reply(CcsImplHeader *hdr,int repLen,const void *repData);

/*Send any registered clients kill messages before we exit*/
void CcsImpl_kill(void);

/*Convert CCS header & message data into a converse message to handler*/
char *CcsImpl_ccs2converse(const CcsImplHeader *hdr,const void *data,int *ret_len);

/******************* ccs-server.C routines ***************/
/*Make a new Ccs Server socket, on the given port.
Returns the actual port and IP address.
*/
void CcsServer_new(skt_ip_t *ret_ip,int *use_port,const char *securityFile);

/*Get the Ccs Server socket.  This socket can
be added to the rdfs list for calling select().
*/
SOCKET CcsServer_fd(void);

/*Connect to the Ccs Server socket, and 
receive a ccs request from the network.
Returns 1 if a request was successfully received;
0 otherwise.
reqData is allocated with malloc(hdr->len).
*/
int CcsServer_recvRequest(CcsImplHeader *hdr,void **reqData);

/*Send a Ccs reply down the given socket.
Closes the socket afterwards.
*/
void CcsServer_sendReply(CcsImplHeader *hdr,int repBytes,const void *repData);

/*No request will be sent through this socket.
Closes it.
*/
void CcsServer_noReply(CcsImplHeader *hdr);

#if !CMK_USE_LRTS_STDIO
void print_node0(const char *format, va_list args);
#endif

#else /*CCS not available*/

#define CcsServer_new(i,p) /*empty*/
#define CcsServer_fd() SOCKET_ERROR
#define CcsServer_recvReq(h,b) 0
#define CcsServer_sendReply(f,l,d) /*empty*/
#define CcsServer_noReply(f) /*empty*/
#define CcsImpl_kill() /*empty*/
#endif /*CCS available*/

#ifdef __cplusplus
}
#endif
#endif
