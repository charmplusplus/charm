/*
Converse Client/Server: Server-side interface
Orion Sky Lawlor, 9/11/2000, olawlor@acm.org

This file describes the under-the-hood implementation
of the CCS Server.  Here's where Ccs requests from the
network are actually received.
*/
#include <stdio.h>
#include "conv-ccs.h"
#include "ccs-server.h"

#if CMK_CCS_AVAILABLE

#define CCSDBG(x) /*printf x*/

static SOCKET ccs_server_fd=SOCKET_ERROR;/*CCS request socket*/

/*Make a new Ccs Server socket, on the given port.
Returns the actual port and IP address.
*/
void CcsServer_new(int *ret_ip,int *use_port)
{
  unsigned int ip;
  unsigned int port=0;if (use_port!=NULL) port=*use_port;
  skt_init();
  ip=skt_my_ip();
  ccs_server_fd=skt_server(&port);
  printf("ccs: %s\nccs: Server IP = %u, Server port = %u $\n", 
           CMK_CCS_VERSION, ip, port);
  fflush(stdout);
  if (ret_ip!=NULL) *ret_ip=ip;
  if (use_port!=NULL) *use_port=port;
}

/*Get the Ccs Server socket.  This socket can
be added to the rdfs list for calling select().
*/
SOCKET CcsServer_fd(void) {return ccs_server_fd;}

/*Connect to the Ccs Server socket, and 
receive a ccs request from the network.
Returns 1 if a request was successfully received.
reqData is allocated with malloc(hdr->len).
*/
int CcsServer_recvRequest(CcsImplHeader *hdr,void **reqData) 
{
  CcsMessageHeader req;/*CCS header, from requestor*/
  unsigned int ip,port;
  int pe,reqBytes;
  SOCKET fd;
  CCSDBG(("CCS Receiving connection...\n"));
  fd=skt_accept(ccs_server_fd,&ip,&port);
  CCSDBG(("CCS   Connected to IP=%d, port=%d...\n",ip,port));
  skt_recvN(fd,(char *)&req,sizeof(req));
  pe=ChMessageInt(req.pe);
  reqBytes=ChMessageInt(req.len);
  CCSDBG(("CCS   Got %d-byte request for handler '%s'\n",reqBytes,req.handler));
  
  /*Fill out the internal CCS header*/
  CcsImplHeader_new(req.handler,pe,ip,port,fd,reqBytes,hdr);

  /*Grab the user data portion of the message*/
  if (reqBytes<=0) *reqData=NULL;
  else {
    *reqData=(char *)malloc(reqBytes);
    skt_recvN(fd,*reqData,reqBytes);
  }
  CCSDBG(("CCS   Got all %d data bytes for request.\n",reqBytes));
  return 1;
}

static int reply_abortFn(int code,const char *msg) {
	/*Just ignore bad replies-- just indicates a client has died*/
	fprintf(stderr,"CCS  ABORT called during reply-- ignoring\n");
	CCSDBG(("CCS  ABORT called during reply-- ignoring\n"));
	return -1;
}

/*Send a Ccs reply down the given socket.
Closes the socket afterwards.
*/
void CcsServer_sendReply(SOCKET fd,int repBytes,const void *repData)
{
  ChMessageInt_t len=ChMessageInt_new(repBytes);
  skt_abortFn old=skt_set_abort(reply_abortFn);
  CCSDBG(("CCS   Sending %d bytes of reply data\n",repBytes));
  skt_sendN(fd,(const char *)&len,sizeof(len));
  skt_sendN(fd,repData,repBytes);
  skt_close(fd);
  CCSDBG(("CCS Reply socket closed.\n"));
  skt_set_abort(old);
}

/*Build a new CcsImplHeader*/
void CcsImplHeader_new(char *handler,
		       int pe,int ip,int port,
		       SOCKET replyFd,int userBytes,
		       CcsImplHeader *imp)
{
  strncpy(imp->handler,handler,CCS_MAXHANDLER);
  imp->pe=ChMessageInt_new(pe);
  imp->ip=ChMessageInt_new(ip);
  imp->port=ChMessageInt_new(port);
  imp->replyFd=ChMessageInt_new(replyFd);
  imp->len=ChMessageInt_new(userBytes);
}

#endif /*CMK_CCS_AVAILABLE*/


