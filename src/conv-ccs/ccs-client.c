/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/
/*
Converse Client-Server:
  Lets you execute, from an arbitrary program on the network, 
pre-registered "handlers" in a running converse program.  
Also allows you to recv replies from the handlers.  
All requests and replies consist of user-defined binary data.

  This file provides the client interface.

CCS Protocol spec:

 A CCS request message asks a running Converse program to
execute a pre-registered "handler" routine.  You send the
request directly to conv-host's CCS server port.
The request, with header, has the following format on the
network: 
Ccs Message----------------------------------------------
 /--CcsMessageHeader---------------------------       ^
 | 4 bytes  |   Message data length d         ^       |
 | 4 bytes  |   Dest. processor number        |       |
 |          |   (big-endian binary integers)  |   40+d bytes
 +-----------------------------------      40 bytes   |
 |32 bytes  |   CCS Handler name              |       |
 |          |   (ASCII, Null-terminated)      v       |
 \---------------------------------------------       |
    d bytes |   User data (passed to handler)         v
-------------------------------------------------------

 A CCS reply message (if any) comes back on the request socket,
and has only a length header:
CCS Reply ----------------------------------
 | 4 bytes  |   Message data length d        
 |          |   (big-endian binary integer)  
 +----------------------------------------- 
 | d bytes  |   User data                   
--------------------------------------------

 */
#include "ccs-client.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*Include the socket and message interface routines
  here *whole*, which keeps client linking simple.*/
#include "sockRoutines.c"
#include "ccs-auth.c"

#define DEBUGF(x) printf x

/*Parse list of nodes given to us by conv-host.
*/
static void parseInfo(CcsServer *svr,const char *data)
{
  /*Data conv-host sends us is just a big list of integers*/
  const ChMessageInt_t *d=(const ChMessageInt_t *)data;
  int i,index=0; /*Current offset in above array*/

  svr->numNodes=ChMessageInt(d[index++]);
  svr->numProcs = (int *) malloc(svr->numNodes * sizeof(int));
  svr->numPes = 0;
  for(i=0;i<svr->numNodes;i++)
    svr->numPes+=svr->numProcs[i]=ChMessageInt(d[index++]);
}

static void printSvr(CcsServer *svr)
{
  char ipBuf[200];
  int i;
  DEBUGF(("hostIP: %d\n", skt_print_ip(ipBuf,svr->hostIP)));
  DEBUGF(("hostPort: %d\n", svr->hostPort));
  DEBUGF(("authentication: %d\n", svr->isAuth));
  DEBUGF(("replyFd: %d\n", svr->replyFd));
  DEBUGF(("numNodes: %d\n", svr->numNodes));
  DEBUGF(("numPes: %d\n", svr->numPes));
  for(i=0;i<svr->numNodes;i++) {
    DEBUGF(("Node[%d] has %d processors\n",i, svr->numProcs[i]));
  }
}

/* Authenticate with this CCS server.
 */
static const char *CcsImpl_authInit(SOCKET fd,CcsServer *svr)
{
  struct {
    unsigned char type[4];
    ChMessageInt_t s1;
  } request;
  int s1;
  ChMessageInt_t s2;
  SHA1_hash_t s2hash;
  struct {
    SHA1_hash_t s1hash;
    ChMessageInt_t clientID;
    ChMessageInt_t clientSalt;
  } reply;
  if (fd==-1)
    return "ERROR> Contacting server";
  request.type[0]=0x80; /*SHA-1 authentication*/
  request.type[1]=0x00; /*Version 0*/
  request.type[2]=0x01; /*Request for salt*/
  request.type[3]=svr->level; /*Security level*/
  s1=CCS_RAND_next(&svr->rand);
  request.s1=ChMessageInt_new(s1);
  if (-1==skt_sendN(fd,&request,sizeof(request))) 
    return "ERROR> AuthInit request send";
  if (-1==skt_recvN(fd,&s2,sizeof(s2)))  
    return "ERROR> AuthInit challenge recv";
  CCS_AUTH_hash(&svr->key,ChMessageInt(s2),NULL,&s2hash);
  if (-1==skt_sendN(fd,&s2hash,sizeof(s2hash))) 
    return "ERROR> AuthInit challenge reply send";
  if (-1==skt_recvN(fd,&reply,sizeof(reply))) 
    return "ERROR> AuthInit final recv (authentication failure?)";
  if (CCS_AUTH_differ(&svr->key,s1,NULL,&reply.s1hash))
    return "ERROR> AuthInit server key does not match";
  
  svr->clientID=ChMessageInt(reply.clientID);
  svr->clientSalt=ChMessageInt(reply.clientSalt);
  return 0;
}


/**
 * Converse Client-Server Module: Client Side
 */

int CcsConnect(CcsServer *svr, const char *host, int port,const CcsSec_secretKey *key){
    return CcsConnectWithtimeout(svr, host, port, key, 120);
}

int CcsConnectWithtimeout(CcsServer *svr, const char *host, int port,
	const CcsSec_secretKey *key, int timeout) 
{
	skt_init();
    return CcsConnectIpWithTimeout(svr,skt_lookup_ip(host),port,key, timeout);
}

int CcsConnectIp(CcsServer *svr, skt_ip_t ip, int port,const CcsSec_secretKey *key){
    return CcsConnectIpWithTimeout(svr, ip, port, key, 120);
}

int CcsConnectIpWithTimeout(CcsServer *svr, skt_ip_t ip, int port,
	const CcsSec_secretKey *key, int timeout)
{
  unsigned int msg_len;char *msg_data;/*Reply message*/
  skt_init();
  svr->hostIP = ip;
  svr->hostPort = port;
  svr->replyFd=INVALID_SOCKET;

  svr->clientID=svr->clientSalt=-1;
  if (key==NULL) 
    svr->isAuth=0;
  else 
  { /*Authenticate with server*/
    SOCKET fd;
    const char *err;
    svr->isAuth=1;
    svr->level=0; /*HACK: hardcoded at security level 0*/
    svr->key=*key;
    CCS_RAND_new(&svr->rand);
    fd=skt_connect(svr->hostIP, svr->hostPort,timeout);

    if (NULL!=(err=CcsImpl_authInit(fd,svr))) {
      fprintf(stderr,"CCS Client error> %s\n",err);
      skt_close(fd);
      return -1;
    }
    skt_close(fd);
  }

  /*Request the parallel machine's node info*/
  if(CcsSendRequestWithTimeout(svr,"ccs_getinfo",0,0,NULL,timeout) == -1){
      fprintf(stderr,"CCS Client Not Alive\n");
      return -1;
  }
  
  /*Wait for conv-host to get back to us*/
  DEBUGF(("Waiting for conv-host to call us back...\n"));

  if(CcsRecvResponseMsg(svr,&msg_len,&msg_data,timeout) == -1){
      fprintf(stderr,"CCS Client Not Alive\n");
      return -1;
  }

  parseInfo(svr,msg_data);
  free(msg_data);
  
  /**/ printSvr(svr);/**/
  return 0;
}

int CcsNumNodes(CcsServer *svr)
{
  return svr->numNodes;
}

int CcsNumPes(CcsServer *svr)
{
  return svr->numPes;
}

int CcsNodeFirst(CcsServer *svr, int node)
{
  int retval=0,i;
  for(i=0;i<node;i++) {
    retval += svr->numProcs[node];
  }
  return retval;
}

int CcsNodeSize(CcsServer *svr,int node)
{
  return svr->numProcs[node];
}

int CcsSendRequest(CcsServer *svr, const char *hdlrID, int pe, unsigned int size, const char *msg){
    return CcsSendRequestWithTimeout(svr, hdlrID, pe, size, msg, 120);
}

int CcsSendRequestWithTimeout(CcsServer *svr, const char *hdlrID, int pe, unsigned int size, const char *msg, int timeout)
{
  CcsMessageHeader hdr;/*CCS request header*/
  hdr.len=ChMessageInt_new(size);
  hdr.pe=ChMessageInt_new(pe);
  strncpy(hdr.handler,hdlrID,CCS_HANDLERLEN);

  /*Connect to conv-host, and send the message */
  svr->replyFd=skt_connect(svr->hostIP, svr->hostPort,timeout);
  if (svr->replyFd==-1) return -1;
  
  if (svr->isAuth==1) 
  {/*Authenticate*/
    struct {
      unsigned char type[4];
      ChMessageInt_t clientID;
      ChMessageInt_t replySalt;
      SHA1_hash_t hash;
      CcsMessageHeader hdr;
    } auth;
    auth.type[0]=0x80; /*SHA-1 authentication*/
    auth.type[1]=0x00; /*Version 0*/
    auth.type[2]=0x00; /*Ordinary message*/   
    auth.type[3]=svr->level; /*Security level*/    
    auth.clientID=ChMessageInt_new(svr->clientID);
    svr->replySalt=CCS_RAND_next(&svr->rand);
    auth.replySalt=ChMessageInt_new(svr->replySalt);
    CCS_AUTH_hash(&svr->key,svr->clientSalt++,
		  &hdr,&auth.hash);
    auth.hdr=hdr;
    if (-1==skt_sendN(svr->replyFd, &auth, sizeof(auth))) return -1;
  }
  else
  {/*No authentication*/
    if (-1==skt_sendN(svr->replyFd, &hdr, sizeof(hdr))) return -1;
  }
  if (-1==skt_sendN(svr->replyFd, msg, size)) return -1;
  /*Leave socket open for reply*/
  return 0;
}

/*Receive and check server reply authentication*/
int CcsImpl_recvReplyAuth(CcsServer *svr)
{
  SHA1_hash_t hash;
  if (!svr->isAuth) return 0;
  if (-1==skt_recvN(svr->replyFd,&hash,sizeof(hash))) return -1;
  if (CCS_AUTH_differ(&svr->key,svr->replySalt,
		  NULL,&hash)) return -1;
  return 0;
}


/*Receive data back from the server. (Arbitrary length response)
*/
int CcsRecvResponseMsg(CcsServer *svr, unsigned int *size,char **newBuf, int timeout)
{
  ChMessageInt_t netLen;
  unsigned int len;  
  SOCKET fd=svr->replyFd;
  if (-1==CcsImpl_recvReplyAuth(svr)) return -1;
  if (-1==skt_recvN(fd,(char *)&netLen,sizeof(netLen))) return -1;
  *size=len=ChMessageInt(netLen);
  *newBuf=(char *)malloc(len);
  if (-1==skt_recvN(fd,(char *)*newBuf,len)) return -1;

  /*Close the connection*/
  skt_close(svr->replyFd);svr->replyFd=-1;
  return len;
}

/*Receive data from the server. (In-place receive)
*/
int CcsRecvResponse(CcsServer *svr,  unsigned int maxsize, char *recvBuffer,int timeout)
{
  ChMessageInt_t netLen;
  unsigned int len;
  SOCKET fd=svr->replyFd;
  if (-1==CcsImpl_recvReplyAuth(svr)) return -1;
  if (-1==skt_recvN(fd,(char *)&netLen,sizeof(netLen))) return -1;
  len=ChMessageInt(netLen);
  if (len>maxsize) 
    {skt_close(fd);return -1;/*Buffer too small*/}
  if (-1==skt_recvN(fd,(char *)recvBuffer,len)) return -1;

  /*Close the connection*/
  skt_close(svr->replyFd);svr->replyFd=-1;
  return len;
}

int CcsProbe(CcsServer *svr)
{
  return skt_select1(svr->replyFd,0);
}

void CcsFinalize(CcsServer *svr)
{
  if (svr->replyFd!=-1) skt_close(svr->replyFd);
}





