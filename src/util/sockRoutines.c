/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#if CMK_SCYLD
#include <sys/bproc.h>
#endif

#include "sockRoutines.h"
#ifndef CMK_NO_SOCKETS /*<- for ASCI Red*/

/*Just print out error message and exit*/
static int default_skt_abort(int code,const char *msg)
{
  fprintf(stderr,"Fatal socket error: code %d-- %s\n",code,msg);
  exit(1);
  return -1;
}

static skt_idleFn idleFunc=NULL;
static skt_abortFn skt_abort=default_skt_abort;
void skt_set_idle(skt_idleFn f) {idleFunc=f;}
skt_abortFn skt_set_abort(skt_abortFn f) 
{
	skt_abortFn old=skt_abort;
	skt_abort=f;
	return old;
}


#if defined(_WIN32) && !defined(__CYGWIN__) /*Windows systems:*/
static void doCleanup(void)
{ WSACleanup();}
static int skt_inited=0;
/*Initialization routine (Windows only)*/
void skt_init(void)
{
  WSADATA WSAData;
  WORD version=0x0002;
  if (skt_inited) return;
  skt_inited=1;
  WSAStartup(version, &WSAData);
  atexit(doCleanup);
}

void skt_close(SOCKET fd)
{
	closesocket(fd);
}
#else /*UNIX Systems:*/
void skt_close(SOCKET fd)
{
	close(fd);
}
#endif

/*Called when a socket or select routine returns
an error-- determines how to respond.
Return 1 if the last call was interrupted
by, e.g., an alarm and should be retried.
*/
static int skt_should_retry(void)
{
	int isinterrupt=0,istransient=0;
#if defined(_WIN32) && !defined(__CYGWIN__) /*Windows systems-- check Windows Sockets Error*/
	int err=WSAGetLastError();
	if (err==WSAEINTR) isinterrupt=1;
	if (err==WSATRY_AGAIN||err==WSAECONNREFUSED)
		istransient=1;
#else /*UNIX systems-- check errno*/
	int err=errno;
	if (err==EINTR) isinterrupt=1;
	if (err==EAGAIN||err==ECONNREFUSED)
		istransient=1;
#endif
	if (isinterrupt) {
		/*We were interrupted by an alarm.  Schedule, then retry.*/
		if (idleFunc!=NULL) idleFunc();
	}
	else if (istransient)
	{ /*A transient error-- idle a while, then try again later.*/
		if (idleFunc!=NULL) idleFunc();
		else sleep(1);
	}
	else 
		return 0; /*Some unrecognized problem-- abort!*/
	return 1;/*Otherwise, we recognized it*/
}

/*Sleep on given read socket until msec or readable*/
int skt_select1(SOCKET fd,int msec)
{
  int sec=msec/1000;
  fd_set  rfds;
  struct timeval tmo;
  int  secLeft=sec;
  int            begin, nreadable;
  
  FD_ZERO(&rfds);
  FD_SET(fd, &rfds);

  if (msec>0) begin = time(0);
  do
  {
    tmo.tv_sec=secLeft;
    tmo.tv_usec = (msec-1000*sec)*1000;
    nreadable = select(FD_SETSIZE, &rfds, NULL, NULL, &tmo);
    
    if (nreadable < 0) {
		if (skt_should_retry()) continue;
		else skt_abort(93200,"Fatal error in select");
	}
    if (nreadable >0) return 1; /*We gotta good socket*/
  }
  while(msec>0 && (secLeft = sec - (time(0) - begin)));

  return 0;/*Timed out*/
}

unsigned long skt_my_ip(void)
{  
  static unsigned long ip = 0;/*Cached IP address*/
  unsigned long self_ip=0x7F000001u;/*Host byte order 127.0.0.1*/
  char hostname[1000];
  
  if (ip==0) 
  {
    if (gethostname(hostname, 999)==0) 
		ip=skt_lookup_ip(hostname);
  }
  if (ip==0) ip=self_ip;
  return ip;
}

unsigned long skt_lookup_ip(const char *name)
{
  const unsigned int inval=0xffFFffFFu;
  unsigned long ret;
  ret=inet_addr(name);/*Try dotted decimal*/
  if (ret!=inval && ntohl(ret)!=inval) 
	  return ntohl(ret);
  else {/*Try a DNS lookup*/
    struct hostent *h = gethostbyname(name);
    unsigned long ip;
    if (h==0) return 0;
    ip=ntohl(*((unsigned int *)(h->h_addr_list[0])));
    return ip;
  }
}

/* these 2 functions will return the inner node IP, special for
   Linux Scyld.  G. Zheng 
*/
unsigned long skt_innode_my_ip(void)
{  
  static unsigned long ip = 0;/*Cached IP address*/
  unsigned long self_ip=0x7F000001u;/*Host byte order 127.0.0.1*/
  char hostname[1000];
  
#if CMK_SCYLD
  /* on Scyld, the hostname is just the node number */
  sprintf(hostname, "%d", bproc_currnode());
  ip=skt_innode_lookup_ip(hostname);
#else
  ip = skt_my_ip();
#endif
  return ip;
}

unsigned long skt_innode_lookup_ip(const char *name)
{
#if CMK_SCYLD
  struct sockaddr_in addr;
  int len = sizeof(struct sockaddr_in);
  if (-1 == bproc_nodeaddr(atoi(name), &addr, &len)) {
    return 0;
  }
  else {
    return ntohl(addr.sin_addr.s_addr);
  }
#else
  return skt_lookup_ip(name);
#endif
}


struct sockaddr_in skt_build_addr(unsigned int IP,unsigned int port)
{
  struct sockaddr_in ret={0};
  ret.sin_family=AF_INET;
  ret.sin_port = htons((short)port);
  ret.sin_addr.s_addr = htonl(IP);
  return ret;
}

SOCKET skt_datagram(unsigned int *port, unsigned int bufsize)
{  
  int connPort=(port==NULL)?0:*port;
  struct sockaddr_in addr=skt_build_addr(INADDR_ANY,connPort);
  int                len;
  SOCKET             ret;
  
retry:
  ret = socket(AF_INET,SOCK_DGRAM,0);
  if (ret == SOCKET_ERROR) {
    if (skt_should_retry()) goto retry;  
    return skt_abort(93490,"Error creating datagram socket.");
  }
  if (bind(ret, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR)
	  return skt_abort(93491,"Error binding datagram socket.");
  
  len = sizeof(addr);
  if (getsockname(ret, (struct sockaddr *)&addr , &len))
	  return skt_abort(93492,"Error getting address on datagram socket.");

  if (bufsize) 
  {
    len = sizeof(int);
    if (setsockopt(ret, SOL_SOCKET , SO_RCVBUF , (char *)&bufsize, len) == SOCKET_ERROR) 
		return skt_abort(93495,"Error on RCVBUF sockopt for datagram socket.");
    if (setsockopt(ret, SOL_SOCKET , SO_SNDBUF , (char *)&bufsize, len) == SOCKET_ERROR) 
		return skt_abort(93496,"Error on SNDBUF sockopt for datagram socket.");
  }
  
  if (port!=NULL) *port = ntohs(addr.sin_port);
  return ret;
}

SOCKET skt_server(unsigned int *port)
{
  SOCKET             ret;
  int                len;
  int connPort=(port==NULL)?0:*port;
  struct sockaddr_in addr=skt_build_addr(0,connPort);
  
retry:
  ret = socket(PF_INET, SOCK_STREAM, 0);
  
  if (ret == SOCKET_ERROR) {
    if (skt_should_retry()) goto retry;
    else return skt_abort(93483,"Error creating server socket.");
  }
  if (bind(ret, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) 
	  return skt_abort(93484,"Error binding server socket.");
  if (listen(ret,5) == SOCKET_ERROR) 
	  return skt_abort(93485,"Error listening on server socket.");
  len = sizeof(addr);
  if (getsockname(ret, (struct sockaddr *)&addr, &len) == SOCKET_ERROR) 
	  return skt_abort(93486,"Error getting name on server socket.");

  if (port!=NULL) *port = ntohs(addr.sin_port);
  return ret;
}

SOCKET skt_accept(SOCKET src_fd, unsigned int *pip, unsigned int *port)
{
  int len;
  struct sockaddr_in addr={0};
  SOCKET ret;
  len = sizeof(addr);
retry:
  ret = accept(src_fd, (struct sockaddr *)&addr, &len);
  if (ret == SOCKET_ERROR) {
    if (skt_should_retry()) goto retry;
    else return skt_abort(93523,"Error in accept.");
  }
  
  if (port!=NULL) *port=ntohs(addr.sin_port);
  if (pip!=NULL) *pip=ntohl(addr.sin_addr.s_addr);
  return ret;
}


SOCKET skt_connect(unsigned int ip, int port, int timeout)
{
  struct sockaddr_in addr=skt_build_addr(ip,port);
  int                ok, begin;
  SOCKET             ret;
  
  begin = time(0);
  while (time(0)-begin < timeout) 
  {
    ret = socket(AF_INET, SOCK_STREAM, 0);
    if (ret==SOCKET_ERROR) 
    {
	  if (skt_should_retry()) continue;  
      else return skt_abort(93512,"Error creating socket");
    }
    ok = connect(ret, (struct sockaddr *)&(addr), sizeof(addr));
    if (ok != SOCKET_ERROR) 
	  return ret;/*Good connect*/
	else { /*Bad connect*/
	  skt_close(ret);
	  if (skt_should_retry()) continue;
	  else return skt_abort(93515,"Error connecting to socket\n");
    }
  }
  /*Timeout*/
  if (timeout==60)
     return skt_abort(93517,"Timeout in socket connect\n");
  return INVALID_SOCKET;
}

int skt_recvN(SOCKET hSocket,void *buff,int nBytes)
{
  int nLeft,nRead;
  char *pBuff=(char *)buff;

  nLeft = nBytes;
  while (0 < nLeft)
  {
    if (0==skt_select1(hSocket,60*1000))
	return skt_abort(93610,"Timeout on socket recv!");
    nRead = recv(hSocket,pBuff,nLeft,0);
    if (nRead<=0)
    {
       if (nRead==0) return skt_abort(93620,"Socket closed before recv.");
       if (skt_should_retry()) continue;/*Try again*/
       else return skt_abort(93650+hSocket,"Error on socket recv!");
    }
    else
    {
      nLeft -= nRead;
      pBuff += nRead;
    }
  }
  return 0;
}

int skt_sendN(SOCKET hSocket,const void *buff,int nBytes)
{
  int nLeft,nWritten;
  const char *pBuff=(const char *)buff;
  
  nLeft = nBytes;
  while (0 < nLeft)
  {
    nWritten = send(hSocket,pBuff,nLeft,0);
    if (nWritten<=0)
    {
          if (nWritten==0) return skt_abort(93720,"Socket closed before send.");
	  if (skt_should_retry()) continue;/*Try again*/
	  else return skt_abort(93700+hSocket,"Error on socket send!");
    }
    else
    {
      nLeft -= nWritten;
      pBuff += nWritten;
    }
  }
  return 0;
}


/***********************************************
  Routines for manipulating simple binary messages,
 e.g., to/from conv-host.

 A conv-host message with header has the following format 
on the network: 
ChMessage---------------------------------------------
 /--ChMessageHeader----------------------------     ^
 |12 bytes  |   Message type field           ^      |
 |          |   (ASCII, Null-terminated)     |      |
 +-----------------------------------     24 bytes  |
 | 4 bytes  |   Message data length d        |      |
 |          |   (big-endian binary integer)  |      |
 +-----------------------------------        |      |
 | 4 bytes  |   Return IP address            |      |
 | 4 bytes  |   Return TCP port number       |      |
 |          |   (big-endian binary integers) v  24+d bytes
 \---------------------------------------------     |
 d bytes  |   User data                             v
------------------------------------------------------

For completeness, a big-endian (network byte order) 4 byte 
integer has this format on the network:
ChMessageInt---------------------------------
  1 byte | Most significant byte  (&0xff000000; <<24)
  1 byte | More significant byte  (&0x00ff0000; <<16)
  1 byte | Less significant byte  (&0x0000ff00; <<8)
  1 byte | Least significant byte (&0x000000ff; <<0)
----------------------------------------------
*/

ChMessageInt_t ChMessageInt_new(unsigned int src)
{ /*Convert integer to bytes*/
  int i; ChMessageInt_t ret;
  for (i=0;i<4;i++) ret.data[i]=(unsigned char)(src>>(8*(3-i)));
  return ret;
}
unsigned int ChMessageInt(ChMessageInt_t src)
{ /*Convert bytes to integer*/
  int i; unsigned int ret=0;
  for (i=0;i<4;i++) {ret<<=8;ret+=src.data[i];}
  return ret;
}

int ChMessage_recv(SOCKET fd,ChMessage *dst)
{
  /*Get the binary header*/
  if (0!=skt_recvN(fd,(char *)&dst->header,sizeof(dst->header))) return -1;
  /*Allocate a recieve buffer*/
  dst->len=ChMessageInt(dst->header.len);
  dst->data=(char *)malloc(dst->len);
  /*Get the actual data*/
  if (0!=skt_recvN(fd,dst->data,dst->len)) return -1;
  return 0;
}
void ChMessage_free(ChMessage *doomed)
{
  free(doomed->data);
  strncpy(doomed->header.type,"Free'd",CH_TYPELEN);
  doomed->data=NULL;
  doomed->len=-1234;
}
void ChMessageHeader_new(const char *type,unsigned int len,
		   ChMessageHeader *dst)
{
  dst->len=ChMessageInt_new(len);
  if (type==NULL) type="default";
  strncpy(dst->type,type,CH_TYPELEN);
}
void ChMessage_new(const char *type,unsigned int len,
		   ChMessage *dst)
{
  ChMessageHeader_new(type,len,&dst->header);
  dst->len=len;
  dst->data=(char *)malloc(dst->len);
}
int ChMessage_send(SOCKET fd,const ChMessage *src)
{
  if (0!=skt_sendN(fd,&src->header,sizeof(src->header))) return -1;
  if (0!=skt_sendN(fd,src->data,src->len)) return -1;
  return 0;
} /*You must free after send*/

#endif /*!CMK_NO_SOCKETS*/



