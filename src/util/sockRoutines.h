/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**************************************************************************
 *
 * SKT - simple TCP and UDP socket routines.  
 *  All IP addresses and sockets are taken and returned 
 *  in *host* byte order.  This means you can hardcode port
 *  numbers and IP addresses in the code normally, and they
 *  will be properly translated everywhere.
 *
 *  SOCKET is just a #define for "unsigned int".
 *
 *  Errors are handled in the library by calling a user-overridable
 *  abort function.
 * 
 * unsigned long skt_my_ip(void)
 *   - return the IP address of the current machine.
 *
 * unsigned long skt_lookup_ip(const char *name)
 *   - return the IP address of the given machine.
 *     Returns 0 on failure.
 *
 * SOCKET skt_datagram(unsigned int *port, int bufsize)
 *
 *   - creates a UDP datagram socket on the given port.  
 *     Performs the whole socket/bind/getsockname procedure.  
 *     Returns the actual port of the socket and
 *     the file descriptor.  Bufsize, if nonzero, controls the amount
 *     of buffer space the kernel sets aside for the socket.
 *
 * SOCKET skt_server(unsigned int *port)
 *
 *   - create a TCP server socket on the given port (0 for any port).  
 *     Performs the whole socket/bind/listen procedure.  
 *     Returns the actual port of the socket and the file descriptor.
 *
 * SOCKET skt_accept(SOCKET src_fd,unsigned int *pip, unsigned int *port)
 *
 *   - accepts a TCP connection to the specified server socket.  Returns the
 *     IP of the caller, the port number of the caller, and the file
 *     descriptor to talk to the caller.
 *
 * SOCKET skt_connect(unsigned int ip, int port, int timeout)
 *
 *   - Opens a TCP connection to the specified server.  Returns a socket for
 *     communication.
 *
 * void skt_close(SOCKET fd)
 *   - Finishes communication on and closes the given socket.
 *
 * int skt_select1(SOCKET fd,int msec)
 *   - Call select on the given socket, returning as soon as
 *     the socket can recv or accept, or (failing that) in the given
 *     number of milliseconds.  Returns 0 on timeout; 1 on readable.
 *
 * int skt_recvN(SOCKET fd,      void *buf,int nBytes)
 * int skt_sendN(SOCKET fd,const void *buf,int nBytes)
 *   - Blocking send/recv nBytes on the given socket.
 *     Retries if possible (e.g., if interrupted), but aborts 
 *     on serious errors.  Returns zero or an abort code.
 *
 * void skt_set_idle(idleFunc f)
 *   - Specify a routine to be called while waiting for the network.
 *     Replaces any previous routine.
 * 
 * void skt_set_abort(abortFunc f)
 *   - Specify a routine to be called when an unrecoverable
 *     (i.e., non-transient) socket error is encountered.
 *     The default is to log the message to stderr and call exit(1).
 *
 **************************************************************************/
#ifndef __SOCK_ROUTINES_H
#define __SOCK_ROUTINES_H

#include "conv-mach.h" /*<- for CMK_NO_SOCKETS*/

#ifdef CMK_NO_SOCKETS
#define SOCKET int
#define SOCKET_ERROR (-1)
#define INVALID_SOCKET (SOCKET)(~0)
#else /*Use actual sockets*/

/*Preliminaries*/
#if defined(_WIN32) && ! defined(__CYGWIN__)
  /*For windows systems:*/
#include <winsock.h>
static void sleep(int secs) {Sleep(1000*secs);}

void skt_init(void);/*Is a function*/
#else
  /*For non-windows (UNIX) systems:*/
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#ifndef SOCKET
#define SOCKET int
#define INVALID_SOCKET (SOCKET)(~0)
#define SOCKET_ERROR (-1)
#endif /*def SOCKET*/

#define skt_init() /*not needed on UNIX systems*/
#endif /*WIN32*/


/*Error and idle handling*/
typedef void (*skt_idleFn)(void);
typedef int (*skt_abortFn)(int errCode,const char *msg);
void skt_set_idle(skt_idleFn f);
skt_abortFn skt_set_abort(skt_abortFn f);

/*DNS*/
unsigned long skt_my_ip(void);
unsigned long skt_lookup_ip(const char *name);
struct sockaddr_in skt_build_addr(unsigned int IP,unsigned int port);

/*UDP*/
SOCKET skt_datagram(unsigned int *port, unsigned int bufsize);

/*TCP*/
SOCKET skt_server(unsigned int *port);
SOCKET skt_accept(SOCKET src_fd, unsigned int *pip, unsigned int *port);
SOCKET skt_connect(unsigned int ip, int port, int timeout);

/*Utility*/
void skt_close(SOCKET fd);
int skt_select1(SOCKET fd,int msec);

/*Blocking Send/Recv*/
int skt_sendN(SOCKET hSocket,const void *pBuff,int nBytes);
int skt_recvN(SOCKET hSocket,      void *pBuff,int nBytes);

#endif /*!CMK_NO_SOCKETS*/

/***********************************
Conv-host messages: these are a simple
binary format message, with the usual header,
then data arrangement.

A fundamental data type is a ChMessageInt_t,
a simple 4-byte big-endian (network byte order)
integer.  Routines are provided to read/write
these integers on all platforms, regardless of 
endian-ness or native integer size.

 ChMessage_recv reads a ChMessage on a socket.
The ChMessage->data field is allocated to contain
the entire message, and the header is filled out
with the received fields.  You may keep or dispose 
of the message memory with ChMessage_free.

 ChMessageHeader_new fills out the fields of a header--
no allocation is done.

 ChMessage_new fills out the header and allocates a 
data area of the given size.

 ChMessage_send delivers the given ChMessage to a
socket.  You are still responsible for the ChMessage
memory (use ChMessage_free).  If you prefer, you may
receive sizeof(ChMessageHeader) header bytes, then 
header->len data bytes on any socket yourself.
*/
typedef struct {
  unsigned char data[4];/*4-byte, big-endian integer*/
} ChMessageInt_t;
ChMessageInt_t ChMessageInt_new(unsigned int src); /*Convert integer to bytes*/
unsigned int ChMessageInt(ChMessageInt_t src); /*Convert bytes to integer*/

#define CH_TYPELEN 12 /*Maximum length for the message type field*/
typedef struct ChMessageHeader {
  ChMessageInt_t len; /*Length of message to follow (not incl. header)*/
  char type[CH_TYPELEN];/*Kind of message to follow:
     (zero-terminated ASCII string) 
     "getinfo" -- return a list of node IPs and control ports
     "req" -- a CCS message
     */
} ChMessageHeader;

typedef struct ChMessage {
  ChMessageHeader header;
  int len; /*Length of message data below*/
  char *data; /*Pointer to heap-allocated data*/
} ChMessage;
int ChMessage_recv(SOCKET fd,ChMessage *dst);
void ChMessage_free(ChMessage *doomed);
void ChMessageHeader_new(const char *type,unsigned int len,
		   ChMessageHeader *dst);
void ChMessage_new(const char *type,unsigned int len,
		   ChMessage *dst);
int ChMessage_send(SOCKET fd,const ChMessage *src); /*You must free after send*/

/******* CCS Message type (included here for convenience) *******/
#define CCS_HANDLERLEN 32 /*Maximum length for the handler field*/
typedef struct {
  ChMessageInt_t len;/*Length of user data to follow header*/
  ChMessageInt_t pe;/*Destination processor number*/
  char handler[CCS_HANDLERLEN];/*Handler name for message to follow*/
} CcsMessageHeader;

#endif /*SOCK_ROUTINES_H*/

