/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <windows.h>
#include <winsock.h>
#include <process.h>
#include <io.h>
#include <time.h>
#include <direct.h>

#include <stdio.h>
#include <ctype.h>
#include <fcntl.h>
#include <errno.h>

#include <stdlib.h>
#include <signal.h>

#include <sys/types.h>
#include <sys/stat.h>


#if 0
/*TESTING: OSL, 11/3/1999
I think we're flipping between host and network byte order too
many times-- we should just stick to one representation.
*/
#define htonl(x) (x)
#define htons(x) (x)
#define ntohl(x) (x)
#define ntohs(x) (x)

#endif

/**************************************************************************
 *
 * SKT - socket routines
 *
 * Uses Module: SCHED  [implicitly TIMEVAL, QUEUE, THREAD]
 *
 *
 * unsigned int skt_ip()
 *
 *   - returns the IP address of the current machine.
 *
 * void skt_server(unsigned int *pip, unsigned int *ppo, unsigned int *pfd)
 *
 *   - create a tcp server socket.  Performs the whole socket/bind/listen
 *     procedure.  Returns the IP address of the socket (eg, the IP of the
 *     current machine), the port of the socket, and the file descriptor.
 *
 * void skt_accept(int src,
 *                 unsigned int *pip, unsigned int *ppo, unsigned int *pfd)
 *
 *   - accepts a connection to the specified socket.  Returns the
 *     IP of the caller, the port number of the caller, and the file
 *     descriptor to talk to the caller.
 *
 * int skt_connect(unsigned int ip, int port)
 *
 *   - Opens a connection to the specified server.  Returns a socket for
 *     communication.
 *
 *
 **************************************************************************/

unsigned int skt_ip()
{
  static unsigned int ip = 0;
  struct hostent *hostent;
  char hostname[100];
  if (ip==0) {
    if (gethostname(hostname, 99)<0) ip=0x7f000001;
    hostent = gethostbyname(hostname);
    if (hostent == 0) return 0x7f000001;
    ip = htonl(*((int *)(hostent->h_addr_list[0])));
  }
  return ip;
}

void skt_server(pip,ppo,pfd)
    unsigned int *pip;
    unsigned int *ppo;
    unsigned int *pfd;
{
  int fd= -1;
  int ok, len;
  struct sockaddr_in addr;
  
  fd = socket(PF_INET, SOCK_STREAM, 0);
  if (fd < 0) { perror("socket"); exit(1); }
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(*ppo);
  ok = bind(fd, (struct sockaddr *)&addr, sizeof(addr));
  if (ok < 0) { perror("bind"); exit(1); }
  ok = listen(fd,5);
  if (ok < 0) { perror("listen"); exit(1); }
  len = sizeof(addr);
  ok = getsockname(fd, (struct sockaddr *)&addr, &len);
  if (ok < 0) { perror("getsockname"); exit(1); }

  *pfd = fd;
  *pip = skt_ip();
  //*ppo = ntohs(addr.sin_port);
}

void skt_accept(src,pip,ppo,pfd)
    int src;
    unsigned int *pip;
    unsigned int *ppo;
    unsigned int *pfd;
{
  int i, fd, ok;
  struct sockaddr_in remote;
  i = sizeof(remote);
 acc:
  fd = accept(src, (struct sockaddr *)&remote, &i);

  if ((fd<0)&&(errno==WSAEINTR)) goto acc;
  if ((fd<0)&&(errno==WSAEMFILE)) goto acc;
  //if ((fd<0)&&(errno==EPROTO)) goto acc;
  if (fd<0) { perror("accept");}
  *pip=htonl(remote.sin_addr.s_addr);
  *ppo=htons(remote.sin_port);
  *pfd=fd;
}

int skt_connect(ip, port)
    unsigned int ip; int port;
{
  struct sockaddr_in remote; short sport=port;
  int fd, ok, len;
    
  /* create an address structure for the server */ 
  memset(&remote, 0, sizeof(remote));
  remote.sin_family = AF_INET;
  remote.sin_port = htons(sport);
  remote.sin_addr.s_addr = htonl(ip);
    
 sock:
  fd = socket(AF_INET, SOCK_STREAM, 0);
  if ((fd<0)&&(errno=EMFILE)) goto sock;
  if (fd < 0) return -1;
  
 conn:
  ok = connect(fd, (struct sockaddr *)&(remote), sizeof(remote));
  if (ok<0) {
    switch (errno) {
    case WSAEADDRINUSE: _close(fd); goto sock;
    default: return -1;
    }
  }
  return fd;
}
