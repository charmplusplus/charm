#include <sys/types.h>
#include <sys/socket.h>
#include "ccs.h"
#include <stdio.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netdb.h>
#include <errno.h>


static void zap_newline(char *s)
{
  char *p;
  p = s + strlen(s)-1;
  if (*p == '\n') *p = '\0';
}

/*
 * return IP address for hostname. If hostname=0, return self IP
 */
static unsigned int skt_ip(char *hostname)
{
  unsigned int ip;
  struct hostent *hostent;
  if(strcmp(hostname, "") == 0)
    hostent = gethostent();
  else {
    hostent = gethostbyname(hostname);
  }
  if (hostent == 0) return 0x7f000001;
  ip = htonl(*((int *)(hostent->h_addr_list[0])));

  /*Debugging*/
  /* printf("hostname = %s, IP address = %x\n", hostname, ip); */

  return ip;
}

static void jsleep(int sec, int usec)
{
  int ntimes,i;
  struct timeval tm;

  ntimes = sec*200 + usec/5000;
  for(i=0;i<ntimes;i++) {
    tm.tv_sec = 0;
    tm.tv_usec = 5000;
    while(1) {
      if (select(0,NULL,NULL,NULL,&tm)==0) break;
      if ((errno!=EBADF)&&(errno!=EINTR)) return;
    }
  }
}

/*
 * Create a socket connected to <ip> at port <port>
 */
static int skt_connect(ip, port, seconds)
unsigned int ip; int port; int seconds;
{
  struct sockaddr_in remote; short sport=port;
  int fd, ok, len, retry, begin;
    
  /* create an address structure for the server */
  memset(&remote, 0, sizeof(remote));
  remote.sin_family = AF_INET;
  remote.sin_port = htons(sport);
  remote.sin_addr.s_addr = htonl(ip);
    
  begin = time(0); ok= -1;
  while (time(0)-begin < seconds) {
  sock:
    fd = socket(AF_INET, SOCK_STREAM, 0);
    if ((fd<0)&&((errno==EINTR)||(errno==EBADF))) goto sock;
    if (fd < 0) { perror("socket 3"); exit(1); }
    
  conn:
    ok = connect(fd, (struct sockaddr *)&(remote), sizeof(remote));
    if (ok>=0) break;
    close(fd);
    switch (errno) {
    case EINTR: case EBADF: case EALREADY: break;
    case ECONNREFUSED: jsleep(1,0); break;
    case EADDRINUSE: jsleep(1,0); break;
    case EADDRNOTAVAIL: jsleep(5,0); break;
    default: return -1;
    }
  }
  if (ok<0) return -1;
  return fd;
}

/*
 * Create a server socket
 */
static void skt_server(CcsServer *svr)
{
  int fd= -1;
  int ok, len;
  struct sockaddr_in addr;
  char hostname[100];
 
  fd = socket(PF_INET, SOCK_STREAM, 0);
  if (fd < 0) { perror("socket"); exit(1); }
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  ok = bind(fd, (struct sockaddr *)&addr, sizeof(addr));
  if (ok < 0) { perror("bind"); exit(1); }
  ok = listen(fd,5);
  if (ok < 0) { perror("listen"); exit(1); }
  len = sizeof(addr);
  ok = getsockname(fd, (struct sockaddr *)&addr, &len);
  if (ok < 0) { perror("getsockname"); exit(1); }

  if (gethostname(hostname, 99) < 0) strcpy(hostname, "");
  
  /*Debugging*/
  /* printf("hostname = %s, %d\n", hostname, strlen(hostname)); */

  svr->myFd = fd;
  svr->myIP = skt_ip(hostname);
  svr->myPort = ntohs(addr.sin_port);

  /*Debugging*/
  /* printf("myPort = %d\n", svr->myPort); */
}

static int skt_accept(CcsServer *svr)
{
  int i, fd, ok;
  struct sockaddr_in remote;
  i = sizeof(remote);
 acc:
  fd = accept(svr->myFd, (struct sockaddr *)&remote, &i);
  if ((fd<0)&&(errno==EINTR)) goto acc;
  if ((fd<0)&&(errno==EMFILE)) { usleep(0); goto acc;}
  if ((fd<0)&&(errno==EPROTO)) { goto acc;}
  if (fd<0) { perror("accept"); exit(1); }

  return fd;
}

int my_read(int fd, char *ptr)
{
  static int read_cnt = 0;
  static char *read_ptr;
  static char read_buf[MAXLINE];

  if (read_cnt <= 0) {
  again:
    if ( (read_cnt = read(fd, read_buf, sizeof(read_buf))) < 0) {
      if (errno == EINTR)
	goto again;
      return -1;
    } else if (read_cnt == 0)
      return 0;
    read_ptr = read_buf;
  }
  read_cnt--;
  *ptr = *read_ptr++;
  return 1;
}

int readline(fd,vptr,maxlen)
int fd;
void *vptr;
int maxlen;
{
  int n,rc;
  char c, *ptr;

  ptr = vptr;
  for(n=1; n < maxlen; n++) {
    if ((rc = my_read(fd,&c)) ==1) {
      *ptr++ = c;
      if (c == '\n')
	break;
    } else if (rc == 0) {
      if (n == 1) 
	return 0;
      else
	break;
    } else
      return -1;
  }
  *ptr = 0;
  return n;
}

static char *skipstuff(char *line)
{
  while (*line != ' ') line++;
  return line;
}

static char *skipblanks(char *line)
{
  while (*line == ' ') line++;
  return line;
}

static void parseInfo(CcsServer *svr, char *line)
{
  char ans[32];
  int num, i;
  line = skipblanks(line);

  /*DEBUGGING */
  /* printf("in client, Line = %s\n", line); */

  sscanf(line, "%s", ans);
  line = skipstuff(line); line = skipblanks(line);
  sscanf(line, "%d", &(svr->numNodes));
  line = skipstuff(line); line = skipblanks(line);
  svr->numProcs = (int *) malloc(svr->numNodes * sizeof(int));
  svr->nodeIPs = (int *) malloc(svr->numNodes * sizeof(int));
  svr->nodePorts = (int *) malloc(svr->numNodes * sizeof(int));
  svr->numPes = 0;
  for(i=0;i<svr->numNodes;i++) {
    sscanf(line, "%d", &(svr->numProcs[i]));
    line = skipstuff(line); line= skipblanks(line);
    svr->numPes += svr->numProcs[i];
  }
  for(i=0;i<svr->numNodes;i++) {
    sscanf(line, "%d", &(svr->nodeIPs[i]));
    line = skipstuff(line); line= skipblanks(line);
  }
  for(i=0;i<svr->numNodes;i++) {
    sscanf(line, "%d", &(svr->nodePorts[i]));
    line = skipstuff(line); line= skipblanks(line);
  }
}

static void printSvr(CcsServer *svr)
{
  int i;
  printf("hostIP: %d\n", svr->hostIP);
  printf("hostPort: %d\n", svr->hostPort);
  printf("myIP: %d\n", svr->myIP);
  printf("myPort: %d\n", svr->myPort);
  printf("myFd: %d\n", svr->myFd);
  printf("numNodes: %d\n", svr->numNodes);
  printf("numPes: %d\n", svr->numPes);
  for(i=0;i<svr->numNodes;i++) {
    printf("Node[%d] has %d processors at IP=%d, port=%d\n",
            i, svr->numProcs[i], svr->nodeIPs[i], svr->nodePorts[i]);
  }
}

/**
 * Converse Client-Server Module: Client Side
 */
int CcsConnect(CcsServer *svr, char *host, int port)
{
  int fd;
  char ans[32];
  char line[1024];
  FILE *f;
  strcpy(svr->hostAddr, host);
  svr->hostPort = port;
  svr->hostIP = skt_ip(host);
  svr->persFd = -1;
  skt_server(svr);

  fd = skt_connect(svr->hostIP, svr->hostPort, 120);
  if(fd == (-1)) {
    fprintf(stderr, "Cannot connect to server\n");
    exit(1);
  }
  write(fd, "getinfo ", strlen("getinfo "));
  sprintf(ans, "%d %d\n", svr->myIP, svr->myPort);
  write(fd, ans, strlen(ans));
  close(fd);

  printf("Waiting for connection\n");
  
  fd = skt_accept(svr);

  printf("connected\n");

  f = fdopen(fd, "r+");
  line[0] = 0;
  fgets(line, 1023, f);
  fclose(f);
  close(fd);
  zap_newline(line);
  parseInfo(svr, line);
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

int CcsSendRequest(CcsServer *svr, char *hdlrID, int pe, uint size, void *msg)
{
  int startpe=0, endpe=0, i;
  int fd;
  char line[1024];
  for(i=0;i<svr->numNodes;i++) {
    endpe += svr->numProcs[i];
    if(pe >= startpe && pe < endpe)
      break;
    startpe = endpe;
  }
  pe -= startpe;
  fd = skt_connect(svr->nodeIPs[i], svr->nodePorts[i], 120);
  sprintf(line, "req %d %d %d %d %s\n", pe, size, svr->myIP, svr->myPort, 
                                        hdlrID);
  write(fd, line, strlen(line));
  write(fd, msg, size);
  close(fd);
}

int CcsSendRequestFd(CcsServer *svr, char *hdlrID, int pe, uint size, void *msg)
{
  int startpe=0, endpe=0, i;
  int fd;
  char line[1024];
  for(i=0;i<svr->numNodes;i++) {
    endpe += svr->numProcs[i];
    if(pe >= startpe && pe < endpe)
      break;
    startpe = endpe;
  }
  pe -= startpe;

  if(svr->persFd == -1){
    fd = skt_connect(svr->nodeIPs[i], svr->nodePorts[i], 120);
    svr->persFd = fd;
  }
  else{
    fd = svr->persFd;
  }
  sprintf(line, "req %d %d %d %d %s\n", pe, size, svr->myIP, svr->myPort, 
                                        hdlrID);
  write(fd, line, strlen(line));
  write(fd, msg, size);
}


int CcsRecvResponse(CcsServer *svr, uint maxsize, void *recvBuffer, int timeout)
{
  char line[1024], ans[16];
  int size = 0, fd, nreadable = 0;
  FILE *f;
  fd_set listenfds;
  struct timeval tmo;
  int nread = 0, totalRead = 0;

  strcpy(ans, "");
  if(timeout != 0){
  selectblock:
    FD_ZERO(&listenfds);
    FD_SET(svr->myFd, &listenfds);
    tmo.tv_sec = timeout / 1000;
    tmo.tv_usec = (timeout * 1000) % 1000000;
    nreadable = select(svr->myFd + 1, &listenfds, NULL, NULL, &tmo);
    if ((nreadable<0)&&((errno==EINTR)||(errno==EBADF))) goto selectblock;
    if (nreadable == 0) return 0;
  }

  /* Ignore spurious accept requests */
  if(!FD_ISSET(svr->myFd, &listenfds)){
    return(0);
  }

  fd = skt_accept(svr);
  f = fdopen(fd, "r+");

  line[0] = 0;
  nread = read(fd, line, FIXED_LENGTH);
  zap_newline(line);

  sscanf(line, "%s%d", ans, &size);

  /* Spurious data */
  if(line[0] == 0){
    return(0);
  }

  while(totalRead < size){
    nread = read(fd, recvBuffer+totalRead, size-totalRead);
    totalRead += nread;
  }
  fclose(f);
  close(fd);
  usleep(1);
  return 1;
}

int CcsRecvResponseFd(CcsServer *svr, uint maxsize, void *recvBuffer, int timeout)
{
  char line[1024], ans[16];
  int size = 0, nreadable = 0;
  fd_set listenfds;
  struct timeval tmo;
  int nread = 0, totalRead = 0;

  strcpy(ans, "");

  if(timeout != 0){
  selectblock:
    FD_ZERO(&listenfds);
    FD_SET(svr->persFd, &listenfds);
    tmo.tv_sec = timeout / 1000;
    tmo.tv_usec = (timeout * 1000) % 1000000;
    nreadable = select(svr->persFd + 1, &listenfds, NULL, NULL, &tmo);
    if ((nreadable<0)&&((errno==EINTR)||(errno==EBADF))) goto selectblock;
    if (nreadable == 0) return 0;
  }

  /* Ignore spurious accept requests */
  if(!FD_ISSET(svr->persFd, &listenfds)){
    return(0);
  }

  line[0] = 0;
  nread = read(svr->persFd, line, FIXED_LENGTH);
  zap_newline(line);

  sscanf(line, "%s%d", ans, &size);

  /* Spurious data */
  if(line[0] == 0){
    return(0);
  }
  
  while(totalRead < size){
    nread = read(svr->persFd, recvBuffer+totalRead, size-totalRead);
    totalRead += nread;
  }
  usleep(1);
  return 1;
}


int CcsProbe(CcsServer *svr)
{
  fprintf(stderr, "CcsProbe not implemented.\n");
  exit(1);
}

int CcsResponseHandler(CcsServer *svr, CcsHandlerFn fn)
{
  svr->callback = fn;
  fprintf(stderr, "CcsResponseHandler not implemented.\n");
  exit(1);
}

int CcsFinalize(CcsServer *svr)
{
  close(svr->myFd);
}
