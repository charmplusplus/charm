#include <stdlib.h>
/* #include <sys/timer.h> */
#include <sys/unistd.h>
#include <errno.h>
#include <sys/time.h>

#include "conv-ccs.h"

#if CMK_WEB_MODE
int appletFd = -1;
#endif

#if NODE_0_IS_CONVHOST
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

int serverFlag = 0;
extern int inside_comm;
CpvExtern(int, strHandlerID);

int hostport, hostskt;
int hostskt_ready_read;

CpvStaticDeclare(int, CHostHandlerIndex);
static unsigned int *nodeIPs;
static unsigned int *nodePorts;
static int numRegistered;

static void KillEveryoneCode(int n)
{
  char str[128];
  sprintf(str, "Fatal Error: code %d\n", n);
  CmiAbort(str);
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

void writeall(int fd, char *buf, int size)
{
  int ok;
  while (size) {
    retry:
    ok = write(fd, buf, size);
    if ((ok<0)&&((errno==EBADF)||(errno==EINTR))) goto retry;
    if (ok<=0) {
      CmiAbort("Write failed ..\n");
    }
    size-=ok; buf+=ok;
  }
}

void skt_server(ppo, pfd)
unsigned int *ppo;
unsigned int *pfd;
{
  int fd= -1;
  int ok, len;
  struct sockaddr_in addr;

retry:
  fd = socket(PF_INET, SOCK_STREAM, 0);
  if ((fd<0)&&((errno==EINTR)||(errno==EBADF))) goto retry;
  if (fd < 0) { perror("socket 1"); KillEveryoneCode(93483); }
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  ok = bind(fd, (struct sockaddr *)&addr, sizeof(addr));
  if (ok < 0) { perror("bind"); KillEveryoneCode(22933); }
  ok = listen(fd,5);
  if (ok < 0) { perror("listen"); KillEveryoneCode(3948); }
  len = sizeof(addr);
  ok = getsockname(fd, (struct sockaddr *)&addr, &len);
  if (ok < 0) { perror("getsockname"); KillEveryoneCode(93583); }

  *pfd = fd;
  *ppo = ntohs(addr.sin_port);
}

int skt_connect(ip, port, seconds)
unsigned int ip; int port; int seconds;
{
  struct sockaddr_in remote; short sport=port;
  int fd, ok, begin;

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
    if (fd < 0) KillEveryoneCode(234234);

  conn:
    ok = connect(fd, (struct sockaddr *)&(remote), sizeof(remote));
    if (ok>=0) break;
    close(fd);
    switch (errno) {
    case EINTR: case EBADF: case EALREADY: case EISCONN: break;
    case ECONNREFUSED: jsleep(1,0); break;
    case EADDRINUSE: jsleep(1,0); break;
    case EADDRNOTAVAIL: jsleep(1,0); break;
    default: return -1;
    }
  }
  if (ok<0) return -1;
  return fd;
}

static void skt_accept(src, pip, ppo, pfd)
int src;
unsigned int *pip;
unsigned int *ppo;
unsigned int *pfd;
{
  int i, fd;
  struct sockaddr_in remote;
  i = sizeof(remote);
 acc:
  fd = accept(src, (struct sockaddr *)&remote, &i);
  if ((fd<0)&&((errno==EINTR)||(errno==EBADF))) goto acc;
  if (fd<0) { perror("accept"); KillEveryoneCode(39489); }
  *pip=htonl(remote.sin_addr.s_addr);
  *ppo=htons(remote.sin_port);
  *pfd=fd;
}

static void CheckSocketsReady(void)
{
  static fd_set rfds;
  static fd_set wfds;
  struct timeval tmo;
  int nreadable;

  FD_ZERO(&rfds);
  FD_ZERO(&wfds);
  FD_SET(hostskt, &rfds);
  FD_SET(hostskt, &wfds);
  tmo.tv_sec = 0;
  tmo.tv_usec = 0;
  nreadable = select(FD_SETSIZE, &rfds, &wfds, NULL, &tmo);
  if (nreadable <= 0) {
    hostskt_ready_read = 0;
    return;
  }
  hostskt_ready_read = (FD_ISSET(hostskt, &rfds));
}

void CHostRegister(void)
{
  struct hostent *hostent;
  char hostname[100];
  int ip;
  char *msg;
  int *ptr;
  int msgSize = CmiMsgHeaderSizeBytes + 3 * sizeof(unsigned int);

  if(gethostname(hostname, 99) < 0) {
    hostent = gethostent();
  }
  else{
    hostent = gethostbyname(hostname);
  }
  if (hostent == 0)
    ip = 0x7f000001;
  else
    ip = htonl(*((CmiInt4 *)(hostent->h_addr_list[0])));

  msg = (char *)CmiAlloc(msgSize * sizeof(char));
  ptr = (int *)(msg + CmiMsgHeaderSizeBytes);
  ptr[0] = CmiMyPe();
  ptr[1] = ip;
  ptr[2] = hostport;
  CmiSetHandler(msg, CpvAccess(CHostHandlerIndex));
  CmiSyncSendAndFree(0, msgSize, msg);
}

unsigned int clientIP, clientPort, clientKillPort;

void CHostGetOne()
{
  char line[10000];
  char rest[1000];
  int ip, port, fd;  FILE *f;
#if CMK_WEB_MODE
  char hndlrId[100];
  int dont_close = 0;
  int svrip, svrport;
#endif

  skt_accept(hostskt, &ip, &port, &fd);
  f = fdopen(fd,"r");
  while (fgets(line, 9999, f)) {
    if (strncmp(line, "req ", 4)==0) {
      char cmd[5], *msg;
      int pe, size, len;
      int ret;
#if CMK_WEB_MODE   
      sscanf(line, "%s%d%d%d%d%s", cmd, &pe, &size, &svrip, &svrport, hndlrId);
      if(strcmp(hndlrId, "MonitorHandler") == 0) {
	appletFd = fd;
	dont_close = 1;
      }
#else
      sscanf(line, "%s%d%d", cmd, &pe, &size);
#endif
      /* DEBUGGING */
      CmiPrintf("Line = %s\n", line);

      sscanf(line, "%s%d%d", cmd, &pe, &size);
      len = strlen(line);
      msg = (char *) CmiAlloc(len+size+CmiMsgHeaderSizeBytes+1);
      if (!msg)
        CmiPrintf("%d: Out of mem\n", CmiMyPe());
      CmiSetHandler(msg, CpvAccess(strHandlerID));
      CmiPrintf("hdlr ID = %d\n", CpvAccess(strHandlerID));
      strcpy(msg+CmiMsgHeaderSizeBytes, line);
      ret = fread(msg+CmiMsgHeaderSizeBytes+len, 1, size, f);
      CmiPrintf("size = %d, ret =%d\n", size, ret);
      msg[CmiMsgHeaderSizeBytes+len+size] = '\0';
      CmiSyncSendAndFree(CmiMyPe(), CmiMsgHeaderSizeBytes+len+size+1, msg);

#if CMK_USE_PERSISTENT_CCS
      if(dont_close == 1) break;
#endif

    }
    else if (strncmp(line, "getinfo ", 8)==0) {
      char pre[1024], reply[1024], ans[1024];
      char cmd[20];
      int fd;
      int i;
      int nscanfread;
      int nodetab_rank0_size = CmiNumNodes();

      /* DEBUGGING */
      CmiPrintf("Line = %s\n", line);
      nscanfread = sscanf(line, "%s%u%u", cmd, &clientIP, &clientPort);
      if(nscanfread != 3){

	/* DEBUGGING */
	CmiPrintf("Entering further read...\n");

        fgets(rest, 999, f);

	/* DEBUGGING */
	CmiPrintf("Rest = %s\n", rest);

        sscanf(rest, "%u%u", &clientIP, &clientPort);
      }
      clientIP = (CmiInt4) clientIP;
      strcpy(pre, "info");
      reply[0] = 0;
      sprintf(ans, "%d ", nodetab_rank0_size);
      strcat(reply, ans);
      for(i=0;i<nodetab_rank0_size;i++) {
        strcat(reply, "1 ");
      }
      for(i=0;i<nodetab_rank0_size;i++) {
        sprintf(ans, "%d ", (CmiInt4) nodeIPs[i]);
        strcat(reply, ans);
      }
      for(i=0;i<nodetab_rank0_size;i++) {
        sprintf(ans, "%d ", nodePorts[i]);
        strcat(reply, ans);
      }
      fd = skt_connect(clientIP, clientPort, 60);

      /** Debugging **/
      CmiPrintf("After Connect for getinfo reply\n");


      if (fd<=0) KillEveryoneCode(2932);
      write(fd, pre, strlen(pre));
      write(fd, " ", 1);
      write(fd, reply, strlen(reply));
      close(fd);
    }
    else if (strncmp(line, "clientdata", strlen("clientdata"))==0){
      int nread;
      char cmd[20];
      
      nread = sscanf(line, "%s%d", cmd, &clientKillPort);
      if(nread != 2){
	fgets(rest, 999, f);
	
	/* DEBUGGING */
	CmiPrintf("Rest = %s\n", rest);
	
        sscanf(rest, "%d", &clientKillPort);

	/* Debugging */

	CmiPrintf("After sscanf\n");
      }
    }
    else {
      CmiPrintf("Request: %s\n", line);
      KillEveryoneCode(2933);
    }
  }
  CmiPrintf("Out of fgets loop\n");
#if CMK_WEB_MODE
  if(dont_close==0) {
#endif
  fclose(f);
  close(fd);
#if CMK_WEB_MODE
  }
#endif
}

void CommunicationServer()
{
  if(inside_comm)
    return;
    CheckSocketsReady();
     /*if (hostskt_ready_read) { CHostGetOne(); continue; }*/
}

void CHostHandler(char *msg)
{
  int pe;
  int *ptr = (int *)(msg + CmiMsgHeaderSizeBytes);
  pe = ptr[0];
  nodeIPs[pe] = (unsigned int)(ptr[1]);
  nodePorts[pe] = (unsigned int)(ptr[2]);
  numRegistered++;
 
  if(numRegistered == CmiNumPes()){
    if (serverFlag == 1)  {
      CmiPrintf("%s\nServer IP = %u, Server port = %u $\n", CMK_CCS_VERSION, (CmiInt4) nodeIPs[0], nodePorts[0]);
    }
  }
}

void CHostInit()
{
  nodeIPs = (unsigned int *)malloc(CmiNumPes() * sizeof(unsigned int));
  nodePorts = (unsigned int *)malloc(CmiNumPes() * sizeof(unsigned int));
  CpvInitialize(int, CHostHandlerIndex);
  CpvAccess(CHostHandlerIndex) = CmiRegisterHandler(CHostHandler);
}

#endif

/* move */

#if CMK_DEBUG_MODE
#include "fifo.h"
CpvDeclare(int, freezeModeFlag);
CpvDeclare(int, continueFlag);
CpvDeclare(int, stepFlag);
CpvDeclare(void *, debugQueue);
unsigned int freezeIP;
int freezePort;
char* breakPointHeader;
char* breakPointContents;

void dummyF()
{
}

static void CpdDebugHandler(char *msg)
{
  char *reply, *temp;
  int index;
  
  if(CcsIsRemoteRequest()) {
    char name[128];
    unsigned int ip, port;
    CcsCallerId(&ip, &port);
    sscanf(msg+CmiMsgHeaderSizeBytes, "%s", name);
    reply = NULL;

    if (strcmp(name, "freeze") == 0) {
      CpdFreeze();
      msgListCleanup();
      msgListCache();
      CmiPrintf("freeze received\n");
    }
    else if (strcmp(name, "unfreeze") == 0) {
      CpdUnFreeze();
      msgListCleanup();
      CmiPrintf("unfreeze received\n");
    }
    else if (strcmp(name, "getObjectList") == 0){
      CmiPrintf("getObjectList received\n");
      reply = getObjectList();
      CmiPrintf("list obtained");
      if(reply == NULL){
	CmiPrintf("list empty");
	CcsSendReply(ip, port, strlen("$") + 1, "$");
      }
      else{
	CmiPrintf("list : %s\n", reply);
	CcsSendReply(ip, port, strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if(strncmp(name,"getObjectContents",strlen("getObjectContents"))==0){
      CmiPrintf("getObjectContents received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getObjectContents(index);
      CmiPrintf("Object Contents : %s\n", reply);
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      free(reply);
    }
    else if (strcmp(name, "getMsgListSched") == 0){
      CmiPrintf("getMsgListSched received\n");
      reply = getMsgListSched();
      if(reply == NULL)
	CcsSendReply(ip, port, strlen("$") + 1, "$");
      else{
	CcsSendReply(ip, port, strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if (strcmp(name, "getMsgListFIFO") == 0){
      CmiPrintf("getMsgListFIFO received\n");
      reply = getMsgListFIFO();
      if(reply == NULL)
	CcsSendReply(ip, port, strlen("$") + 1, "$");
      else{
	CcsSendReply(ip, port, strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if (strcmp(name, "getMsgListPCQueue") == 0){
      CmiPrintf("getMsgListPCQueue received\n");
      reply = getMsgListPCQueue();
      if(reply == NULL)
	CcsSendReply(ip, port, strlen("$") + 1, "$");
      else{
	CcsSendReply(ip, port, strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if (strcmp(name, "getMsgListDebug") == 0){
      CmiPrintf("getMsgListDebug received\n");
      reply = getMsgListDebug();
      if(reply == NULL)
	CcsSendReply(ip, port, strlen("$") + 1, "$");
      else{
	CcsSendReply(ip, port, strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if(strncmp(name,"getMsgContentsSched",strlen("getMsgContentsSched"))==0){
      CmiPrintf("getMsgContentsSched received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getMsgContentsSched(index);
      CmiPrintf("Message Contents : %s\n", reply);
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      free(reply);
    }
    else if(strncmp(name,"getMsgContentsFIFO",strlen("getMsgContentsFIFO"))==0){
      CmiPrintf("getMsgContentsFIFO received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getMsgContentsFIFO(index);
      CmiPrintf("Message Contents : %s\n", reply);
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      free(reply);
    }
    else if (strncmp(name, "getMsgContentsPCQueue", strlen("getMsgContentsPCQueue")) == 0){
      CmiPrintf("getMsgContentsPCQueue received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getMsgContentsPCQueue(index);
      CmiPrintf("Message Contents : %s\n", reply);
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      free(reply);
    }
    else if (strncmp(name, "getMsgContentsDebug", strlen("getMsgContentsDebug")) == 0){
      CmiPrintf("getMsgContentsDebug received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getMsgContentsDebug(index);
      CmiPrintf("Message Contents : %s\n", reply);
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      free(reply);
    } 
    else if (strncmp(name, "step", strlen("step")) == 0){
      CmiPrintf("step received\n");
      CpvAccess(stepFlag) = 1;
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &freezePort);
      freezeIP = ip;
      CpdUnFreeze();
    }
    else if (strncmp(name, "continue", strlen("continue")) == 0){
      CmiPrintf("continue received\n");
      CpvAccess(continueFlag) = 1;
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &freezePort);
      freezeIP = ip;
      CpdUnFreeze();
    }
    else if (strcmp(name, "getBreakStepContents") == 0){
      CmiPrintf("getBreakStepContents received\n");
      if(breakPointHeader == 0){
	CcsSendReply(ip, port, strlen("$") + 1, "$");
      }
      else{
	reply = (char *)malloc(strlen(breakPointHeader) + strlen(breakPointContents) + 1);
	strcpy(reply, breakPointHeader);
	strcat(reply, "@");
	strcat(reply, breakPointContents);
	CcsSendReply(ip, port, strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if (strcmp(name, "getSymbolTableInfo") == 0){
      CmiPrintf("getSymbolTableInfo received");
      reply = getSymbolTableInfo();
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      reply = getBreakPoints();
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      free(reply);
    }
    else if (strncmp(name, "setBreakPoint", strlen("setBreakPoint")) == 0){
      CmiPrintf("setBreakPoint received\n");
      temp = strstr(name, "#");
      temp++;
      setBreakPoints(temp);
    }
    else if (strncmp(name, "gdbRequest", strlen("gdbRequest")) == 0){
      CmiPrintf("gdbRequest received\n");
      dummyF();
    }

    else if (strcmp(name, "quit") == 0){
      CpdUnFreeze();
      CsdExitScheduler();
    }
    else{
      CmiPrintf("incorrect command:%s received,len=%ld\n",name,strlen(name));
    }
  }
}

void CpdInit(void)
{
  CpvInitialize(int, freezeModeFlag);
  CpvAccess(freezeModeFlag) = 0;

  CpvInitialize(int, continueFlag);
  CpvInitialize(int, stepFlag);
  CpvAccess(continueFlag) = 0;
  CpvAccess(stepFlag) = 0;

  CpvInitialize(void *, debugQueue);
  CpvAccess(debugQueue) = FIFO_Create();
    
  CpdInitializeObjectTable();
  CpdInitializeHandlerArray();
  CpdInitializeBreakPoints();

  CcsRegisterHandler("DebugHandler", CpdDebugHandler);
}  

void CpdFreeze(void)
{
  CpvAccess(freezeModeFlag) = 1;
}  

void CpdUnFreeze(void)
{
  CpvAccess(freezeModeFlag) = 0;
}  

#endif


#if CMK_WEB_MODE

#define WEB_INTERVAL 2000
#define MAXFNS 20

/* For Web Performance */
typedef int (*CWebFunction)();
unsigned int appletIP;
unsigned int appletPort;
int countMsgs;
char **valueArray;
CWebFunction CWebPerformanceFunctionArray[MAXFNS];
int CWebNoOfFns;
CpvDeclare(int, CWebPerformanceDataCollectionHandlerIndex);
CpvDeclare(int, CWebHandlerIndex);

static void sendDataFunction(void)
{
  char *reply;
  int len = 0, i;

  for(i=0; i<CmiNumPes(); i++){
    len += (strlen((char*)(valueArray[i]+
			   CmiMsgHeaderSizeBytes+sizeof(int)))+1);
    /* for the spaces in between */
  }
  len+=6; /* for 'perf ' and the \0 at the end */

  reply = (char *)malloc(len * sizeof(char));
  strcpy(reply, "perf ");

  for(i=0; i<CmiNumPes(); i++){
    strcat(reply, (valueArray[i] + CmiMsgHeaderSizeBytes + sizeof(int)));
    strcat(reply, " ");
  }

  /* Do the CcsSendReply */
#if CMK_USE_PERSISTENT_CCS
  CcsSendReplyFd(appletIP, appletPort, strlen(reply) + 1, reply);
#else
  CcsSendReply(appletIP, appletPort, strlen(reply) + 1, reply);
#endif
  /* foo */
  CmiPrintf("Reply = %s\n", reply);
  free(reply);

  /* Free valueArray contents */
  for(i = 0; i < CmiNumPes(); i++){
    CmiFree(valueArray[i]);
    valueArray[i] = 0;
  }

  countMsgs = 0;
}

void CWebPerformanceDataCollectionHandler(char *msg){
  int src;
  char *prev;

  if(CmiMyPe() != 0){
    CmiAbort("Wrong processor....\n");
  }
  src = ((int *)(msg + CmiMsgHeaderSizeBytes))[0];
  CmiGrabBuffer((void **)&msg);
  prev = valueArray[src]; /* Previous value, ideally 0 */
  valueArray[src] = (msg);
  if(prev == 0) countMsgs++;
  else CmiFree(prev);

  if(countMsgs == CmiNumPes()){
    sendDataFunction();
  }
}

void CWebPerformanceGetData(void *dummy)
{
  char *msg, data[100];
  int msgSize;
  int i;

  if(appletIP == 0) {
    return;  /* No use if client is not yet connected */
  }

  strcpy(data, "");
  /* Evaluate each of the functions and get the values */
  for(i = 0; i < CWebNoOfFns; i++)
    sprintf(data, "%s %d", data, (*(CWebPerformanceFunctionArray[i]))());

  msgSize = (strlen(data)+1) + sizeof(int) + CmiMsgHeaderSizeBytes;
  msg = (char *)CmiAlloc(msgSize);
  ((int *)(msg + CmiMsgHeaderSizeBytes))[0] = CmiMyPe();
  strcpy(msg + CmiMsgHeaderSizeBytes + sizeof(int), data);
  CmiSetHandler(msg, CpvAccess(CWebPerformanceDataCollectionHandlerIndex));
  CmiSyncSendAndFree(0, msgSize, msg);

  CcdCallFnAfter(CWebPerformanceGetData, 0, WEB_INTERVAL);
}

void CWebPerformanceRegisterFunction(CWebFunction fn)
{
  CWebPerformanceFunctionArray[CWebNoOfFns] = fn;
  CWebNoOfFns++;
}

static void CWebHandler(char *msg){
  int msgSize;
  char *getStuffMsg;
  int i;

  if(CcsIsRemoteRequest()) {
    char name[32];
    unsigned int ip, port;

    CcsCallerId(&ip, &port);
    sscanf(msg+CmiMsgHeaderSizeBytes, "%s", name);

    if(strcmp(name, "getStuff") == 0){
      appletIP = ip;
      appletPort = port;

      valueArray = (char **)malloc(sizeof(char *) * CmiNumPes());
      for(i = 0; i < CmiNumPes(); i++)
        valueArray[i] = 0;

      for(i = 1; i < CmiNumPes(); i++){
        msgSize = CmiMsgHeaderSizeBytes + 2*sizeof(int);
        getStuffMsg = (char *)CmiAlloc(msgSize);
        ((int *)(getStuffMsg + CmiMsgHeaderSizeBytes))[0] = appletIP;
        ((int *)(getStuffMsg + CmiMsgHeaderSizeBytes))[1] = appletPort;
        CmiSetHandler(getStuffMsg, CpvAccess(CWebHandlerIndex));
        CmiSyncSendAndFree(i, msgSize, getStuffMsg);

        CcdCallFnAfter(CWebPerformanceGetData, 0, WEB_INTERVAL);
      }
    }
    else{
      CmiPrintf("incorrect command:%s received, len=%ld\n",name,strlen(name));
    }
  }
  else{
    /* Ordinary converse message */
    appletIP = ((int *)(msg + CmiMsgHeaderSizeBytes))[0];
    appletPort = ((int *)(msg + CmiMsgHeaderSizeBytes))[1];

    CcdCallFnAfter(CWebPerformanceGetData, 0, WEB_INTERVAL);
  }
}

int f2()
{
  return(CqsLength(CpvAccess(CsdSchedQueue)));
}

int f3()
{
  struct timeval tmo;

  gettimeofday(&tmo, NULL);
  return(tmo.tv_sec % 10 + CmiMyPe() * 3);
}

/** ADDED 2-14-99 BY MD FOR USAGE TRACKING (TEMPORARY) **/

#define CkUTimer()      ((int)(CmiWallTimer() * 1000000.0))

typedef unsigned int un_int;
CpvDeclare(un_int, startTime);
CpvDeclare(un_int, beginTime);
CpvDeclare(un_int, usedTime);
CpvDeclare(int, PROCESSING);

/* Call this when the program is started
 -> Whenever traceModuleInit would be called
 -> -> see conv-core/convcore.c
*/
void initUsage()
{
   CpvInitialize(un_int, startTime);
   CpvInitialize(un_int, beginTime);
   CpvInitialize(un_int, usedTime);
   CpvInitialize(int, PROCESSING);
   CpvAccess(beginTime)  = CkUTimer();
   CpvAccess(usedTime)   = 0;
   CpvAccess(PROCESSING) = 0;
}

int getUsage()
{
   int usage = 0;
   un_int time      = CkUTimer();
   un_int totalTime = time - CpvAccess(beginTime);

   if(CpvAccess(PROCESSING))
   {
      CpvAccess(usedTime) += time - CpvAccess(startTime);
      CpvAccess(startTime) = time;
   }
   if(totalTime > 0)
      usage = (100 * CpvAccess(usedTime))/totalTime;
   CpvAccess(usedTime)  = 0;
   CpvAccess(beginTime) = time;
   return usage;
}

/* Call this when a BEGIN_PROCESSING event occurs
 -> Whenever a trace_begin_execute or trace_begin_charminit
    would be called
 -> -> See ck-core/init.c,main.c and conv-core/convcore.c
*/
void usageStart()
{
   if(CpvAccess(PROCESSING)) return;

   CpvAccess(startTime)  = CkUTimer();
   CpvAccess(PROCESSING) = 1;
}

/* Call this when an END_PROCESSING event occurs
 -> Whenever a trace_end_execute or trace_end_charminit
    would be called
 -> -> See ck-core/init.c,main.c and conv-core/threads.c
*/
void usageStop()
{
   if(!CpvAccess(PROCESSING)) return;

   CpvAccess(usedTime)   += CkUTimer() - CpvAccess(startTime);
   CpvAccess(PROCESSING) = 0;
}

void CWebInit(void)
{
  CcsRegisterHandler("MonitorHandler", CWebHandler);

  CpvInitialize(int, CWebHandlerIndex);
  CpvAccess(CWebHandlerIndex) = CmiRegisterHandler(CWebHandler);

  CpvInitialize(int, CWebPerformanceDataCollectionHandlerIndex);
  CpvAccess(CWebPerformanceDataCollectionHandlerIndex) =
    CmiRegisterHandler(CWebPerformanceDataCollectionHandler);

  CWebPerformanceRegisterFunction(getUsage);
  CWebPerformanceRegisterFunction(f2);

}

#endif

/* \move */


/* move */

/*****************************************************************************
 *
 * Converse Client-Server Functions
 *
 *****************************************************************************/

#if CMK_CCS_AVAILABLE

typedef struct CcsListNode {
  char name[32];
  int hdlr;
  struct CcsListNode *next;
}CcsListNode;

CpvStaticDeclare(CcsListNode*, ccsList);
CpvStaticDeclare(int, callerIP);
CpvStaticDeclare(int, callerPort);
CpvDeclare(int, strHandlerID);

static void CcsStringHandlerFn(char *msg)
{
  char cmd[10], hdlrName[32], *cmsg, *omsg=msg;
  int ip, port, pe, size, nread, hdlrID;
  CcsListNode *list = CpvAccess(ccsList);

  msg += CmiMsgHeaderSizeBytes;
  nread = sscanf(msg, "%s%d%d%d%d%s", 
                 cmd, &pe, &size, &ip, &port, hdlrName);
  if(nread!=6) CmiAbort("Garbled message from client");
  CmiPrintf("message for %s\n", hdlrName);
  while(list!=0) {
    if(strcmp(hdlrName, list->name)==0) {
      hdlrID = list->hdlr;
      break;
    }
    list = list->next;
  }
  if(list==0) CmiAbort("Invalid Service Request\n");
  while(*msg != '\n') msg++;
  msg++;
  cmsg = (char *) CmiAlloc(size+CmiMsgHeaderSizeBytes+1);
  memcpy(cmsg+CmiMsgHeaderSizeBytes, msg, size);
  cmsg[CmiMsgHeaderSizeBytes+size] = '\0';

  CmiSetHandler(cmsg, hdlrID);
  CpvAccess(callerIP) = ip;
  CpvAccess(callerPort) = port;
  CmiHandleMessage(cmsg);
  CmiGrabBuffer((void **)&omsg);
  CmiFree(omsg);
  CpvAccess(callerIP) = 0;
}

/* note: was static void -jeff */
void CcsInit(void)
{
  CpvInitialize(CcsListNode*, ccsList);
  CpvAccess(ccsList) = 0;
  CpvInitialize(int, callerIP);
  CpvAccess(callerIP) = 0;
  CpvInitialize(int, callerPort);
  CpvAccess(callerPort) = 0;
  CpvInitialize(int, strHandlerID);
  CpvAccess(strHandlerID) = CmiRegisterHandler(CcsStringHandlerFn);
}

void CcsUseHandler(char *name, int hdlr)
{
  CcsListNode *list=CpvAccess(ccsList);
  if(list==0) {
    list = (CcsListNode *)malloc(sizeof(CcsListNode));
    CpvAccess(ccsList) = list;
  } else {
    while(list->next != 0) 
      list = list->next;
    list->next = (CcsListNode *)malloc(sizeof(CcsListNode));
    list = list->next;
  }
  strcpy(list->name, name);
  list->hdlr = hdlr;
  list->next = 0;
}

int CcsRegisterHandler(char *name, CmiHandler fn)
{
  int hdlr = CmiRegisterHandlerLocal(fn);
  CcsUseHandler(name, hdlr);
  return hdlr;
}

int CcsEnabled(void)
{
  return 1;
}

int CcsIsRemoteRequest(void)
{
  return (CpvAccess(callerIP) != 0);
}

void CcsCallerId(unsigned int *pip, unsigned int *pport)
{
  *pip = CpvAccess(callerIP);
  *pport = CpvAccess(callerPort);
}

extern int skt_connect(unsigned int, int, int);
extern void writeall(int, char *, int);

void CcsSendReply(unsigned int ip, unsigned int port, int size, void *msg)
{
  char cmd[100];
  int fd;

  fd = skt_connect(ip, port, 120);
  
  if (fd<0) {
      CmiPrintf("client Exited\n");
      return; /* maybe the requester exited */
  }
  sprintf(cmd, "reply %10d\n", size);
  writeall(fd, cmd, strlen(cmd));
  writeall(fd, msg, size);

#if CMK_SYNCHRONIZE_ON_TCP_CLOSE
  shutdown(fd, 1);
  { char c; while (read(fd, &c, 1)==EINTR); }
  close(fd);
#else
  close(fd);
#endif
}

#if CMK_USE_PERSISTENT_CCS
void CcsSendReplyFd(unsigned int ip, unsigned int port, int size, void *msg)
{
  char cmd[100];
  int fd;

  fd = appletFd;
  if (fd<0) {
    CmiPrintf("client Exited\n");
    return; /* maybe the requester exited */
  }
  sprintf(cmd, "reply %10d\n", size);
  writeall(fd, cmd, strlen(cmd));
  writeall(fd, msg, size);
#if CMK_SYNCHRONIZE_ON_TCP_CLOSE
  shutdown(fd, 1);
  while (read(fd, &c, 1)==EINTR);
#endif
}
#endif

#endif

/* \move */
