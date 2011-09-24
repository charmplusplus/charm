
/**********************************************************
      Converse One Sender multiple receiver to test the message latency and bandwidth
      
      modified from single pingpong

 ***************************************************/

#include <stdlib.h>
#include <converse.h>

enum {nCycles =4096};
enum { maxMsgSize = 1 << 22 };

CpvDeclare(int,msgSize);
CpvDeclare(int,cycleNum);
CpvDeclare(int,recvNum);
CpvDeclare(int,exitHandler);
CpvDeclare(int,node0Handler);
CpvDeclare(int,node1Handler);
CpvStaticDeclare(double,startTime);
CpvStaticDeclare(double,endTime);

#define  REUSE_MSG      1
#define USE_PERSISTENT     0

#if USE_PERSISTENT
PersistentHandle h;
#endif


typedef struct message_{
  char core[CmiMsgHeaderSizeBytes];
  int destPe;
  int size;
  int data[1];
} Message;

#if REUSE_MSG
Message **recvMsgs;
#endif

// Start the pingpong for each message size
void startRing()
{
  CpvAccess(cycleNum) = 0;

  //Increase message in powers of 4. Also add a converse header to that
  CpvAccess(msgSize) = (CpvAccess(msgSize)-CmiMsgHeaderSizeBytes)*2 + 
      CmiMsgHeaderSizeBytes;

  for(int i=1; i<CmiNumPes(); i++)
  {
      Message *msg = (Message *)CmiAlloc(CpvAccess(msgSize));
      msg->size = CpvAccess(msgSize);
      CmiSetHandler(msg,CpvAccess(node1Handler));
      CmiSyncSendAndFree(i, CpvAccess(msgSize), msg);
  }
  CpvAccess(startTime) = CmiWallTimer();
}

//the pingpong has finished, record message time
void ringFinished()
{
  //Print the time for that message size
  //CmiPrintf("\t\t  %.2lf\n", 
  //	     (1e6*(CpvAccess(endTime)-CpvAccess(startTime)))/(2.*nCycles));

  CmiPrintf("%d\t\t  %.2lf\n", 
             CpvAccess(msgSize)-CmiMsgHeaderSizeBytes, 
	     (1e6*(CpvAccess(endTime)-CpvAccess(startTime)))/(2.*nCycles));
  
  //Have we finished all message sizes?
  if (CpvAccess(msgSize) < maxMsgSize)
      //start the ring again
      startRing();
  else {
      //exit
      void *sendmsg = CmiAlloc(CmiMsgHeaderSizeBytes);
      CmiSetHandler(sendmsg,CpvAccess(exitHandler));
      CmiSyncBroadcastAllAndFree(CmiMsgHeaderSizeBytes,sendmsg);
  }
}

//We finished for all message sizes. Exit now
CmiHandler exitHandlerFunc(char *msg)
{
    CmiFree(msg);
    CsdExitScheduler();
#if REUSE_MSG
    if(CmiMyPe() ==0)
        free(recvMsgs);
#endif
    return 0;
}


//Handler on Node 0
CmiHandler node0HandlerFunc(Message *msg)
{
    Message *m;
    CpvAccess(recvNum)++;
#if REUSE_MSG
    recvMsgs[(msg->destPe)-1] = msg;
#endif
    if (CpvAccess(recvNum) == CmiNumPes()-1) {
        CpvAccess(recvNum) = 0;
        CpvAccess(cycleNum)++;
        if (CpvAccess(cycleNum) == nCycles) {
#if REUSE_MSG
            for(int i=0; i<CmiNumPes()-1; i++)
            {
                CmiFree(recvMsgs[i]);
            }
#endif
            CpvAccess(endTime) = CmiWallTimer();
            ringFinished();
        }
        else {
            for(int i=0; i<CmiNumPes()-1; i++)
            {
#if REUSE_MSG
                m= recvMsgs[i];
#endif
                CmiSetHandler(recvMsgs[i],CpvAccess(node1Handler));
#if USE_PERSISTENT
                CmiUsePersistentHandle(&h, 1);
#endif
                CmiSyncSendAndFree(i+1,CpvAccess(msgSize),m);
#if USE_PERSISTENT
                CmiUsePersistentHandle(NULL, 0);
#endif
            }

        }
    }
    return 0;
}

CmiHandler node1HandlerFunc(Message *msg)
{
    CmiSetHandler(msg,CpvAccess(node0Handler));
#if USE_PERSISTENT
    CmiUsePersistentHandle(&h, 1);
#endif
    msg->destPe = CmiMyPe();
    CmiSyncSendAndFree(0,msg->size,msg);
#if USE_PERSISTENT
    CmiUsePersistentHandle(NULL, 0);
#endif
    return 0;
}


//Converse main. Initialize variables and register handlers
CmiStartFn mymain()
{
    CpvInitialize(int,msgSize);
    CpvInitialize(int,cycleNum);
    
    CpvAccess(msgSize)= 4 + CmiMsgHeaderSizeBytes;
    
    CpvInitialize(int,exitHandler);
    CpvAccess(exitHandler) = CmiRegisterHandler((CmiHandler) exitHandlerFunc);
    CpvInitialize(int,node0Handler);
    CpvAccess(node0Handler) = CmiRegisterHandler((CmiHandler) node0HandlerFunc);
    CpvInitialize(int,node1Handler);
    CpvAccess(node1Handler) = CmiRegisterHandler((CmiHandler) node1HandlerFunc);
    
    CpvInitialize(double,startTime);
    CpvInitialize(double,endTime);
    
    int otherPe = CmiMyPe() ^ 1;
    
#if USE_PERSISTENT
    h = CmiCreatePersistent(otherPe, maxMsgSize+1024);
#endif
    
    if (CmiMyPe() == 0)
    {
#if REUSE_MSG
        recvMsgs = (Message**) malloc(sizeof(Message*)*(CmiNumPes()-1));
#endif
        startRing();
    }
    
    return 0;
}

int main(int argc,char *argv[])
{
    ConverseInit(argc,argv,(CmiStartFn)mymain,0,0);
    return 0;
}
