// File: BlueGene.h
// Author: Arun Kumar Singla <asingla@uiuc.edu> Nov 2000
// Updated by: Milind Bhandarkar

#ifndef _BlueGene_h
#define _BlueGene_h

#include "BlueGene.decl.h"
#include "charm++.h"

#define SIZE_INBUFFER     32
#define MAX_NUM_THREADS   200
#define MAX_HANDLERS      16

enum WorkType {LARGE_WORK=0, SMALL_WORK=1};

#if BG_TIMER
#define BGTimer() CkWallTimer()
#else
#define BGTimer() 0.0
#endif

class BgNode ;
class InBuffer ;
class SchedulerQ ;
class ThreadInfo ;
class PacketMsg ;
class Main ;
// CkQ is defined in charm++.h
typedef CkQ<PacketMsg*> MsgQ;

extern "C" void  BgInit(Main *) ;
extern "C" void* BgNodeInit(BgNode *) ;
extern "C" void  BgFinish() ;

typedef void (*BgHandler)(ThreadInfo *) ;

class CreateBgNodeMsg: public CMessage_CreateBgNodeMsg
{
public:
  int numCTh ;
  int numWTh ;
  int numBgX ;
  int numBgY ;
  int numBgZ ;
} ;
//~~~ end of class CreateBgNodeMsg ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PacketMsg: public CMessage_PacketMsg
{
public:
  int handlerID ;
  WorkType type;
  double sendTime ;
  double recvTime ;
} ;
//~~~ end of class PacketMsg ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Main: public Chare 
{
private:
  CkArgMsg *args ;
  int       numBgX, numBgY, numBgZ ;   
  double starttime, endtime;

public:
       Main(CkArgMsg *msg) ;
       ~Main() ;
  void Finish(void) ;

  void CreateBlueGene(CreateBgNodeMsg *msg) ;
  int          getNumArgs() ;
  const char** getArgs() ;
} ;

//~~~ end of class Main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BgNode: public ArrayElement3D
{
private:
  int          numCTh;
  int          numWTh;
  InBuffer*    inBuffer;
  PacketMsg**  matchTable;
  CthThread*   threadTable;
  BgHandler*   handlerTable;
  SchedulerQ*  workThQ;
  SchedulerQ*  commForMsg;
  SchedulerQ*  commForWork;
  MsgQ*        addMsgQ;
  CProxy_BgNode**** proxies;

public:
  int         numBgX ;
  int         numBgY ;
  int         numBgZ ;
  void*       nvData ;

public:   
       BgNode(CreateBgNodeMsg *msgPtr) ;
       ~BgNode() ; 
       BgNode(CkMigrateMessage *msgPtr) {} ;

  void registerHandler(int handlerID, BgHandler h) ;
  void addMessage(PacketMsg *msgPtr, int handlerID, WorkType type) ;

  void putMessage(PacketMsg *msgPtr) ;
  void assignMsg() ;
  void startCommTh(ThreadInfo *info) ;
  void startWorkTh(ThreadInfo *info) ;

  void sendPacket(int x, int y, int z, PacketMsg *msgPtr, 
                  int handlerID, WorkType type) ;
  void getXYZ(int& x, int& y, int& z) ;
  double getNumCTh() { return numCTh; }
  double getNumWTh() { return numWTh; }
  void finish() ;
} ;
//~~~ end of class BgNode


class InBuffer
{
  PacketMsg *msgs[SIZE_INBUFFER];
  int first, count;
  BgNode *bgNode ;
  MsgQ *mq;

public:
  InBuffer(BgNode *bgNodeRef) : 
    bgNode(bgNodeRef), first(0), count(0) 
  {
    for(int i=0;i<SIZE_INBUFFER;i++)
      msgs[i] = 0;
    mq = new MsgQ;
  }
  ~InBuffer()
  {
    for(int i=first, n=0; n < count; n++)
    {
      delete msgs[i];
      i = (i+1)%SIZE_INBUFFER;
    }
    delete mq;
  } 
  PacketMsg* getMessage(void)
  {
    PacketMsg *m = msgs[first];
    if(count) {
      msgs[first] = 0;
      first = (first+1)%SIZE_INBUFFER;
      count --;
      if(count==SIZE_INBUFFER-1) {
        PacketMsg *m1;
        if(m1=mq->deq())
          putMessage(m1);
      }
    }
    return m;
  }
  void putMessage(PacketMsg *msgPtr)
  {
    if(count==SIZE_INBUFFER)
    {
      mq->enq(msgPtr);
      return;
    }
    msgs[(first+count)%SIZE_INBUFFER] = msgPtr;
    count++;
    bgNode->assignMsg();
  }
} ;
//~~~ end of InBuffer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SchedulerQ
{
private:
  int *q;
  int front, count, max ;

public:
  SchedulerQ(int _max) : front(0), count(0), max(_max) { q = new int[_max]; }
  ~SchedulerQ() { delete[] q; }
  int  dequeThread(void)
  {
    int ret = (-1);
    if( count>0)
    { 
      ret = q[front]; 
      front = (front+1)%max; 
      count--;
    }
    return ret;
  }
  void enqueThread(int threadID)
  {
    q[(front+count)%max] = threadID;
    count++;
  }
} ;
//~~~ end of SchedulerQ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class ThreadInfo
{
public:
  BgNode *bgNode ;
  int     selfID ;
  void   *msg ;
  double currTime;
  double handlerStartTime;

public:
  double getTime(void)
  {
    double tp2 ;
    tp2 = BGTimer()*1e6;
    return (tp2 - handlerStartTime + currTime);
  }
  void sendPacket(int x, int y, int z, PacketMsg *msgPtr, int handlerID, 
                  WorkType type)
  {
    msgPtr->sendTime = getTime();
    bgNode->sendPacket(x,y,z,msgPtr,handlerID,type);
  }
} ;
//~~~ end of ThreadInfo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

extern "C" void startCommTh(ThreadInfo *info) ;
extern "C" void startWorkTh(ThreadInfo *info) ;

#endif
