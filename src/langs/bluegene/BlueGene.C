// File: BlueGene.C
// Author: Arun Kumar Singla <asingla@uiuc.edu> Nov 2000
// Updated: Milind Bhandarkar 

#define protected public
#include "BlueGene.h"

#define CYCLES_PER_HOP     5
#define CYCLES_PER_CORNER  75
#define CYCLE_TIME_FACTOR  0.001  // one cycle = nanosecond = 10^(-3) us  

CkChareID mainID ;
CkArrayID bgArrayID ;

//Main methods 
/*
 * 'Main' calls 'BgInit', defined in user application for initialization 
 * of bluegene machine.
 */
Main::Main(CkArgMsg *msg)
{
  args = msg ;
  mainID = thishandle ;

/*
  if (msg->argc < 6) { 
    CkAbort("Usage: <program> <x> <y> <z> <numCommTh> <numWorkTh>\n"); 
  }
    
  CreateBgNodeMsg *bgNodeMsg = new CreateBgNodeMsg;
  numBgX = bgNodeMsg->numBgX = atoi(msg->argv[1]) ;
  numBgY = bgNodeMsg->numBgY = atoi(msg->argv[2]) ; 
  numBgZ = bgNodeMsg->numBgZ = atoi(msg->argv[3]) ;
  bgNodeMsg->numCTh = atoi(msg->argv[4]) ;
  bgNodeMsg->numWTh = atoi(msg->argv[5]) ;

  CreateBlueGene(bgNodeMsg) ;
*/

  BgInit(this) ;
  starttime = CkWallTimer();
  return ;
}

Main::~Main()
{
  CProxy_BgNode bgArray(bgArrayID) ;

  for( int i=0; i<numBgX; i++ )
  for( int j=0; j<numBgY; j++ )
  for( int k=0; k<numBgZ; k++ )
  bgArray(i, j, k).destroy() ;
}

int Main::getNumArgs()
{
  return args->argc ;
}

const char** Main::getArgs()
{
  return (const char**)args->argv ;
}

/*
 * 'CreateBlueGene' initializes BlueGene machine
 * Specifies the machine configuration
 *  - Number of BgNodes in X, Y, and Z dimension
 *  - Number of Communication threads per BgNode
 *  - Number of Worker threads per BgNode
 *  - more: would also include latencies, cycles per hop, cycles per corner, etc
 */
void Main::CreateBlueGene(CreateBgNodeMsg *msg)
{
  numBgX = atoi(msg->argv[1]) ;
  numBgY = atoi(msg->argv[2]) ;
  numBgZ = atoi(msg->argv[3]) ;

  bgArrayID = CProxy_BgNode::ckNew() ;
  CProxy_BgNode bgArray(bgArrayID) ;  
  for( int i=0; i< numBgX; i++ )
  for( int j=0; j< numBgY; j++ )
  for( int k=0; k< numBgZ; k++ )
  {
    CreateBgNodeMsg *tempMsg = new CreateBgNodeMsg ;
    //tempMsg->numCTh = msg->numCTh ;
    //tempMsg->numWTh = msg->numWTh ;
    //tempMsg->numBgX = msg->numBgX ;
    //tempMsg->numBgY = msg->numBgY ;
    //tempMsg->numBgZ = msg->numBgZ ;
    tempMsg->argc   = msg->argc   ;
    tempMsg->argv   = msg->argv   ;
    bgArray(i, j, k).insert(tempMsg) ;
  }
  bgArray.doneInserting() ;
  delete msg ;
}

/*
 * 'Finish' is a simple implementation for informing the system that 
 * application has finished.
 * more: should be modified to include quiescense detection: no hurry.
 */
void Main::Finish(void) 
{
  endtime = CkWallTimer();
  ckout << "Total time : " << (endtime-starttime) << " seconds" << endl;
  CkExit();
}

static void defaultHandler(ThreadInfo *)
{
  CkAbort("BG> Invalid Handler called.\n");
}

//BgNode methods~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/*
 * 'BgNode' is initialization of a Blue Gene Node/CHIP.
 *  - Create a (node-private) inBuffer, schedulerQ's, internal tables
 *  - Creates all communication and worker threads.
 *  - Calls 'BgNodeInit', user defined application function, where user registers 
 *    the handlers for this BgNode and may trigger some message as microtasks or may 
 *    send some messages to other nodes.
 */
BgNode::BgNode(CreateBgNodeMsg *msg)
{
  int i;

  /*
  numCTh = msg->numCTh ;
  numWTh = msg->numWTh ;
  numBgX = msg->numBgX ;
  numBgY = msg->numBgY ;
  numBgZ = msg->numBgZ ;
  */
  numBgX = atoi(msg->argv[1]) ;
  numBgY = atoi(msg->argv[2]) ;
  numBgZ = atoi(msg->argv[3]) ;
  numCTh = atoi(msg->argv[4]) ;
  numWTh = atoi(msg->argv[5]) ;
  argc   = msg->argc ;
  argv   = msg->argv ;

  ckout << "constructor for bgNode " << thisIndex.x <<" "<< thisIndex.y <<" "<< thisIndex.z
  		<< " RANGE: " << atoi(msg->argv[6]) 
        << " , on processor " << CkMyPe() << endl;

  inBuffer     = new InBuffer(this) ;
  workThQ      = new SchedulerQ(numWTh) ;
  commForMsg   = new SchedulerQ(numCTh) ;
  commForWork  = new SchedulerQ(numCTh) ;
  matchTable   = new PacketMsg*[numCTh+numWTh] ;
  threadTable  = new CthThread[numCTh+numWTh] ;
  handlerTable = new BgHandler[MAX_HANDLERS] ;
  addMsgQ      = new MsgQ;

  for(i=0; i<MAX_HANDLERS; i++)
    handlerTable[i] = (BgHandler) defaultHandler;
  for(i=0; i<(numCTh+numWTh); i++)
    matchTable[i] = 0;

 CProxy_BgNode bgArray(bgArrayID) ;
  /*
  proxies = new CProxy_BgNode***[numBgX];
  for(int i=0;i<numBgX;i++) {
    proxies[i] = new CProxy_BgNode**[numBgY];
    for(int j=0;j<numBgY;j++) {
      proxies[i][j] = new CProxy_BgNode*[numBgZ];
      for(int k=0;k<numBgZ;k++) {
        proxies[i][j][k] = new CProxy_BgNode(bgArrayID, CkArrayIndex3D(i,j,k));
        //proxies[i][j][k] = new CProxy_BgNode(bgArrayID, new CkArrayIndex3D(i,j,k));
      }
    }
  }
  */


  //Create Communication Thread and enter in threadTable
  for(i=0; i<numCTh; i++)
  {
    ThreadInfo *info = new ThreadInfo() ;
    info->bgNode = this ;
    info->selfID = i ;
    CthThread t = CthCreate((CthVoidFn)::startCommTh, (void*)info , 2048) ;
    threadTable[i] = t ;
    CthAwaken(t) ;
  }

  //Create worker Thread and enter in threadTable
  for(i=0; i<numWTh; i++)
  {
    ThreadInfo *info = new ThreadInfo() ;
    info->bgNode = this ;
    info->selfID = i + numCTh ;
    CthThread t = CthCreate((CthVoidFn)::startWorkTh, (void*)info , 2048) ;
    threadTable[i+numCTh] = t;
    CthAwaken(t) ;
  }

  nvData = (void*)BgNodeInit(this) ;
  delete msg ;
}

BgNode::~BgNode()
{
  delete inBuffer ;
  delete[] matchTable ;
  delete[] threadTable ;
  delete[] handlerTable ;
  delete workThQ ;
  delete commForMsg ;
}

void BgNode::registerHandler(int handlerID, BgHandler h) 
{
  if(handlerID >= MAX_HANDLERS)
    CkAbort("BG> Handler ID exceeded maximum.\n");
  handlerTable[handlerID] = h;
}

/*
 * Assign the MicroTask in PacketMsg to a free communication thread of the same BgNode.
 * If all the communication threads are busy, allocate it to default communication thread '0'.
 * The communication thread would schedule it.
 * more: change this default stuff to an appropriate load-balanced-strategy
 */
 //more: change it
void BgNode::addMessage(PacketMsg *msgPtr, int handlerID, WorkType type)
{
  msgPtr->handlerID  = handlerID ;
  msgPtr->type = type ;

  //get a communication thread ID, if available, else  enque to added messages
  int commThID ;
  if(-1 == (commThID = commForMsg->dequeThread()))
  {
    addMsgQ->enq(msgPtr);
    return;
  }

  matchTable[commThID] = msgPtr;

  //get thread address and awaken it 
  CthThread t = threadTable[commThID] ;
  CthAwaken(t) ;
}

/* 
 * Put PacketMsg to inBuffer
 * If inBuffer is Full then sleep for some time and retry.
 */
void BgNode::putMessage(PacketMsg *msgPtr)
{
  if(inBuffer==NULL)
  	ckout << "Error: inbuffer is null" <<endl;
  else
  {
  	inBuffer->putMessage(msgPtr);
  }
}

#define ABS(x) (((x)<0)? -(x) : (x))

static inline double MSGTIME(int ox, int oy, int oz, int nx, int ny, int nz)
{
  int xd=ABS(ox-nx), yd=ABS(oy-ny), zd=ABS(oz-nz);
  int ncorners = 2;
  ncorners -= (xd?1:0 + yd?1:0 + zd?1:0);
  ncorners = (ncorners<0)?0:ncorners;
  return (ncorners*CYCLES_PER_CORNER + (xd+yd+zd)*CYCLES_PER_HOP)*CYCLE_TIME_FACTOR;
  //milind
}

/*
 * Send PacketMsg to BgNode[x,y,z]
 * Set appropriate timestamps:
 *  sendTime = currTime
 *      recvTime = estimated time to reach the destination BgNode
 */
void BgNode::sendPacket(int x, int y, int z, PacketMsg *msgPtr, 
                   int handlerID, WorkType type)
{
  msgPtr->handlerID = handlerID ;
  msgPtr->type = type ;

  msgPtr->recvTime = MSGTIME(thisIndex.x,thisIndex.y,thisIndex.z,x,y,z)
                   + msgPtr->sendTime;

  CProxy_BgNode bgArray(bgArrayID);
  bgArray(x, y, z).putMessage(msgPtr);
  //proxies[x][y][z]->putMessage(msgPtr);
}

/*
 * Assign the new Message in inBuffer to a free communication thread for scheduling.
 * If all the communication threads are busy, then leave message in inBuffer and don't worry.
 */
void BgNode::assignMsg()
{
  int commThID = commForMsg->dequeThread() ;
  
  if(-1 == commThID)
  {
    /* ckout << "leaving assignMsg: didn't find commTh " << endl ; */
    return ;
  }

  CthThread t = threadTable[commThID] ;
  CthAwaken(t) ;

  return ;
}

/*
 * Start the communication thread as soon as it is created.
 * The thread calls 'getMessage' on inBuffer to see if there is any 
 * pending message.
 *   If any, then get it from inBuffer and schedule it.
 *  If new message is a small piece of work i.e. type == SMALL_WORK, execute it
 *      else assign it to a free worker thread
 *
 * If there is no pending message in inBuffer, it looks in MatchTable to see if there is
 * any microtask allocated to it. If any, it schedules it.
 *  If the microtask is small piece of work then execute it and remove from matchTable 
 *   else assign it to a free worker thread
 *
 * If no message, then sleep until no new message or microtask awakens it.
 */
void BgNode::startCommTh(ThreadInfo *info)
{
  PacketMsg *msgPtr = 0;
  int selfID = info->selfID;
  info->currTime = 0 ;

  //double tp1, tp2 ;
  //tp1 = CkWallTimer()*1e6;
  while(true)
  {
    msgPtr = 0; 

    msgPtr = inBuffer->getMessage();
    if(msgPtr==0) { msgPtr=matchTable[selfID]; matchTable[selfID] = 0; }
    if(msgPtr==0) { msgPtr=addMsgQ->deq(); }
      
    if(msgPtr==0)
    {
      //suspend thread
      //tp2 = CkWallTimer()*1e6;
      //info->currTime += (tp2-tp1) ;
      info->currTime += 0.001 ;
      commForMsg->enqueThread(selfID) ;
      CthSuspend() ;
      //tp1 = CkWallTimer()*1e6;
      continue ;
    }

    if(SMALL_WORK==msgPtr->type)
    {
      BgHandler handler = handlerTable[msgPtr->handlerID];
      info->msg    = (void*)msgPtr ; 
  
      //tp2 = CkWallTimer()*1e6;
      //info->currTime += (tp2-tp1) ;
      info->currTime += 0.001 ;
      //make timing adjustments for thread
      if(msgPtr->recvTime > info->currTime)
      info->currTime = msgPtr->recvTime ;
      //info->handlerStartTime = tp2 ;
  
      //tp1 = tp2 ;
      handler(info) ;
      info->currTime = info->getTime();
      continue ;
    }
    else if(msgPtr->type==LARGE_WORK)
    {
      //get a worker thread ID, if available, else do polling
      int workThID ; 
      while(-1 == (workThID = workThQ->dequeThread()))
      {
        //tp2 = CkWallTimer()*1e6;
        //info->currTime += (tp2-tp1) ;
        info->currTime += 0.001 ;
        commForWork->enqueThread(selfID);
        CthSuspend();
        //tp1 = CkWallTimer()*1e6;
      }

      matchTable[workThID] = msgPtr;

      //get thread address and awaken it 
      CthThread t = threadTable[workThID] ;
      CthAwaken(t) ;
      continue ;
    }
    else
      ckout << "ERROR: Unidentified thread category" << endl ;
  }
}

/*
 * Start the worker thread as soon as it is created.
 * It looks in matchTable to see if there is any message or microtask allocated to it.
 *  If any, then execute it.
 * If no work, then sleep until communication thread awakens it.
 */
void BgNode::startWorkTh(ThreadInfo *info)
{
  PacketMsg *msgPtr = 0;
  int selfID = info->selfID;
  info->currTime = 0 ;

  //double tp1, tp2 ;
  //tp1 = CkWallTimer()*1e6;
  while(true)
  {
    msgPtr = 0 ;

    msgPtr = matchTable[selfID];
    matchTable[selfID] = 0;

    if(NULL==msgPtr) 
    {
      //tp2 = CkWallTimer()*1e6;
      //info->currTime += (tp2-tp1) ;
      info->currTime += 0.001;
      workThQ->enqueThread(selfID) ;
      int commID = commForWork->dequeThread();	//Awaken other sleeping worker thread, if any
      if (commID!=(-1))
        CthAwaken(threadTable[commID]);
      CthSuspend();
      //tp1 = CkWallTimer()*1e6;
      continue ;
    }

    //get handler for this message and execute it
    BgHandler handler = handlerTable[msgPtr->handlerID] ;  
    info->msg    = (void*)msgPtr ; 

    //tp2 = CkWallTimer()*1e6;
    //info->currTime += (tp2-tp1) ;
    info->currTime += 0.001;
    //make timing adjustments for thread
    if(msgPtr->recvTime > info->currTime)
      info->currTime = msgPtr->recvTime ;
    //info->handlerStartTime = tp2 ;

    //tp1 = tp2 ;
    handler(info) ;
    info->currTime = info->getTime();
  }
}

void BgNode::getXYZ(int& i, int& j, int& k)
{
  i = thisIndex.x ; j = thisIndex.y ; k = thisIndex.z ;
}

void BgNode::finish(void)
{
  //more: should I delete all the node components here
  CProxy_Main main(mainID) ;
  main.Finish() ;
}

/***************************************************************************/
/*
 * Convere system requires the startfunction of threads should be in C
 */
extern "C" void startCommTh(ThreadInfo *info) 
{
  (info->bgNode)->startCommTh(info) ;
}

extern "C" void startWorkTh(ThreadInfo *info)
{
  (info->bgNode)->startWorkTh(info) ;
}

#include "BlueGene.def.h"
