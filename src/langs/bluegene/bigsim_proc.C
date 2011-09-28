#include <assert.h>
#include "blue.h"
#include "blue_impl.h"    	// implementation header file
#include "bigsim_record.h"

//#include "blue_timing.h" 	// timing module
#include "ckcheckpoint.h"

#include "bigsim_ooc.h"

#include "bigsim_debug.h"

#undef DEBUGLEVEL
#define DEBUGLEVEL 10

extern BgStartHandler  workStartFunc;
extern "C" void CthResumeNormalThread(CthThreadToken* token);

void correctMsgTime(char *msg);

CpvExtern(int      , CthResumeBigSimThreadIdx);

/**
  threadInfo methods
*/
void commThreadInfo::run()
{
  CpvAccess(CthResumeBigSimThreadIdx) = BgRegisterHandler((BgHandler)CthResumeNormalThread);

  tSTARTTIME = CmiWallTimer();

  if (!tSTARTED) {
    tSTARTED = 1;
//    InitHandlerTable();
    BgNodeStart(BgGetArgc(), BgGetArgv());
    /* bnv should be initialized */
  }

  threadQueue *commQ = myNode->commThQ;

  //int recvd=0; //for debugging only
  for (;;) {
    char *msg = getFullBuffer();
    if (!msg) { 
//      tCURRTIME += (CmiWallTimer()-tSTARTTIME);
      commQ->enq(CthSelf());
      DEBUGF(("[%d] comm thread suspend.\n", BgMyNode()));
      CthSuspend(); 
      DEBUGF(("[%d] comm thread assume.\n", BgMyNode()));
//      tSTARTTIME = CmiWallTimer();
      continue;
    }
    DEBUGF(("[%d] comm thread has a msg.\n", BgMyNode()));
	
	//printf("on node %d, comm thread process a msg %p with type %d\n", BgMyNode(), msg, CmiBgMsgType(msg));
    /* schedule a worker thread, if small work do it itself */
    if (CmiBgMsgType(msg) == SMALL_WORK) {
      if (CmiBgMsgRecvTime(msg) > tCURRTIME)  tCURRTIME = CmiBgMsgRecvTime(msg);
//      tSTARTTIME = CmiWallTimer();
      /* call user registered handler function */
      BgProcessMessage(this, msg);
    }
    else {
#if BIGSIM_TIMING
      correctMsgTime(msg);
#endif
    
    //recvd++;
    //DEBUGM(4, ("[N%d] C[%d] will add a msg (handler=%d | cnt=%d", BgMyNode(), id, CmiBgMsgHandle(msg), recvd));
    int msgLen = CmiBgMsgLength(msg);
    DEBUGM(4, (" | len: %d | type: %d | node id: %d | src pe: %d\n" , msgLen, CmiBgMsgType(msg), CmiBgMsgNodeID(msg), CmiBgMsgSrcPe(msg)));
      
     if (CmiBgMsgThreadID(msg) == ANYTHREAD) {
        DEBUGF(("anythread, call addBgNodeMessage\n"));
        addBgNodeMessage(msg);			/* non-affinity message */
	DEBUGM(4, ("The message is added to node\n\n"));
      }
      else {
        DEBUGF(("[N%d] affinity msg, call addBgThreadMessage to tID:%d\n", 
			BgMyNode(), CmiBgMsgThreadID(msg)));
	
        addBgThreadMessage(msg, CmiBgMsgThreadID(msg));
	DEBUGM(4, ("The message is added to thread(%d)\n\n", CmiBgMsgThreadID(msg)));
      }
    }
    /* let other communication thread do their jobs */
//    tCURRTIME += (CmiWallTimer()-tSTARTTIME);
    if (!schedule_flag) CthYield();
    tSTARTTIME = CmiWallTimer();
  }
}

void BgScheduler(int nmsg)
{
  ASSERT(tTHREADTYPE == WORK_THREAD);
  // end current log
  int isinterrupt = 0;
  if (genTimeLog) {
    if (BgIsInALog(tTIMELINEREC)) {
      isinterrupt = 1;
      BgLogEntryCommit(tTIMELINEREC);
      tTIMELINEREC.bgPrevLog = BgLastLog(tTIMELINEREC);
    }
  }
  stopVTimer();

  ((workThreadInfo*)cta(threadinfo))->scheduler(nmsg);

  // begin a new log, and make dependency
  startVTimer();
  if (genTimeLog && isinterrupt) 
  {
    BgTimeLog *curlog = BgLastLog(tTIMELINEREC);
    BgTimeLog *newLog = BgStartLogByName(tTIMELINEREC, -1, (char*)"BgSchedulerEnd", BgGetCurTime(), curlog);
  }
}

void BgExitScheduler()
{
  ASSERT(tTHREADTYPE == WORK_THREAD);
  ((workThreadInfo*)cta(threadinfo))->stopScheduler();
}

void BgDeliverMsgs(int nmsg)
{
  if (nmsg == 0) nmsg=1;
  BgScheduler(nmsg);
}

//If AMPI_Init is called, then we begin to do out-of-core emulation
//in the case of emulating AMPI programs. This is based on the assumption
//that during initialization, the memory should be enough
//Later this is not necessary if the out-of-core emulation is triggered
//by the free memory available.

//The original version of scheduler
void workThreadInfo::scheduler(int count)
{
  ckMsgQueue &q1 = myNode->nodeQ;
  ckMsgQueue &q2 = myNode->affinityQ[id];

  int cycle = CsdStopFlag;

  int recvd = 0;
  for (;;) {
    char *msg=NULL;
    int e1 = q1.isEmpty();
    int e2 = q2.isEmpty();
    int fromQ2 = 0;		// delay the deq of msg from affinity queue

    // not deq from nodeQ assuming no interrupt in the handler
    if (e1 && !e2) { msg = q2[0]; fromQ2 = 1;}
//    else if (e2 && !e1) { msg = q1.deq(); }
    else if (e2 && !e1) { msg = q1[0]; }
    else if (!e1 && !e2) {
      if (CmiBgMsgRecvTime(q1[0]) < CmiBgMsgRecvTime(q2[0])) {
//        msg = q1.deq();
        msg = q1[0];
      }
      else {
        msg = q2[0];
        fromQ2 = 1;
      }
    }
    /* if no msg is ready, go back to sleep */
    if ( msg == NULL ) {
//      tCURRTIME += (CmiWallTimer()-tSTARTTIME);
      DEBUGM(4,("N[%d] work thread %d has no msg and go to sleep!\n", BgMyNode(), id));
      if (watcher) watcher->replay();
#if BIGSIM_OUT_OF_CORE && BIGSIM_OOC_PREFETCH
      if(bgUseOutOfCore){
          //thread scheduling point!!            
          workThreadInfo *thisThd = schedWorkThds->pop();
          //CmiPrintf("thisThd=%p, actualThd=%p, equal=%d qsize=%d\n", thisThd, this, thisThd==this, schedWorkThds->size()); 
          assert(thisThd==this);
      }
#endif     
      CthSuspend();

      DEBUGM(4, ("N[%d] work thread %d awakened!\n", BgMyNode(), id));      
      continue;
    }
#if BIGSIM_TIMING
    correctMsgTime(msg);
#if THROTTLE_WORK
    if (correctTimeLog) {
      if (CmiBgMsgRecvTime(msg) > gvt+ BG_LEASH) {
	double nextT = CmiBgMsgRecvTime(msg);
	int prio = (int)(nextT*PRIO_FACTOR)+1;
	if (prio < 0) {
	  CmiPrintf("PRIO_FACTOR %e is too small. \n", PRIO_FACTOR);
	  CmiAbort("BigSim time correction abort!\n");
	}
//CmiPrintf("Thread %d YieldPrio: %g gvt: %g leash: %g\n", id, nextT, gvt, BG_LEASH);
	CthYieldPrio(CQS_QUEUEING_IFIFO, sizeof(int), (unsigned int*)&prio);
	continue;
      }
    }
#endif
#endif   /* TIMING */
    DEBUGM(2, ("[N%d] work thread T%d has a msg with recvT:%e msgId:%d.\n", BgMyNode(), id, CmiBgMsgRecvTime(msg), CmiBgMsgID(msg)));

//if (tMYNODEID==0)
//CmiPrintf("[%d] recvT: %e\n", tMYNODEID, CmiBgMsgRecvTime(msg));

    if (CmiBgMsgRecvTime(msg) > currTime) {
      tCURRTIME = CmiBgMsgRecvTime(msg);
    }

#if 1
    if (fromQ2 == 1) q2.deq();
    else q1.deq();
#endif


    recvd ++;  
    DEBUGM(4, ("[N%d] W[%d] will process a msg (handler=%d | cnt=%d", BgMyNode(), id, CmiBgMsgHandle(msg), recvd));
    int msgLen = CmiBgMsgLength(msg);
    DEBUGM(4, (" | len: %d | type: %d | node id: %d | src pe: %d\n" , msgLen, CmiBgMsgType(msg), CmiBgMsgNodeID(msg), CmiBgMsgSrcPe(msg)));
    for(int msgIndex=CmiBlueGeneMsgHeaderSizeBytes-1; msgIndex<msgLen; msgIndex++)
        DEBUGM(2, ("%d,", msg[msgIndex]));
    DEBUGM(2,("\n"));

    DEBUGM(4, ("[N%d] W[%d] now has %d msgs from own queue and %d from affinity before processing msg\n", BgMyNode(), id, q1.length(), q2.length()));


    //CmiMemoryCheck();

    // BgProcessMessage may trap into scheduler
    if(bgUseOutOfCore){
#if 0
    	if(startOutOfCore){
    	    DEBUGM(4, ("to execute in ooc mode\n"));
    	    if(isCoreOnDisk) this->broughtIntoMem();  
    	    BgProcessMessage(this, msg); //startOutOfCore may be changed in processing this msg (AMPI_Init)    
    
    	    if(startOOCChanged){ 
                //indicate AMPI_Init is called and before it is finished, out-of-core is not executed
                //just to track the 0->1 change phase (which means MPI_Init is finished)
                //the 1->0 phase is not tracked because "startOutOfCore" is unset so that
                //the next processing of a msg will not go into this part of code
                startOOCChanged=0;
    	    }else{
                //if(!isCoreOnDisk) { //the condition is added for virtual process
                    this->takenOutofMem();
                //}
    	    }
    	}else{
    	    DEBUGM(4, ("to execute not in ooc mode\n"));
    	    if(isCoreOnDisk) {
                CmiAbort("This should never be entered!\n");
                this->broughtIntoMem();  
    	    }
    	    //put before processing msg since thread may be scheduled during processing the msg
    	    BgProcessMessage(this, msg);
    	}
#else
        //schedWorkThds->print();
        bgOutOfCoreSchedule(this);
        BG_ENTRYSTART(msg);
#if BIGSIM_OOC_PREFETCH 
#if !BIGSIM_OOC_NOPREFETCH_AIO
        //do prefetch here for the next different thread in queue (schedWorkThds)
		assert(schedWorkThds->peek(0)==this);
        for(int offset=1; offset<schedWorkThds->size(); offset++) {
            workThreadInfo *nThd = schedWorkThds->peek(offset);
            //if nThd's core has been dumped to disk, then we could prefetch its core.
            //otherwise, it is the first time for the thread to process a message, thus
            //no need to resort to disk to find its core
            if(nThd!=this && !checkThreadInCore(nThd) 
               && nThd->isCoreOnDisk && oocPrefetchSpace->occupiedThd==NULL) {
                oocPrefetchSpace->newPrefetch(nThd);
            }
        }
#endif
#endif
        BgProcessMessage(this, msg);
#endif
    }else{
        DEBUGM(4, ("to execute not in ooc mode\n"));
        BG_ENTRYSTART(msg);
        BgProcessMessage(this, msg);
    }
    
    DEBUGM(4, ("[N%d] W[%d] now has %d msgs from own queue and %d from affinity after processing msg\n\n", BgMyNode(), id, q1.length(), q2.length()));
    BG_ENTRYEND();

    // counter of processed real mesgs
    stateCounters.realMsgProcCnt++;

    // NOTE: I forgot why I delayed the dequeue after processing it
#if 0
    if (fromQ2 == 1) q2.deq();
    else q1.deq();
#endif

    //recvd ++;

    //DEBUGF(("[N%d] work thread T%d finish a msg.\n", BgMyNode(), id));
    //CmiPrintf("[N%d] work thread T%d finish a msg (msg=%s, cnt=%d).\n", BgMyNode(), id, msg, recvd);
    
    if ( recvd == count) return;

    if (cycle != CsdStopFlag) break;

    /* let other work thread do their jobs */
    if (schedule_flag) {
    DEBUGF(("[N%d] work thread T%d suspend when done - %d to go.\n", BgMyNode(), tMYID, q2.length()));
    CthSuspend();
    DEBUGF(("[N%d] work thread T%d awakened here.\n", BgMyNode(), id));
    }
    else {
#if BIGSIM_OUT_OF_CORE && BIGSIM_OOC_PREFETCH
    //thread scheduling point!!
    //Suspend and put itself back to the end of the queue
      if(bgUseOutOfCore){
          workThreadInfo *thisThd = schedWorkThds->pop();
          //CmiPrintf("thisThd=%p, actualThd=%p, equal=%d qsize=%d\n", thisThd, this, thisThd==this, schedWorkThds->size()); 
          assert(thisThd==this);
          schedWorkThds->push(this);
      }
#endif
    CthYield();

    }
  }

  CsdStopFlag --;
}

static FILE *openBinaryReplayFile(int pe, const char* flags) {
        char fName[200];
        sprintf(fName,"bgfullreplay_%06d.log",pe);
        FILE *f;
        // CkPrintf("openBinaryReplayFile %s\n", fName);
        f = fopen(fName, flags);
        if (f==NULL) {
                CkPrintf("[%d] Could not open replay file '%s'.\n",
                        CkMyPe(),fName);
                CkAbort("openBinaryReplayFile> Could not open replay file");
        }
        return f;
}

void workThreadInfo::run()
{
  tSTARTTIME = CmiWallTimer();

    //  register for charm++ applications threads
  CpvAccess(CthResumeBigSimThreadIdx) = BgRegisterHandler((BgHandler)CthResumeNormalThread);

  if (cva(bgMach).record != -1 && ( cva(bgMach).recordprocs.isEmpty() && cva(bgMach).recordnodes.isEmpty() || (!cva(bgMach).recordprocs.isEmpty() && cva(bgMach).recordprocs.includes(BgGetGlobalWorkerThreadID()))) || (!cva(bgMach).recordnodes.isEmpty() && cva(bgMach).recordnodes.includes(BgMyNode())))
  {
    watcher = new BgMessageRecorder(openBinaryReplayFile(BgGetGlobalWorkerThreadID(), "wb"), cva(bgMach).recordnode!=-1);
  }
  if (cva(bgMach).replay != -1)
  {
    watcher = new BgMessageReplay(openBinaryReplayFile(cva(bgMach).replay, "rb"), 0);
  }
  if (cva(bgMach).replaynode != -1)
  {
    int startpe, endpe;
    BgRead_nodeinfo(cva(bgMach).replaynode, startpe, endpe);
    watcher = new BgMessageReplay(openBinaryReplayFile(startpe+BgGetThreadID(), "rb"), 1);
  }

//  InitHandlerTable();
  // before going into scheduler loop, call workStartFunc
  // in bg charm++, it normally is initCharm
  if (workStartFunc) {
    DEBUGF(("[N%d] work thread %d start.\n", BgMyNode(), id));
    // timing
    startVTimer();
    BG_ENTRYSTART((char*)NULL);
    char **Cmi_argvcopy = CmiCopyArgs(BgGetArgv());
    workStartFunc(BgGetArgc(), Cmi_argvcopy);
    BG_ENTRYEND();
    stopVTimer();
  }

  scheduler(-1);

  CmiAbort("worker thread should never end!\n");
}

void workThreadInfo::addAffMessage(char *msgPtr)
{
  ckMsgQueue &que = myNode->affinityQ[id];
  que.enq(msgPtr);
  if (schedule_flag) {
  /* don't awake directly, put into a priority queue sorted by recv time */
  double nextT = CmiBgMsgRecvTime(msgPtr);
  CthThread tid = me;
  unsigned int prio = (unsigned int)(nextT*PRIO_FACTOR)+1;
  DEBUGF(("[%d] awaken worker thread with prio %d.\n", tMYNODEID, prio));
  CthAwakenPrio(tid, CQS_QUEUEING_IFIFO, sizeof(int), &prio);
  }
  else {
  if (que.length() == 1) {
#if BIGSIM_OUT_OF_CORE && BIGSIM_OOC_PREFETCH
      //thread scheduling point!!
      if(bgUseOutOfCore) schedWorkThds->push(this);
#endif
    CthAwaken(me);
  }
  }
}


//=====Begin of stuff related with out-of-core scheduling======

extern int _BgOutOfCoreFlag;

//The Out-of-core implementation reuses the functions in the checkpoint
//module. For standalone bigsim emulator programs, the definitions of
//those functions (inside libck.a ) are not linked into the final binary.
//Considering the tricky relations between libck.a and standalone bigsim
//emulator program, defining the following macro to temporarily resolve the
//issue. When compiling the bigsim emulator with out-of-core support for
//normal charm++ or AMPI programs, this macro should be turned to 1.
//-Chao Mei
//#undef BIGSIM_OUT_OF_CORE
//#define BIGSIM_OUT_OF_CORE 1

void threadInfo::broughtIntoMem(){
    DEBUGM(5, ("=====[N%d] work thread T[%d] into mem=====.\n", BgMyNode(), id));
    //CmiPrintStackTrace(0);    

    //int idx =( (workThreadInfo *)this)->preStsIdx;
    //CmiPrintf("pe[%d]: on node[%d] thread[%d] has bufsize %ld\n", CkMyPe(), BgMyNode(), id, thdsOOCPreStatus[idx].bufsize);

    assert(isCoreOnDisk==1);

#if BIGSIM_OOC_PREFETCH
#if !BIGSIM_OOC_NOPREFETCH_AIO
   if(oocPrefetchSpace->occupiedThd != this){
	//in this case, the prefetch is not for this workthread

        _BgOutOfCoreFlag=2;
        char *dirname = "/tmp/CORE";
        //every body make dir in case it is local directory
        //CmiMkdir(dirname);
        char filename[128];
        sprintf(filename, "%s/%d.dat", dirname, globalId);
        FILE* fp = fopen(filename, "r");
        if(fp==NULL){
            printf("Error: %s cannot be opened when bringing thread %d to core\n", filename, globalId);
            return;
        }
    
        //_BgOutOfCoreFlag=2;
        PUP::fromDisk p(fp);
        //out-of-core is not a real migration, so turn off the notifyListener option
        #if BIGSIM_OUT_OF_CORE
        CkPupArrayElementsData(p, 0);
        #endif
        fclose(fp);
   }else
#endif
{    
        _BgOutOfCoreFlag=2;
        workThreadInfo *wthd = (workThreadInfo *)this;
#if BIGSIM_OOC_NOPREFETCH_AIO
        oocPrefetchSpace->newPrefetch(wthd);
#endif
        while(thdsOOCPreStatus[wthd->preStsIdx].isPrefetchFinished==0);
        
        PUP::fromMem p(oocPrefetchSpace->bufspace);
        #if BIGSIM_OUT_OF_CORE
        CkPupArrayElementsData(p, 0);
        #endif
        oocPrefetchSpace->resetPrefetch();        
	}
#else
	//not doing prefetch optimization for ooc emulation
	_BgOutOfCoreFlag=2;
	const char *dirname = "/tmp/CORE";
	//every body make dir in case it is local directory
	//CmiMkdir(dirname);
	char filename[128];
	sprintf(filename, "%s/%d.dat", dirname, globalId);
	FILE* fp = fopen(filename, "r");
	if(fp==NULL){
		printf("Error: %s cannot be opened when bringing thread %d to core\n", filename, globalId);
		return;
	}

	//_BgOutOfCoreFlag=2;
	PUP::fromDisk p(fp);
	//out-of-core is not a real migration, so turn off the notifyListener option
	#if BIGSIM_OUT_OF_CORE
	CkPupArrayElementsData(p, 0);
	#endif
	fclose(fp);	
#endif	


    _BgOutOfCoreFlag=0;
    
    //printf("mem usage after thread %d in: %fMB\n",globalId, CmiMemoryUsage()/1024.0/1024.0);    
    isCoreOnDisk = 0;
}

void threadInfo::takenOutofMem(){
    DEBUGM(5, ("=====[N%d] work thread T[%d] outof mem=====.\n", BgMyNode(), id));
    //CmiPrintStackTrace(0);    

    assert(isCoreOnDisk==0);

    _BgOutOfCoreFlag=1;
    const char *dirname = "/tmp/CORE";
    //every body make dir in case it is local directory
    CmiMkdir(dirname);
    char filename[128];
    sprintf(filename, "%s/%d.dat", dirname, globalId);
    FILE* fp = fopen(filename, "wb");
    if(fp==NULL){
        printf("Error: %s cannot be opened when bringing thread %d to core\n", filename, globalId);
        return;
    }

    //_BgOutOfCoreFlag=1;
    PUP::toDisk p(fp);
    //out-of-core is not a real migration, so turn off the notifyListener option
    p.becomeDeleting();

    #if BIGSIM_OUT_OF_CORE
    CkPupArrayElementsData(p, 0);
    #endif 

    CmiUInt8 fsize = 0;
    fseek(fp, 0, SEEK_END);
    fsize = ftell(fp);
    //set this thread's memory usage
    memUsed = fsize/1024.0/1024.0;

    fflush(fp);

    fclose(fp);

    DEBUGM(6,("Before removing array elements on proc[%d]\n", globalId));   
 
    #if BIGSIM_OUT_OF_CORE
    CkRemoveArrayElements();
    #endif

    _BgOutOfCoreFlag=0;
    isCoreOnDisk=1;
    //printf("mem usage after thread %d out: %fMB\n", globalId, CmiMemoryUsage()/1024.0/1024.0);  
#if BIGSIM_OOC_PREFETCH
    thdsOOCPreStatus[preStsIdx].bufsize= fsize;
#endif	
}

/*
static int outCore()
{
  _BgOutOfCoreFlag = 1;
  //printf("memory usage: out core before remove arrays %fMB\n", CmiMemoryUsage()/1024.0/1024.0);
  int id = BgGetGlobalWorkerThreadID();
  //printf("[%d] Out core\n", id);
  char *dirname = "/tmp/COREORIG";
  // every body make dir in case it is local directory
  CmiMkdir(dirname);
  char filename[128];
  sprintf(filename,"%s/%d.dat",dirname,id);
  FILE* fp = fopen(filename,"wb");
  if (fp == 0) {
    perror("file");
    exit(1);
  }
  PUP::toDisk p(fp);
  CkPupArrayElementsData(p, 0);
  fdatasync(fileno(fp));
  fclose(fp);

  CkRemoveArrayElements();
  //printf("memory usage: out core after remove arrays %fMB\n", CmiMemoryUsage()/1024.0/1024.0);
  _BgOutOfCoreFlag = 0;
}

static int inCore()
{
  _BgOutOfCoreFlag = 2;
//CmiPrintStackTrace(0);
  //printf("memory usage: in core before load core %fMB\n", CmiMemoryUsage()/1024.0/1024.0);
  int id = BgGetGlobalWorkerThreadID();
  //printf("[%d] In core\n", id);
  char *dirname = "/tmp/COREORIG";
  // every body make dir in case it is local directory
  CmiMkdir(dirname);
  char filename[128];
  sprintf(filename,"%s/%d.dat",dirname,id);
  FILE* fp = fopen(filename,"r");
  if (fp == NULL) return 0;
  PUP::fromDisk p(fp);
  CkPupArrayElementsData(p, 0);
  fdatasync(fileno(fp));
  fclose(fp);
  //printf("memory usage: in core after load core %fMB\n", CmiMemoryUsage()/1024.0/1024.0);
  _BgOutOfCoreFlag = 0;
}
*/

//=====End of stuff related with out-of-core scheduling=======
