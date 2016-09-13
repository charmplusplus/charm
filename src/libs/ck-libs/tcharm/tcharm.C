/*
Threaded Charm++ "Framework Framework"

Orion Sky Lawlor, olawlor@acm.org, 11/19/2001
 */
#include "tcharm_impl.h"
#include "tcharm.h"
#include "mempool.h"
#include "ckevacuation.h"
#include <ctype.h>

#if 0 
    /*Many debugging statements:*/
#    define DBG(x) ckout<<"["<<thisIndex<<","<<CkMyPe()<<"] TCHARM> "<<x<<endl;
#    define DBGX(x) ckout<<"PE("<<CkMyPe()<<") TCHARM> "<<x<<endl;
#else
    /*No debugging statements*/
#    define DBG(x) /*empty*/
#    define DBGX(x) /*empty*/
#endif

CtvDeclare(TCharm *,_curTCharm);

#if CMK_USE_MEMPOOL_ISOMALLOC
extern "C"
{
CtvExtern(mempool_type *, threadpool);
}
#endif

static int lastNumChunks=0;

class TCharmTraceLibList {
	enum {maxLibs=20,maxLibNameLen=15};
	//List of libraries we want to trace:
	int curLibs;
	char libNames[maxLibs][maxLibNameLen];
	int checkIfTracing(const char *lib) const
	{
		for (int i=0;i<curLibs;i++) 
			if (0==strcmp(lib,libNames[i]))
				return 1;
		return 0;
	}
public:
	TCharmTraceLibList() {curLibs=0;}
	void addTracing(const char *lib) 
	{ //We want to trace this library-- add its name to the list.
		CkPrintf("TCHARM> Will trace calls to library %s\n",lib);
		int i;
		for (i=0;0!=*lib;i++,lib++)
			libNames[curLibs][i]=tolower(*lib);
		libNames[curLibs][i]=0;
		// if already tracing, skip
		if (checkIfTracing(libNames[curLibs])) return;
		curLibs++;
	}
	inline int isTracing(const char *lib) const {
		if (curLibs==0) return 0; //Common case
		else return checkIfTracing(lib);
	}
};
static TCharmTraceLibList tcharm_tracelibs;
static int tcharm_nomig=0, tcharm_nothreads=0;
static int tcharm_stacksize=1*1024*1024; /*Default stack size is 1MB*/
static int tcharm_initted=0;
CkpvDeclare(int, mapCreated);
static CkGroupID mapID;
static char* mapping = NULL;

void TCharm::nodeInit(void)
{
}

void TCharm::procInit(void)
{
  CtvInitialize(TCharm *,_curTCharm);
  CtvAccess(_curTCharm)=NULL;
  tcharm_initted=1;
  CtgInit();

  CkpvInitialize(int, mapCreated);
  CkpvAccess(mapCreated) = 0;

  // called on every pe to eat these arguments
  char **argv=CkGetArgv();
  tcharm_nomig=CmiGetArgFlagDesc(argv,"+tcharm_nomig","Disable migration support (debugging)");
  tcharm_nothreads=CmiGetArgFlagDesc(argv,"+tcharm_nothread","Disable thread support (debugging)");
  tcharm_nothreads|=CmiGetArgFlagDesc(argv,"+tcharm_nothreads",NULL);
  char *traceLibName=NULL;
  while (CmiGetArgStringDesc(argv,"+tcharm_trace",&traceLibName,"Print each call to this library"))
      tcharm_tracelibs.addTracing(traceLibName);
  // CmiGetArgIntDesc(argv,"+tcharm_stacksize",&tcharm_stacksize,"Set the thread stack size (default 1MB)");
  char *str;
  if (CmiGetArgStringDesc(argv,"+tcharm_stacksize",&str,"Set the thread stack size (default 1MB)"))  {
    if (strpbrk(str,"M")) {
      sscanf(str, "%dM", &tcharm_stacksize);
      tcharm_stacksize *= 1024*1024;
    }
    else if (strpbrk(str,"K")) {
      sscanf(str, "%dK", &tcharm_stacksize);
      tcharm_stacksize *= 1024;
    }
    else {
      sscanf(str, "%d", &tcharm_stacksize);
    }
    if (CkMyPe() == 0)
      CkPrintf("TCharm> stack size is set to %d.\n", tcharm_stacksize);
  }
  if (CkMyPe()!=0) { //Processor 0 eats "+vp<N>" and "-vp<N>" later:
  	int ignored;
  	while (CmiGetArgIntDesc(argv,"-vp",&ignored,NULL)) {}
  	while (CmiGetArgIntDesc(argv,"+vp",&ignored,NULL)) {}
  }
  if (CkMyPe()==0) { // Echo various debugging options:
    if (tcharm_nomig) CmiPrintf("TCHARM> Disabling migration support, for debugging\n");
    if (tcharm_nothreads) CmiPrintf("TCHARM> Disabling thread support, for debugging\n");
  }
  if (CkpvAccess(mapCreated)==0) {
    if (0!=CmiGetArgString(argv, "+mapping", &mapping)){
    }
    CkpvAccess(mapCreated)=1;
  }
}

void TCHARM_Api_trace(const char *routineName,const char *libraryName)
{
	if (!tcharm_tracelibs.isTracing(libraryName)) return;
	TCharm *tc=CtvAccess(_curTCharm);
	char where[100];
	if (tc==NULL) sprintf(where,"[serial context on %d]",CkMyPe());
	else sprintf(where,"[%p> vp %d, p %d]",(void *)tc,tc->getElement(),CkMyPe());
	CmiPrintf("%s Called routine %s\n",where,routineName);
	CmiPrintStackTrace(1);
	CmiPrintf("\n");
}

// register thread start functions to get a function handler
// this is portable across heterogeneous platforms, or on machines with
// random stack/function pointer

static CkVec<TCHARM_Thread_data_start_fn> threadFnTable;

int TCHARM_Register_thread_function(TCHARM_Thread_data_start_fn fn)
{
  int idx = threadFnTable.size();
  threadFnTable.push_back(fn);
  return idx+1;                     // make 0 invalid number
}

TCHARM_Thread_data_start_fn getTCharmThreadFunction(int idx)
{
  CmiAssert(idx > 0);
  return threadFnTable[idx-1];
}

static void startTCharmThread(TCharmInitMsg *msg)
{
	DBGX("thread started");
	TCharm::activateThread();
       TCHARM_Thread_data_start_fn threadFn = getTCharmThreadFunction(msg->threadFn);
	threadFn(msg->data);
	TCharm::deactivateThread();
	CtvAccess(_curTCharm)->done();
}

TCharm::TCharm(TCharmInitMsg *initMsg_)
{
  initMsg=initMsg_;
  initMsg->opts.sanityCheck();
  timeOffset=0.0;
  if (tcharm_nothreads)
  { //Don't even make a new thread-- just use main thread
    tid=CthSelf();
  }
  else /*Create a thread normally*/
  {
    if (tcharm_nomig) { /*Nonmigratable version, for debugging*/
      tid=CthCreate((CthVoidFn)startTCharmThread,initMsg,initMsg->opts.stackSize);
    } else {
      tid=CthCreateMigratable((CthVoidFn)startTCharmThread,initMsg,initMsg->opts.stackSize);
    }
#if CMK_BIGSIM_CHARM
    BgAttach(tid);
    BgUnsetStartOutOfCore();
#endif
  }
  threadGlobals=CtgCreate(tid);
  CtvAccessOther(tid,_curTCharm)=this;
  asyncMigrate = false;
  isStopped=true;
	/* FAULT_EVAC*/
	AsyncEvacuate(true);
  exitWhenDone=initMsg->opts.exitWhenDone;
  isSelfDone = false;
  threadInfo.tProxy=CProxy_TCharm(thisArrayID);
  threadInfo.thisElement=thisIndex;
  threadInfo.numElements=initMsg->numElements;
  if (CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC)) {
  	heapBlocks=CmiIsomallocBlockListNew(tid);
  } else
  	heapBlocks=0;
  nUd=0;
  usesAtSync=true;
  run();
}

TCharm::TCharm(CkMigrateMessage *msg)
	:CBase_TCharm(msg)
{
  initMsg=NULL;
  tid=NULL;
  threadGlobals=NULL;
  threadInfo.tProxy=CProxy_TCharm(thisArrayID);
	AsyncEvacuate(true);
  heapBlocks=0;
}

void checkPupMismatch(PUP::er &p,int expected,const char *where)
{
	int v=expected;
	p|v;
	if (v!=expected) {
		CkError("FATAL ERROR> Mismatch %s pup routine\n",where);
		CkAbort("FATAL ERROR: Pup direction mismatch");
	}
}

void TCharm::pup(PUP::er &p) {
  //BIGSIM_OOC DEBUGGING
  //if(!p.isUnpacking()){
  //  CmiPrintf("TCharm[%d] packing: ", thisIndex);
  //  CthPrintThdStack(tid);
  //}

  checkPupMismatch(p,5134,"before TCHARM");
  p(isStopped); p(exitWhenDone); p(isSelfDone); p(asyncMigrate);
  p(threadInfo.thisElement);
  p(threadInfo.numElements);
  
  if (sema.size()>0){
  	CkAbort("TCharm::pup> Cannot migrate with unconsumed semaphores!\n");
  }

  DBG("Packing thread");
#if CMK_ERROR_CHECKING
  if (!p.isSizing() && !isStopped && !CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC)){
    if(_BgOutOfCoreFlag==0) //not doing out-of-core scheduling
	CkAbort("Cannot pup a running thread.  You must suspend before migrating.\n");
  }	
  if (tcharm_nomig) CkAbort("Cannot migrate with the +tcharm_nomig option!\n");
#endif

  //This seekBlock allows us to reorder the packing/unpacking--
  // This is needed because the userData depends on the thread's stack
  // and heap data both at pack and unpack time.
  PUP::seekBlock s(p,2);
  
  if (p.isUnpacking())
  {//In this case, unpack the thread & heap before the user data
    s.seek(1);
    pupThread(p);
    //Restart our clock: set it up so packTime==CkWallTimer+timeOffset
    double packTime;
    p(packTime);
    timeOffset=packTime-CkWallTimer();
  }
  
//Pack all user data
  // Set up TCHARM context for use during user's pup routines:
  if(isStopped) {
    CtvAccess(_curTCharm)=this;
    activateThread();
  }

  s.seek(0);
  checkPupMismatch(p,5135,"before TCHARM user data");
  p(nUd);
  for(int i=0;i<nUd;i++) {
    if (p.isUnpacking()) ud[i].update(tid);
    ud[i].pup(p);
  }
  checkPupMismatch(p,5137,"after TCHARM_Register user data");

  if(isStopped) {
    if (CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC))
      deactivateThread();
  }
  p|sud;           //  sud vector block can not be in isomalloc
  checkPupMismatch(p,5138,"after TCHARM_Global user data");
  
  // Tear down TCHARM context after calling user pup routines
  if(isStopped) {
    if (!CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC))
      deactivateThread();
    CtvAccess(_curTCharm)=NULL;
  }

  if (!p.isUnpacking())
  {//In this case, pack the thread & heap after the user data
    s.seek(1);
    pupThread(p);
    //Stop our clock:
    double packTime=CkWallTimer()+timeOffset;
    p(packTime);
  }
  
  s.endBlock(); //End of seeking block
  checkPupMismatch(p,5140,"after TCHARM");
  
  //BIGSIM_OOC DEBUGGING
  //if(p.isUnpacking()){
  //  CmiPrintf("TCharm[%d] unpacking: ", thisIndex);
  //  CthPrintThdStack(tid);
  //}

}

// Pup our thread and related data
void TCharm::pupThread(PUP::er &pc) {
    pup_er p=(pup_er)&pc;
    checkPupMismatch(pc,5138,"before TCHARM thread"); 
#if CMK_USE_MEMPOOL_ISOMALLOC
    CmiIsomallocBlockListPup(p,&heapBlocks,tid);
#else
    if (CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC))
      CmiIsomallocBlockListPup(p,&heapBlocks,tid);
#endif
    tid = CthPup(p, tid);
    if (pc.isUnpacking()) {
      CtvAccessOther(tid,_curTCharm)=this;
#if CMK_BIGSIM_CHARM
      BgAttach(tid);
#endif
    }
    threadGlobals=CtgPup(p,threadGlobals);
    checkPupMismatch(pc,5139,"after TCHARM thread");
}

//Pup one group of user data
void TCharm::UserData::pup(PUP::er &p)
{
  pup_er pext=(pup_er)(&p);
  p(mode);
  switch(mode) {
  case 'c': { /* C mode: userdata is on the stack, so keep address */
//     p((char*)&data,sizeof(data));
     p(pos);
     //FIXME: function pointers may not be valid across processors
     p((char*)&cfn, sizeof(TCHARM_Pup_fn));
     char *data = CthPointer(t, pos);
     if (cfn) cfn(pext,data);
     } break;
  case 'g': { /* Global mode: zero out userdata on arrival */
     if (CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC))
     {
        // keep the pointer value if using isomalloc, no need to use pup
       p(pos);
     }
     else if (p.isUnpacking())      //  zero out userdata on arrival
       pos=0;

       //FIXME: function pointers may not be valid across processors
     p((char*)&gfn, sizeof(TCHARM_Pup_global_fn));
     if (gfn) gfn(pext);
     } break;
  default:
     break;
  };
}

TCharm::~TCharm()
{
  //BIGSIM_OOC DEBUGGING
  //CmiPrintf("TCharm destructor called with heapBlocks=%p!\n", heapBlocks);
  
#if CMK_USE_MEMPOOL_ISOMALLOC
  mempool_type *mptr = NULL;
  if(tid != NULL) mptr = CtvAccessOther(tid,threadpool);
#else
  if (heapBlocks) CmiIsomallocBlockListDelete(heapBlocks);
#endif
  CthFree(tid);
  CtgFree(threadGlobals);
#if CMK_USE_MEMPOOL_ISOMALLOC 
  if(mptr != NULL) mempool_destroy(mptr);
#endif
  delete initMsg;
}

void TCharm::migrateTo(int destPE) {
	if (destPE==CkMyPe()) return;
	if (CthMigratable() == 0) {
	    CkPrintf("Warning> thread migration is not supported!\n");
            return;
        }
	asyncMigrate = true;
	// Make sure migrateMe gets called *after* we suspend:
	thisProxy[thisIndex].migrateDelayed(destPE);
	suspend();
}
void TCharm::migrateDelayed(int destPE) {
	migrateMe(destPE);
}
void TCharm::ckJustMigrated(void) {
	ArrayElement::ckJustMigrated();
    if (asyncMigrate) {
        asyncMigrate = false;
        resume();
    }
}

void TCharm::ckJustRestored(void) {
	ArrayElement::ckJustRestored();
}

/*
	FAULT_EVAC

	If a Tcharm object is about to migrate it should be suspended first
*/
void TCharm::ckAboutToMigrate(void){
	ArrayElement::ckAboutToMigrate();
	isStopped = true;
}

// clear the data before restarting from disk
void TCharm::clear()
{
#if CMK_USE_MEMPOOL_ISOMALLOC
  mempool_type *mptr = NULL;
  if(tid != NULL) mptr = CtvAccessOther(tid,threadpool);
#else 
  if (heapBlocks) CmiIsomallocBlockListDelete(heapBlocks);
#endif
  CthFree(tid);
#if CMK_USE_MEMPOOL_ISOMALLOC
  if(mptr != NULL) mempool_destroy(mptr);
#endif
  delete initMsg;
}

//Register user data to be packed with the thread
int TCharm::add(const TCharm::UserData &d)
{
  if (nUd>=maxUserData)
    CkAbort("TCharm: Registered too many user data fields!\n");
  int nu=nUd++;
  ud[nu]=d;
  return nu;
}
void *TCharm::lookupUserData(int i) {
	if (i<0 || i>=nUd)
		CkAbort("Bad user data index passed to TCharmGetUserdata!\n");
	return ud[i].getData();
}

//Start the thread running
void TCharm::run(void)
{
  DBG("TCharm::run()");
  if (tcharm_nothreads) {/*Call user routine directly*/
	  startTCharmThread(initMsg);
  } 
  else /* start the thread as usual */
  	  start();
}

//Block the thread until start()ed again.
void TCharm::stop(void)
{
#if CMK_ERROR_CHECKING
  if (tid != CthSelf())
    CkAbort("Called TCharm::stop from outside TCharm thread!\n");
  if (tcharm_nothreads)
    CkAbort("Cannot make blocking calls using +tcharm_nothreads!\n");
#endif
  stopTiming();
  isStopped=true;
  DBG("thread suspended");

  CthSuspend();
// 	DBG("thread resumed");
  /*SUBTLE: We have to do the get() because "this" may have changed
    during a migration-suspend.  If you access *any* members
    from this point onward, you'll cause heap corruption if
    we're resuming from migration!  (OSL 2003/9/23)
   */
  TCharm *dis=TCharm::get();
  dis->isStopped=false;
  dis->startTiming();
  //CkPrintf("[%d] Thread resumed  for tid %p\n",dis->thisIndex,dis->tid);
}

//Resume the waiting thread
void TCharm::start(void)
{
  //  since this thread is scheduled, it is not a good idea to migrate 
  isStopped=false;
  DBG("thread resuming soon");
  //CkPrintf("TCharm[%d]::start()\n", thisIndex);
  //CmiPrintStackTrace(0);
  CthAwaken(tid);
}

//Block our thread, schedule, and come back:
void TCharm::schedule(void) {
  DBG("thread schedule");
  start(); // Calls CthAwaken
  stop(); // Calls CthSuspend
}

//Go to sync, block, possibly migrate, and then resume
void TCharm::migrate(void)
{
#if CMK_LBDB_ON
  DBG("going to sync");
  AtSync();
  stop();
#else
  DBG("skipping sync, because there is no load balancer");
#endif
}



void TCharm::evacuate(){
	/*
		FAULT_EVAC
	*/
	//CkClearAllArrayElementsCPP();
	if(CkpvAccess(startedEvac)){
		CcdCallFnAfter((CcdVoidFn)CkEmmigrateElement, (void *)myRec, 1);
		suspend();
		return;
	}
	return;

}

//calls atsync with async mode
void TCharm::async_migrate(void)
{
#if CMK_LBDB_ON
  DBG("going to sync at async mode");
  ReadyMigrate(false);
  asyncMigrate = true;
  AtSync(0);
  schedule();
#else
  DBG("skipping sync, because there is no load balancer");
#endif
}

/*
Note:
 thread can only migrate at the point when this is called
*/
void TCharm::allow_migrate(void)
{
#if CMK_LBDB_ON
  int nextPe = MigrateToPe();
  if (nextPe != -1) {
    migrateTo(nextPe);
  }
#else
  DBG("skipping sync, because there is no load balancer");
#endif
}

//Resume from sync: start the thread again
void TCharm::ResumeFromSync(void)
{
  DBG("thread resuming from sync");
  start();
}


/****** TcharmClient ******/
void TCharmClient1D::ckJustMigrated(void) {
  ArrayElement1D::ckJustMigrated();
  findThread();
  tcharmClientInit();
}

void TCharmClient1D::pup(PUP::er &p) {
  ArrayElement1D::pup(p);
  p|threadProxy;
}

CkArrayID TCHARM_Get_threads(void) {
	TCHARMAPI("TCHARM_Get_threads");
	return TCharm::get()->getProxy();
}

/************* Startup/Shutdown Coordination Support ************/

// Useless values to reduce over:
int _vals[2]={0,1};

//Called when we want to go to a barrier
void TCharm::barrier(void) {
	//Contribute to a synchronizing reduction
	contribute(CkCallback(CkReductionTarget(TCharm, atBarrier), thisProxy[0]));
#if CMK_BIGSIM_CHARM
        void *curLog;		// store current log in timeline
        _TRACE_BG_TLINE_END(&curLog);
	TRACE_BG_AMPI_BREAK(NULL, "TCharm_Barrier_START", NULL, 0, 1);
#endif
	stop();
#if CMK_BIGSIM_CHARM
	 _TRACE_BG_SET_INFO(NULL, "TCHARM_Barrier_END",  &curLog, 1);
#endif
}

//Called when we've reached the barrier
void TCharm::atBarrier(void){
	DBGX("clients all at barrier");
	thisProxy.start(); //Just restart everybody
}

//Called when the thread is done running
void TCharm::done(void) {
	//CmiPrintStackTrace(0);
	DBG("TCharm thread "<<thisIndex<<" done")
	if (exitWhenDone) {
		//Contribute to a synchronizing reduction
		contribute(CkCallback(CkReductionTarget(TCharm, atExit), thisProxy[0]));
	}
	isSelfDone = true;
	stop();
}
//Called when all threads are done running
void TCharm::atExit(void) {
	DBGX("TCharm::atExit1> exiting");
	//thisProxy.unsetFlags();
	CkExit();
	//CkPrintf("After CkExit()!!!!!!!\n");
}

/************* Setup **************/

//Globals used to control setup process
static TCHARM_Fallback_setup_fn g_fallbackSetup=NULL;
void TCHARM_Set_fallback_setup(TCHARM_Fallback_setup_fn f)
{
	g_fallbackSetup=f;
}
void TCHARM_Call_fallback_setup(void) {
	if (g_fallbackSetup) 
		(g_fallbackSetup)();
	else
		CkAbort("TCHARM: Unexpected fallback setup--missing TCHARM_User_setup routine?");
}

/************** User API ***************/
/**********************************
Callable from UserSetup:
*/

// Read the command line to figure out how many threads to create:
CDECL int TCHARM_Get_num_chunks(void)
{
	TCHARMAPI("TCHARM_Get_num_chunks");
	if (CkMyPe()!=0) CkAbort("TCHARM_Get_num_chunks should only be called on PE 0 during setup!");
	int nChunks=CkNumPes();
	char **argv=CkGetArgv();
	CmiGetArgIntDesc(argv,"-vp",&nChunks,"Set the total number of virtual processors");
	CmiGetArgIntDesc(argv,"+vp",&nChunks,NULL);
	lastNumChunks=nChunks;
	return nChunks;
}
FDECL int FTN_NAME(TCHARM_GET_NUM_CHUNKS,tcharm_get_num_chunks)(void)
{
	return TCHARM_Get_num_chunks();
}

// Fill out the default thread options:
TCHARM_Thread_options::TCHARM_Thread_options(int doDefault)
{
	stackSize=0; /* default stacksize */
	exitWhenDone=0; /* don't exit when done by default. */
}
void TCHARM_Thread_options::sanityCheck(void) {
	if (stackSize<=0) stackSize=tcharm_stacksize;
}


TCHARM_Thread_options g_tcharmOptions(1);

/*Set the size of the thread stack*/
CDECL void TCHARM_Set_stack_size(int newStackSize)
{
	TCHARMAPI("TCHARM_Set_stack_size");
	g_tcharmOptions.stackSize=newStackSize;
}
FDECL void FTN_NAME(TCHARM_SET_STACK_SIZE,tcharm_set_stack_size)
	(int *newSize)
{ TCHARM_Set_stack_size(*newSize); }

CDECL void TCHARM_Set_exit(void) { g_tcharmOptions.exitWhenDone=1; }

/*Create a new array of threads, which will be bound to by subsequent libraries*/
CDECL void TCHARM_Create(int nThreads,
			int threadFn)
{
	TCHARMAPI("TCHARM_Create");
	TCHARM_Create_data(nThreads,
			 threadFn,NULL,0);
}
FDECL void FTN_NAME(TCHARM_CREATE,tcharm_create)
	(int *nThreads,int threadFn)
{ TCHARM_Create(*nThreads,threadFn); }

static CProxy_TCharm TCHARM_Build_threads(TCharmInitMsg *msg);

/*As above, but pass along (arbitrary) data to threads*/
CDECL void TCHARM_Create_data(int nThreads,
		  int threadFn,
		  void *threadData,int threadDataLen)
{
	TCHARMAPI("TCHARM_Create_data");
	TCharmInitMsg *msg=new (threadDataLen,0) TCharmInitMsg(
		threadFn,g_tcharmOptions);
	msg->numElements=nThreads;
	memcpy(msg->data,threadData,threadDataLen);
	TCHARM_Build_threads(msg);
	
	// Reset the thread options:
	g_tcharmOptions=TCHARM_Thread_options(1);
}

FDECL void FTN_NAME(TCHARM_CREATE_DATA,tcharm_create_data)
	(int *nThreads,
		  int threadFn,
		  void *threadData,int *threadDataLen)
{ TCHARM_Create_data(*nThreads,threadFn,threadData,*threadDataLen); }

CkGroupID CkCreatePropMap(void);

static CProxy_TCharm TCHARM_Build_threads(TCharmInitMsg *msg)
{
  CkArrayOptions opts(msg->numElements);
  CkAssert(CkpvAccess(mapCreated)==1);

  if(haveConfigurableRRMap()){
    CkPrintf("USING ConfigurableRRMap\n");
    mapID=CProxy_ConfigurableRRMap::ckNew();
  } else if(mapping==NULL){
#if CMK_BIGSIM_CHARM
    mapID=CProxy_BlockMap::ckNew();
#else
#if __FAULT__
	mapID=CProxy_RRMap::ckNew();
#else
    mapID=CkCreatePropMap();
#endif
#endif
  } else if(0 == strcmp(mapping,"BLOCK_MAP")) {
    CkPrintf("USING BLOCK_MAP\n");
    mapID = CProxy_BlockMap::ckNew();
  } else if(0 == strcmp(mapping,"RR_MAP")) {
    CkPrintf("USING RR_MAP\n");
    mapID = CProxy_RRMap::ckNew();
  } else if(0 == strcmp(mapping,"MAPFILE")) {
    CkPrintf("Reading map from file\n");
    mapID = CProxy_ReadFileMap::ckNew();
  } else {  // "PROP_MAP" or anything else
    mapID = CkCreatePropMap();
  }
  opts.setMap(mapID);
  return CProxy_TCharm::ckNew(msg,opts);
}

// Helper used when creating a new array bound to the TCHARM threads:
CkArrayOptions TCHARM_Attach_start(CkArrayID *retTCharmArray,int *retNumElts)
{
	TCharm *tc=TCharm::get();
	if (!tc)
		CkAbort("You must call TCHARM initialization routines from a TCHARM thread!");
	int nElts=tc->getNumElements();
      
        //CmiPrintf("TCHARM Elements = %d\n", nElts);  
      
	if (retNumElts!=NULL) *retNumElts=nElts;
	*retTCharmArray=tc->getProxy();
	CkArrayOptions opts(nElts);
	opts.bindTo(tc->getProxy());
	return opts;
}

void TCHARM_Suspend(void) {
	TCharm *tc=TCharm::get();
	tc->suspend();
}

/***********************************
Callable from worker thread
*/
CDECL int TCHARM_Element(void)
{ 
	TCHARMAPI("TCHARM_Element");
	return TCharm::get()->getElement();
}
CDECL int TCHARM_Num_elements(void)
{ 
	TCHARMAPI("TCHARM_Num_elements");
	return TCharm::get()->getNumElements();
}

FDECL int FTN_NAME(TCHARM_ELEMENT,tcharm_element)(void) 
{ return TCHARM_Element();}
FDECL int FTN_NAME(TCHARM_NUM_ELEMENTS,tcharm_num_elements)(void) 
{ return TCHARM_Num_elements();}

//Make sure this address will migrate with us when we move:
static void checkAddress(void *data)
{
	if (tcharm_nomig||tcharm_nothreads) return; //Stack is not isomalloc'd
	if (CmiThreadIs(CMI_THREAD_IS_ALIAS)||CmiThreadIs(CMI_THREAD_IS_STACKCOPY)) return; // memory alias thread
	if (CmiIsomallocEnabled()) {
          if (!CmiIsomallocInRange(data))
	    CkAbort("The UserData you register must be allocated on the stack!\n");
        }
        else {
	  if(CkMyPe() == 0)
	    CkPrintf("Warning> checkAddress failed because isomalloc not supported.\n");
	}
}

/* Old "register"-based userdata: */
CDECL int TCHARM_Register(void *data,TCHARM_Pup_fn pfn)
{ 
	TCHARMAPI("TCHARM_Register");
	checkAddress(data);
	return TCharm::get()->add(TCharm::UserData(pfn,TCharm::get()->getThread(),data));
}
FDECL int FTN_NAME(TCHARM_REGISTER,tcharm_register)
	(void *data,TCHARM_Pup_fn pfn)
{ 
	TCHARMAPI("TCHARM_Register");
	checkAddress(data);
	return TCharm::get()->add(TCharm::UserData(pfn,TCharm::get()->getThread(),data));
}

CDECL void *TCHARM_Get_userdata(int id)
{
	TCHARMAPI("TCHARM_Get_userdata");
	return TCharm::get()->lookupUserData(id);
}
FDECL void *FTN_NAME(TCHARM_GET_USERDATA,tcharm_get_userdata)(int *id)
{ return TCHARM_Get_userdata(*id); }

/* New hardcoded-ID userdata: */
CDECL void TCHARM_Set_global(int globalID,void *new_value,TCHARM_Pup_global_fn pup_or_NULL)
{
	TCHARMAPI("TCHARM_Set_global");
	TCharm *tc=TCharm::get();
	if (tc->sud.length()<=globalID)
	{ //We don't have room for this ID yet: make room
		int newLen=2*globalID;
		tc->sud.resize(newLen);
	}
	tc->sud[globalID]=TCharm::UserData(pup_or_NULL,tc->getThread(),new_value);
}
CDECL void *TCHARM_Get_global(int globalID)
{
	//Skip TCHARMAPI("TCHARM_Get_global") because there's no dynamic allocation here,
	// and this routine should be as fast as possible.
	CkVec<TCharm::UserData> &v=TCharm::get()->sud;
	if (v.length()<=globalID) return NULL; //Uninitialized global
	return v[globalID].getData();
}

CDECL void TCHARM_Migrate(void)
{
	TCHARMAPI("TCHARM_Migrate");
	if (CthMigratable() == 0) {
	  if(CkMyPe() == 0)
	    CkPrintf("Warning> thread migration is not supported!\n");
          return;
        }
	TCharm::get()->migrate();
}
FORTRAN_AS_C(TCHARM_MIGRATE,TCHARM_Migrate,tcharm_migrate,(void),())

CDECL void TCHARM_Async_Migrate(void)
{
	TCHARMAPI("TCHARM_Async_Migrate");
	TCharm::get()->async_migrate();
}
FORTRAN_AS_C(TCHARM_ASYNC_MIGRATE,TCHARM_Async_Migrate,tcharm_async_migrate,(void),())

CDECL void TCHARM_Allow_Migrate(void)
{
	TCHARMAPI("TCHARM_Allow_Migrate");
	TCharm::get()->allow_migrate();
}
FORTRAN_AS_C(TCHARM_ALLOW_MIGRATE,TCHARM_Allow_Migrate,tcharm_allow_migrate,(void),())

CDECL void TCHARM_Migrate_to(int destPE)
{
	TCHARMAPI("TCHARM_Migrate_to");
	TCharm::get()->migrateTo(destPE);
}

CDECL void TCHARM_Evacuate()
{
	TCHARMAPI("TCHARM_Migrate_to");
	TCharm::get()->evacuate();
}

FORTRAN_AS_C(TCHARM_MIGRATE_TO,TCHARM_Migrate_to,tcharm_migrate_to,
	(int *destPE),(*destPE))

CDECL void TCHARM_Yield(void)
{
	TCHARMAPI("TCHARM_Yield");
	TCharm::get()->schedule();
}
FORTRAN_AS_C(TCHARM_YIELD,TCHARM_Yield,tcharm_yield,(void),())

CDECL void TCHARM_Barrier(void)
{
	TCHARMAPI("TCHARM_Barrier");
	TCharm::get()->barrier();
}
FORTRAN_AS_C(TCHARM_BARRIER,TCHARM_Barrier,tcharm_barrier,(void),())

CDECL void TCHARM_Done(void)
{
	TCHARMAPI("TCHARM_Done");
	TCharm *c=TCharm::getNULL();
	if (!c) CkExit();
	else c->done();
}
FORTRAN_AS_C(TCHARM_DONE,TCHARM_Done,tcharm_done,(void),())


CDECL double TCHARM_Wall_timer(void)
{
  TCHARMAPI("TCHARM_Wall_timer");
  TCharm *c=TCharm::getNULL();
  if(!c) return CkWallTimer();
  else { //Have to apply current thread's time offset
    return CkWallTimer()+c->getTimeOffset();
  }
}

#if 1
/*Include Fortran-style "iargc" and "getarg" routines.
These are needed to get access to the command-line arguments from Fortran.
*/
FDECL int FTN_NAME(TCHARM_IARGC,tcharm_iargc)(void) {
  TCHARMAPI("tcharm_iargc");
  return CkGetArgc()-1;
}

FDECL void FTN_NAME(TCHARM_GETARG,tcharm_getarg)
	(int *i_p,char *dest,int destLen)
{
  TCHARMAPI("tcharm_getarg");
  int i=*i_p;
  if (i<0) CkAbort("tcharm_getarg called with negative argument!");
  if (i>=CkGetArgc()) CkAbort("tcharm_getarg called with argument > iargc!");
  const char *src=CkGetArgv()[i];
  strcpy(dest,src);
  for (i=strlen(dest);i<destLen;i++) dest[i]=' ';
}

#endif

//These silly routines are used for serial startup:
extern void _initCharm(int argc, char **argv);
CDECL void TCHARM_Init(int *argc,char ***argv) {
	if (!tcharm_initted) {
	  ConverseInit(*argc, *argv, (CmiStartFn) _initCharm,1,1);
	  _initCharm(*argc,*argv);
	}
}

FDECL void FTN_NAME(TCHARM_INIT,tcharm_init)(void)
{
	int argc=1;
	const char *argv_sto[2]={"foo",NULL};
	char **argv=(char **)argv_sto;
	TCHARM_Init(&argc,&argv);
}

/***********************************
* TCHARM Semaphores:
* The idea is one side "puts", the other side "gets"; 
* but the calls can come in any order--
* if the "get" comes first, it blocks until the put.
* This makes a convenient, race-condition-free way to do
* onetime initializations.  
*/
/// Find this semaphore, or insert if there isn't one:
TCharm::TCharmSemaphore *TCharm::findSema(int id) {
	for (unsigned int s=0;s<sema.size();s++)
		if (sema[s].id==id) 
			return &sema[s];
	sema.push_back(TCharmSemaphore(id));
	return &sema[sema.size()-1];
}
/// Remove this semaphore from the list
void TCharm::freeSema(TCharmSemaphore *doomed) {
	int id=doomed->id;
	for (unsigned int s=0;s<sema.size();s++)
		if (sema[s].id==id) {
			sema[s]=sema[sema.length()-1];
			sema.length()--;
			return;
		}
	CkAbort("Tried to free nonexistent TCharm semaphore");
}

/// Block until this semaphore has data:
TCharm::TCharmSemaphore *TCharm::getSema(int id) {
	TCharmSemaphore *s=findSema(id);
	if (s->data==NULL) 
	{ //Semaphore isn't filled yet: wait until it is
		s->thread=CthSelf();
		suspend(); //Will be woken by semaPut
		// Semaphore may have moved-- find it again
		s=findSema(id);
		if (s->data==NULL) CkAbort("TCharm::semaGet awoken too early!");
	}
	return s;
}

/// Store data at the semaphore "id".
///  The put can come before or after the get.
void TCharm::semaPut(int id,void *data) {
	TCharmSemaphore *s=findSema(id);
	if (s->data!=NULL) CkAbort("Duplicate calls to TCharm::semaPut!");
	s->data=data;
	DBG("semaPut "<<id<<" "<<data);
	if (s->thread!=NULL) {//Awaken the thread
		s->thread=NULL;
		resume();
	}
}

/// Retreive data from the semaphore "id".
///  Blocks if the data is not immediately available.
///  Consumes the data, so another put will be required for the next get.
void *TCharm::semaGet(int id) {
	TCharmSemaphore *s=getSema(id);
	void *ret=s->data;
	DBG("semaGet "<<id<<" "<<ret);
	// Now remove the semaphore from the list:
	freeSema(s);
	return ret;
}

/// Retreive data from the semaphore "id".
///  Blocks if the data is not immediately available.
void *TCharm::semaGets(int id) {
	TCharmSemaphore *s=getSema(id);
	return s->data;
}

/// Retreive data from the semaphore "id", or returns NULL.
void *TCharm::semaPeek(int id) {
	TCharmSemaphore *s=findSema(id);
	return s->data;
}

/****** System Call support ******/
/*
TCHARM_System exists to work around a bug where Linux ia32
glibc2.2.x with pthreads crashes at the fork() site when 
called from a user-levelthread. 

The fix is to call system from the main thread, by 
passing the request out of the thread to our array element 
before calling system().
*/

CDECL int 
TCHARM_System(const char *shell_command)
{
	return TCharm::get()->system(shell_command);
}
int TCharm::system(const char *cmd)
{
	int ret=-1778;
	callSystemStruct s;
	s.cmd=cmd;
	s.ret=&ret;
	thisProxy[thisIndex].callSystem(s);
	suspend();
	return ret;
}

void TCharm::callSystem(const callSystemStruct &s)
{
	*s.ret = ::system(s.cmd);
	resume();
}



#include "tcharm.def.h"
