/*
Threaded Charm++ "Framework Framework"

Orion Sky Lawlor, olawlor@acm.org, 11/19/2001
 */
#include "tcharm_impl.h"
#include "tcharm.h"
#include <ctype.h>

#if 0
    /*Many debugging statements:*/
#    define DBG(x) ckout<<"["<<thisIndex<<"] TCHARM> "<<x<<endl;
#    define DBGX(x) ckout<<"PE("<<CkMyPe()<<") TCHARM> "<<x<<endl;
#else
    /*No debugging statements*/
#    define DBG(x) /*empty*/
#    define DBGX(x) /*empty*/
#endif

CtvDeclare(TCharm *,_curTCharm);
CpvDeclare(inState,_stateTCharm);

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

void TCharm::nodeInit(void)
{
  CtvInitialize(TCharm *,_curTCharm);
  CtvAccess(_curTCharm)=NULL;
  CpvInitialize(inState,_stateTCharm);
  char **argv=CkGetArgv();
  tcharm_nomig=CmiGetArgFlag(argv,"+tcharm_nomig");
  tcharm_nothreads=CmiGetArgFlag(argv,"+tcharm_nothread");
  tcharm_nothreads|=CmiGetArgFlag(argv,"+tcharm_nothreads");
  char *traceLibName=NULL;
  while (CmiGetArgString(argv,"+tcharm_trace",&traceLibName))
      tcharm_tracelibs.addTracing(traceLibName);
  CmiGetArgInt(argv,"+tcharm_stacksize",&tcharm_stacksize);
  if (CkMyPe()!=0) { //Processor 0 eats "+vp<N>" and "-vp<N>" later:
  	int ignored;
  	while (CmiGetArgInt(argv,"-vp",&ignored)) {}
  	while (CmiGetArgInt(argv,"+vp",&ignored)) {}
  }
  
  TCharm::setState(inNodeSetup);
  TCHARM_User_node_setup();
  FTN_NAME(TCHARM_USER_NODE_SETUP,tcharm_user_node_setup)();
  TCharm::setState(inInit);
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

static void startTCharmThread(TCharmInitMsg *msg)
{
	TCharm::setState(inDriver);
	CtvAccess(_curTCharm)->activateHeap();
	typedef void (*threadFn_t)(void *);
	((threadFn_t)msg->threadFn)(msg->data);
	CmiIsomallocBlockListActivate(NULL); //Turn off migratable memory
	CtvAccess(_curTCharm)->done();
}

TCharm::TCharm(TCharmInitMsg *initMsg_)
{
  initMsg=initMsg_;
  timeOffset=0.0;
  if (tcharm_nothreads) 
  { //Don't even make a new thread-- just use main thread
    tid=CthSelf();
  } 
  else /*Create a thread normally*/
  {
    if (tcharm_nomig) /*Nonmigratable version, for debugging*/ 
      tid=CthCreate((CthVoidFn)startTCharmThread,initMsg,initMsg->stackSize);
    else
      tid=CthCreateMigratable((CthVoidFn)startTCharmThread,initMsg,initMsg->stackSize);
  }
  CtvAccessOther(tid,_curTCharm)=this;
  TCharm::setState(inInit);
  isStopped=true;
  threadInfo.tProxy=CProxy_TCharm(thisArrayID);
  threadInfo.thisElement=thisIndex;
  threadInfo.numElements=initMsg->numElements;
  heapBlocks=CmiIsomallocBlockListNew();
  nUd=0;
  usesAtSync=CmiTrue;
  ready();
}

TCharm::TCharm(CkMigrateMessage *msg)
	:ArrayElement1D(msg)
{
  initMsg=NULL;
  tid=NULL;
  threadInfo.tProxy=CProxy_TCharm(thisArrayID);  
}

void TCharm::pup(PUP::er &p) {
//Pup superclass
  ArrayElement1D::pup(p);  

  p(isStopped);
  p(threadInfo.thisElement);
  p(threadInfo.numElements);

#ifndef CMK_OPTIMIZE
  DBG("Packing thread");
  if (!isStopped)
    CkAbort("Cannot pup a running thread.  You must suspend before migrating.\n");
  if (tcharm_nomig) CkAbort("Cannot migrate with the +tcharm_nomig option!\n");
#endif

//Pup thread (EVIL & UGLY):
  //This seekBlock allows us to reorder the packing/unpacking--
  // This is needed because the userData depends on the thread's stack
  // and heap data both at pack and unpack time.
  PUP::seekBlock s(p,2);
  if (p.isUnpacking()) 
  {//In this case, unpack the thread & heap before the user data
    s.seek(1);
    tid = CthPup((pup_er) &p, tid);
    CtvAccessOther(tid,_curTCharm)=this;
    CmiIsomallocBlockListPup((pup_er) &p,&heapBlocks);
    //Restart our clock: set it up so packTime==CkWallTimer+timeOffset
    double packTime;
    p(packTime);
    timeOffset=packTime-CkWallTimer();
  }
  
  //Pack all user data
  TCharm::setState(inPup);
  s.seek(0);
  p(nUd);
  for(int i=0;i<nUd;i++) 
    ud[i].pup(p);
  sud.pup(p);
  TCharm::setState(inFramework);

  if (!p.isUnpacking()) 
  {//In this case, pack the thread & heap after the user data
    s.seek(1);
    tid = CthPup((pup_er) &p, tid);
    CmiIsomallocBlockListPup((pup_er) &p,&heapBlocks);
    //Stop our clock:
    double packTime=CkWallTimer()+timeOffset;
    p(packTime);
  }
  s.endBlock(); //End of seeking block
}

//Pup one group of user data
void TCharm::UserData::pup(PUP::er &p)
{
  pup_er pext=(pup_er)(&p);
  p(isC);
  //Save address of userdata-- assumes user data is on the stack
  p((void*)&data,sizeof(data));
  if (isC) { //C version
    //FIXME: function pointers may not be valid across processors
    p((void*)&cfn, sizeof(TCpupUserDataC));
    if (cfn) cfn(pext,data);
  } 
  else { //Fortran version
    //FIXME: function pointers may not be valid across processors
    p((void*)&ffn, sizeof(TCpupUserDataF));        
    if (ffn) ffn(pext,data);
  }
}


TCharm::~TCharm() 
{
  CmiIsomallocBlockListDelete(heapBlocks);
  CthFree(tid);
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
  start();
}

//Block the thread until start()ed again.
void TCharm::stop(void)
{
  if (isStopped) return; //Nothing to do
#ifndef CMK_OPTIMIZE
  DBG("suspending thread");
  if (tid != CthSelf())
    CkAbort("Called TCharm::stop from outside TCharm thread!\n");
  if (tcharm_nothreads)
    CkAbort("Cannot make blocking calls using +tcharm_nothreads!\n");
#endif
  isStopped=true;
  stopTiming();
  TCharm::setState(inFramework);
  CthSuspend();
  TCharm::setState(inDriver);
  /*We have to do the get() because "this" may have changed
    during a migration-suspend.*/
  TCharm::get()->startTiming();
}

//Resume the waiting thread
void TCharm::start(void)
{
  if (!isStopped) return; //Already started
  isStopped=false;
  TCharm::setState(inDriver);
  DBG("awakening thread");
  if (tcharm_nothreads) /*Call user routine directly*/
	  startTCharmThread(initMsg);
  else /*Jump to thread normally*/
	  CthAwaken(tid);
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

//Resume from sync: start the thread again
void TCharm::ResumeFromSync(void)
{
  start();
}

#ifndef CMK_OPTIMIZE
//Make sure we're actually in driver
void TCharm::check(void)
{
	if (getState()!=inDriver)
		::CkAbort("TCharm> Can only use that routine from within driver!\n");
}
#endif

static int propMapCreated=0;
static CkGroupID propMapID;
CkGroupID CkCreatePropMap(void);

static void TCHARM_Build_threads(TCharmInitMsg *msg,TCharmSetupCookie &cook)
{
	CkArrayOptions opts(msg->numElements);
	if (!propMapCreated) {
		propMapCreated=1;
		propMapID=CkCreatePropMap();
	}
	opts.setMap(propMapID);
	int nElem=msg->numElements; //<- save it because msg will be deleted.
	CkArrayID id=CProxy_TCharm::ckNew(msg,opts);
	cook.setThreads(id,nElem);
}

/****** TcharmClient ******/
void TCharmClient1D::ckJustMigrated(void) {
  ArrayElement1D::ckJustMigrated();
  tcharmClientInit();
}

void TCharmClient1D::pup(PUP::er &p) {
  ArrayElement1D::pup(p);
  p|threadProxy;
}


/****** Readonlys *****/
CkVec<TCpupReadonlyGlobal> TCharmReadonlys::entries;
void TCharmReadonlys::add(TCpupReadonlyGlobal fn)
{
	entries.push_back(fn);
}

//Pups all registered readonlys
void TCharmReadonlys::pupAllReadonlys(PUP::er &p) {
	//Pup the globals for this node:
	int i,n=entries.length();
	p.comment("TCharm Readonly global variables:");
	p(n);
	if (n!=entries.length())
		CkAbort("TCharmReadonly list length mismatch!\n");
	for (i=0;i<n;i++)
		(entries[i])((pup_er)&p);
}

void TCharmReadonlys::pup(PUP::er &p) {
	if (p.isUnpacking()) {
		//HACK: Rather than sending this message only where its needed,
		// we send it everywhere and just ignore it if it's not needed.
		if (CkMyPe()==0) return; //Processor 0 is the source-- no unpacking needed
		if (CkMyRank()!=0) return; //Some other processor will do the unpacking
	}
	pupAllReadonlys(p);
}

CDECL void TCHARM_Readonly_globals(TCpupReadonlyGlobal fn)
{
	TCHARMAPI("TCHARM_Readonly_globals");
	if (TCharm::getState()!=inNodeSetup)
		CkAbort("Can only call TCHARM_ReadonlyGlobals from in TCHARM_UserNodeSetup!\n");
	TCharmReadonlys::add(fn);
}
FDECL void FTN_NAME(TCHARM_READONLY_GLOBALS,tcharm_readonly_globals)
	(TCpupReadonlyGlobal fn)
{
	TCHARM_Readonly_globals(fn);
}

/************* Startup/Shutdown Coordination Support ************/

enum {TC_READY=23, TC_BARRIER=87, TC_DONE=42};

//Called when a client is ready to run
void TCharm::ready(void) {
	DBG("TCharm thread "<<thisIndex<<" ready")
	int vals[2]={0,1};
	if (thisIndex==0) vals[0]=TC_READY;
	//Contribute to a synchronizing reduction
	contribute(sizeof(vals),&vals,CkReduction::sum_int);
}

//Called when we want to go to a barrier
void TCharm::barrier(void) {
	int vals[2]={0,1};
	if (thisIndex==0) vals[0]=TC_BARRIER;
	//Contribute to a synchronizing reduction
	contribute(sizeof(vals),&vals,CkReduction::sum_int);
	stop();
}

//Called when the thread is done running
void TCharm::done(void) {
	DBG("TCharm thread "<<thisIndex<<" done")
	int vals[2]={0,1};
	if (thisIndex==0) vals[0]=TC_DONE;
	//Contribute to a synchronizing reduction
	contribute(sizeof(vals),&vals,CkReduction::sum_int);
	stop();
}

//Called when an array reduction is complete
static void coordinatorReduction(void *coord_,int dataLen,void *reductionData)
{
	TCharmCoordinator *coord=(TCharmCoordinator *)coord_;
	int *vals=(int *)reductionData;
	if (dataLen!=2*sizeof(int))
		CkAbort("Unexpected length in TCharm array reduction!\n");
	DBGX("Finished coordinator reduction: "<<vals[0]<<", "<<vals[1]);
	switch (vals[0]) {
	case TC_READY: coord->clientReady(); break;
	case TC_BARRIER: coord->clientBarrier(); break;
	case TC_DONE: coord->clientDone(); break;
	default:
		CkAbort("Unexpected value from TCharm array reduction!\n");
	};
}

int TCharmCoordinator::nArrays=0; //Total number of running thread arrays
TCharmCoordinator *TCharmCoordinator::head=NULL; //List of coordinators


TCharmCoordinator::TCharmCoordinator(CkArrayID threads_,int nThreads_)
	:threads(threads_), nThreads(nThreads_), nClients(0), nReady(0)
{
	nArrays++;
	//Link into the coordinator list
	next=head;
	head=this;

	threads.setReductionClient(coordinatorReduction,this);
	nClients=1; //Thread array itself is a client
}
TCharmCoordinator::~TCharmCoordinator()
{
	//Coordinators never get deleted
}
void TCharmCoordinator::addClient(const CkArrayID &client)
{
	nClients++;
}
void TCharmCoordinator::clientReady(void)
{
	DBGX("client "<<nReady+1<<" of "<<nClients<<" ready");
	nReady++;
	if (nReady>=nClients) { //All client arrays are ready-- start threads
		DBGX("starting threads");
		threads.run();
	}
}
void TCharmCoordinator::clientBarrier(void)
{
	DBGX("clients all at barrier");
	threads.run();
}
void TCharmCoordinator::clientDone(void)
{
	DBGX("clientDone");	
	nArrays--;
	if (nArrays<=0) { //All arrays have exited
		DBGX("done with computation");
		CkExit();
	}
}

/************* Setup **************/

//Cookie used during setup
TCharmSetupCookie *TCharmSetupCookie::theCookie;

//Globals used to control setup process
static int g_numDefaultSetups=0;
static TCHARM_Fallback_setup_fn g_fallbackSetup=NULL;
void TCHARM_Set_fallback_setup(TCHARM_Fallback_setup_fn f)
{
	g_fallbackSetup=f;
}
CDECL void TCharmInDefaultSetup(void) {
	g_numDefaultSetups++;
}

//Tiny simple main chare
class TCharmMain : public Chare {
public:
  TCharmMain(CkArgMsg *msg) {
    if (0!=tcharm_nomig)
        CmiPrintf("TCHARM> Disabling migration support, for debugging\n");
    if (0!=tcharm_nothreads)
       CmiPrintf("TCHARM> Disabling thread support, for debugging\n");

    TCharmSetupCookie cookie(msg->argv);
    TCharmSetupCookie::theCookie=&cookie;
    g_numDefaultSetups=0;
    
    /*Call user-overridable C setup*/
    TCHARM_User_setup();
    /*Call user-overridable Fortran setup*/
    FTN_NAME(TCHARM_USER_SETUP,tcharm_user_setup)();
    
    if (g_numDefaultSetups==2) 
    { //User didn't override either setup routine
	    if (g_fallbackSetup)
		    (g_fallbackSetup)();
	    else
		    CmiAbort("You need to override TCharmUserSetup to start your computation, or else link in a framework module\n");
    }	    
    
    delete msg;
    
    if (0==TCharmCoordinator::getTotal())
	    CkAbort("You didn't create any TCharm arrays in TCharmUserSetup!\n");

    //Send out the readonly globals:
    TCharmReadonlys r;
    CProxy_TCharmReadonlyGroup::ckNew(r);
  }
};

#ifndef CMK_OPTIMIZE
/*The setup cookie, used to store global initialization state*/
TCharmSetupCookie &TCharmSetupCookie::check(void)
{
	if (magic!=correctMagic)
		CkAbort("TCharm setup cookie is damaged!\n");
	return *this;
}
#endif

void TCharmSetupCookie::setThreads(const CkArrayID &aid,int nel)
{
	coord=new TCharmCoordinator(aid,nel);
	tc=aid; numElements=nel;
}

TCharmSetupCookie::TCharmSetupCookie(char **argv_)
{
	magic=correctMagic;
	argv=argv_;
	coord=NULL;
	stackSize=tcharm_stacksize;
}

CkArrayOptions TCHARM_Attach_start(CkArrayID *retTCharmArray,int *retNumElts)
{
	TCharmSetupCookie *tc=TCharmSetupCookie::get();
	if (!tc->hasThreads())
		CkAbort("You must create a thread array with TCharmCreate before calling Attach!\n");
	int nElts=tc->getNumElements();
	if (retNumElts!=NULL) *retNumElts=nElts;
	*retTCharmArray=tc->getThreads();
	CkArrayOptions opts(nElts);
	opts.bindTo(tc->getThreads());
	return opts;
}
void TCHARM_Attach_finish(const CkArrayID &libraryArray)
{
	TCharmSetupCookie *tc=TCharmSetupCookie::get();
	tc->addClient(libraryArray);
}


/************** User API ***************/

#define cookie (*TCharmSetupCookie::get())

/**********************************
Callable from UserSetup: 
*/

/*Set the size of the thread stack*/
CDECL void TCHARM_Set_stack_size(int newStackSize)
{
	TCHARMAPI("TCHARM_Set_stack_size");
	if (TCharm::getState()!=inInit)
		CkAbort("TCharm> Can only set stack size from in init!\n");
	cookie.setStackSize(newStackSize);
}
FDECL void FTN_NAME(TCHARM_SET_STACK_SIZE,tcharm_set_stack_size)
	(int *newSize)
{ TCHARM_Set_stack_size(*newSize); }


/*Create a new array of threads, which will be bound to by subsequent libraries*/
CDECL void TCHARM_Create(int nThreads,
			TCHARM_Thread_start_fn threadFn)
{
	TCHARMAPI("TCHARM_Create");
	TCHARM_Create_data(nThreads,
			 (TCHARM_Thread_data_start_fn)threadFn,NULL,0);
}
FDECL void FTN_NAME(TCHARM_CREATE,tcharm_create)
	(int *nThreads,TCHARM_Thread_start_fn threadFn)
{ TCHARM_Create(*nThreads,threadFn); }


/*As above, but pass along (arbitrary) data to threads*/
CDECL void TCHARM_Create_data(int nThreads,
		  TCHARM_Thread_data_start_fn threadFn,
		  void *threadData,int threadDataLen)
{
	TCHARMAPI("TCHARM_Create_data");
	if (TCharm::getState()!=inInit)
		CkAbort("TCharm> Can only create threads from in init!\n");
	TCharmSetupCookie &cook=cookie;
	TCharmInitMsg *msg=new (threadDataLen,0) TCharmInitMsg(
		(CthVoidFn)threadFn,cook.getStackSize());
	msg->numElements=nThreads;
	memcpy(msg->data,threadData,threadDataLen);
	TCHARM_Build_threads(msg,cook);
}

FDECL void FTN_NAME(TCHARM_CREATE_DATA,tcharm_create_data)
	(int *nThreads,
		  TCHARM_Thread_data_start_fn threadFn,
		  void *threadData,int *threadDataLen)
{ TCHARM_Create_data(*nThreads,threadFn,threadData,*threadDataLen); }


CDECL int TCHARM_Get_num_chunks(void)
{
	TCHARMAPI("TCHARM_Get_num_chunks");
	if (CkMyPe()!=0) CkAbort("TCHARM_Get_num_chunks should only be called on PE 0 during setup!");
	int nChunks=CkNumPes();
	char **argv=CkGetArgv();
	CmiGetArgInt(argv,"-vp",&nChunks);
	CmiGetArgInt(argv,"+vp",&nChunks);
	lastNumChunks=nChunks;
	return nChunks;
}
FDECL int FTN_NAME(TCHARM_GET_NUM_CHUNKS,tcharm_get_num_chunks)(void)
{
	return TCHARM_Get_num_chunks();
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
	if (TCharm::getState()==inDriver)
		return TCharm::get()->getNumElements();
	else
		return lastNumChunks;
}

FDECL int FTN_NAME(TCHARM_ELEMENT,tcharm_element)(void) 
{ return TCHARM_Element();}
FDECL int FTN_NAME(TCHARM_NUM_ELEMENTS,tcharm_num_elements)(void) 
{ return TCHARM_Num_elements();}

//Make sure this address will migrate with us when we move:
static void checkAddress(void *data)
{
	if (tcharm_nomig||tcharm_nothreads) return; //Stack is not isomalloc'd
	if (!CmiIsomallocInRange(data))
	    CkAbort("The UserData you register must be allocated on the stack!\n");
}

/* Old "register"-based userdata: */
CDECL int TCHARM_Register(void *data,TCHARM_Pup_fn pfn)
{ 
	TCHARMAPI("TCHARM_Register");
	checkAddress(data);
	return TCharm::get()->add(TCharm::UserData(pfn,data));
}
FDECL int FTN_NAME(TCHARM_REGISTER,tcharm_register)
	(void *data,TCpupUserDataF pfn)
{ 
	TCHARMAPI("TCHARM_Register");
	checkAddress(data);
	return TCharm::get()->add(TCharm::UserData(
		pfn,data,TCharm::UserData::isFortran()));
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
		tc->sud.setSize(newLen);
		tc->sud.length()=newLen;
	}
	tc->sud[globalID]=TCharm::UserData((TCHARM_Pup_fn) pup_or_NULL,new_value);
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
	TCharm::get()->migrate();
}
FDECL void FTN_NAME(TCHARM_MIGRATE,tcharm_migrate)(void)
{
	TCHARMAPI("TCHARM_Migrate");
	TCharm::get()->migrate();
}

CDECL void TCHARM_Yield(void)
{
	TCHARMAPI("TCHARM_Yield");
	TCharm::get()->schedule();
}
FDECL void FTN_NAME(TCHARM_YIELD,tcharm_yield)(void)
{
	TCHARM_Yield();
}

CDECL void TCHARM_Barrier(void)
{
	TCHARMAPI("TCHARM_Barrier");
	TCharm::get()->barrier();
}
FDECL void FTN_NAME(TCHARM_BARRIER,tcharm_barrier)(void)
{
	TCHARM_Barrier();
}

CDECL void TCHARM_Done(void)
{
	TCHARMAPI("TCHARM_Done");
	if (TCharm::getState()!=inDriver) CkExit();
	else TCharm::get()->done();
}
FDECL void FTN_NAME(TCHARM_DONE,tcharm_done)(void)
{
	TCHARM_Done();
}

CDECL double TCHARM_Wall_timer(void) 
{
  TCHARMAPI("TCHARM_Wall_timer");
  if(TCharm::getState()!=inDriver) return CkWallTimer();
  else { //Have to apply current thread's time offset
    return CkWallTimer()+TCharm::get()->getTimeOffset();
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
	ConverseInit(*argc, *argv, (CmiStartFn) _initCharm,1,1);
	_initCharm(*argc,*argv);
}

FDECL void FTN_NAME(TCHARM_INIT,tcharm_init)(void) 
{
	int argc=1;
	char *argv[2]={"foo",NULL};
	ConverseInit(argc,argv, (CmiStartFn) _initCharm,1,1);
	_initCharm(argc,argv);
}

#include "tcharm.def.h"
