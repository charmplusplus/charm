/*
Threaded Charm++ "Framework Framework"

Orion Sky Lawlor, olawlor@acm.org, 11/19/2001
 */
#include "tcharm.h"

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

void TCharm::nodeInit(void)
{
  CtvInitialize(TCharm *,_curTCharm);
  CpvInitialize(inState,_stateTCharm);
  TCharm::setState(inNodeSetup);
  TCharmUserNodeSetup();
  FTN_NAME(TCHARM_USER_NODE_SETUP,tcharm_user_node_setup)();
  TCharm::setState(inInit);
}

static void startTCharmThread(TCharmInitMsg *msg)
{
	TCharm::setState(inDriver);
	typedef void (*threadFn_t)(void *);
	((threadFn_t)msg->threadFn)(msg->data);
	CtvAccess(_curTCharm)->done();
}

TCharm::TCharm(TCharmInitMsg *initMsg_)
{
  initMsg=initMsg_;
  tid=CthCreateMigratable((CthVoidFn)startTCharmThread,initMsg,initMsg->stackSize);
  CtvAccessOther(tid,_curTCharm)=this;
  TCharm::setState(inInit);
  isStopped=true;
  threadInfo.tProxy=CProxy_TCharm(thisArrayID);
  threadInfo.thisElement=thisIndex;
  threadInfo.numElements=initMsg->numElements;
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
#endif

//Pup thread (EVIL & UGLY):
  //This seekBlock allows us to reorder the packing/unpacking--
  // This is needed because the userData depends on the thread's stack
  // both at pack and unpack time.
  PUP::seekBlock s(p,2);
  if (p.isUnpacking()) 
  {//In this case, unpack the thread before the user data
    s.seek(1);
    tid = CthPup((pup_er) &p, tid);
    CtvAccessOther(tid,_curTCharm)=this;
  }
  
  //Pack all user data
  TCharm::setState(inPup);
  s.seek(0);
  p(nUd);
  for(int i=0;i<nUd;i++) 
    ud[i].pup(p);
  TCharm::setState(inFramework);

  if (!p.isUnpacking()) 
  {//In this case, pack the thread after the user data
    s.seek(1);
    tid = CthPup((pup_er) &p, tid);
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
    cfn(pext,data);
  } 
  else { //Fortran version
    //FIXME: function pointers may not be valid across processors
    p((void*)&ffn, sizeof(TCpupUserDataF));        
    ffn(pext,data);
  }
}

TCharm::~TCharm() 
{
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
		CkAbort("TCharm> Can only use that routine from within driver!\n");
}
#endif

static int propMapCreated=0;
static CkGroupID propMapID;
CkGroupID CkCreatePropMap(void);

static void TCharmBuildThreads(TCharmInitMsg *msg,TCharmSetupCookie &cook)
{
	CkArrayOptions opts(msg->numElements);
	if (!propMapCreated) {
		propMapCreated=1;
		propMapID=CkCreatePropMap();
	}
	opts.setMap(propMapID);
	CkArrayID id=CProxy_TCharm::ckNew(msg,opts);
	cook.setThreads(id,msg->numElements);
}

/****** Readonlys *****/
CkVec<TCpupReadonlyGlobal> TCharmReadonlys::entries;
void TCharmReadonlys::add(TCpupReadonlyGlobal fn)
{
	entries.push_back(fn);
}
//Pups all registered readonlys
void TCharmReadonlys::pup(PUP::er &p) {
	if (p.isUnpacking()) {
		//HACK: Rather than sending this message only where its needed,
		// we send it everywhere and just ignore it if it's not needed.
		if (CkMyPe()==0) return; //Processor 0 is the source-- no unpacking needed
		if (CkMyRank()!=0) return; //Some other processor will do the unpacking
	}
	//Pup the globals for this node:
	int i,n=entries.length();
	p(n);
	if (n!=entries.length())
		CkAbort("TCharmReadonly list length mismatch!\n");
	for (i=0;i<n;i++)
		(entries[i])((pup_er)&p);
}

CDECL void TCharmReadonlyGlobals(TCpupReadonlyGlobal fn)
{
	if (TCharm::getState()!=inNodeSetup)
		CkAbort("Can only call TCharmReadonlyGlobals from in TCharmUserNodeSetup!\n");
	TCharmReadonlys::add(fn);
}
FDECL void FTN_NAME(TCHARM_READONLY_GLOBALS,tcharm_readonly_globals)
	(TCpupReadonlyGlobal fn)
{
	TCharmReadonlyGlobals(fn);
}

/************* Startup/Shutdown Coordination Support ************/

enum {TC_READY=23, TC_DONE=42};

//Called when a client is ready to run
void TCharm::ready(void) {
	DBG("TCharm thread "<<thisIndex<<" ready")
	int vals[2]={0,1};
	if (thisIndex==0) vals[0]=TC_READY;
	//Contribute to a synchronizing reduction
	contribute(sizeof(vals),&vals,CkReduction::sum_int);
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
static TCharmFallbackSetupFn g_fallbackSetup=NULL;
void TCharmSetFallbackSetup(TCharmFallbackSetupFn f)
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
    TCharmSetupCookie cookie(msg->argv);
    TCharmSetupCookie::theCookie=&cookie;
    g_numDefaultSetups=0;
    
    /*Call user-overridable C setup*/
    TCharmUserSetup();
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
	stackSize=0;
	argv=argv_;
	coord=NULL;
}


/************** User API ***************/

#define cookie (*TCharmSetupCookie::get())

/**********************************
Callable from UserSetup: 
*/

/*Set the size of the thread stack*/
CDECL void TCharmSetStackSize(int newStackSize)
{
	if (TCharm::getState()!=inInit)
		CkAbort("TCharm> Can only set stack size from in init!\n");
	cookie.setStackSize(newStackSize);
}
FDECL void FTN_NAME(TCHARM_SET_STACK_SIZE,tcharm_set_stack_size)
	(int *newSize)
{ TCharmSetStackSize(*newSize); }


/*Create a new array of threads, which will be bound to by subsequent libraries*/
CDECL void TCharmCreate(int nThreads,
			TCharmThreadStartFn threadFn)
{
	TCharmCreateData(nThreads,
			 (TCharmThreadDataStartFn)threadFn,NULL,0);
}
FDECL void FTN_NAME(TCHARM_CREATE,tcharm_create)
	(int *nThreads,TCharmThreadStartFn threadFn)
{ TCharmCreate(*nThreads,threadFn); }


/*As above, but pass along (arbitrary) data to threads*/
CDECL void TCharmCreateData(int nThreads,
		  TCharmThreadDataStartFn threadFn,
		  void *threadData,int threadDataLen)
{
	if (TCharm::getState()!=inInit)
		CkAbort("TCharm> Can only create threads from in init!\n");
	TCharmSetupCookie &cook=cookie;
	TCharmInitMsg *msg=new (threadDataLen,0) TCharmInitMsg(
		(CthVoidFn)threadFn,cook.getStackSize());
	msg->numElements=nThreads;
	memcpy(msg->data,threadData,threadDataLen);
	TCharmBuildThreads(msg,cook);
}

FDECL void FTN_NAME(TCHARM_CREATE_DATA,tcharm_create_data)
	(int *nThreads,
		  TCharmThreadDataStartFn threadFn,
		  void *threadData,int *threadDataLen)
{ TCharmCreateData(*nThreads,threadFn,threadData,*threadDataLen); }


/*Get the unconsumed command-line arguments*/
CDECL char **TCharmArgv(void)
{
	if (TCharm::getState()!=inInit)
		CkAbort("TCharm> Can only get arguments from in init!\n");
	return cookie.getArgv();
}
CDECL int TCharmArgc(void)
{
	if (TCharm::getState()!=inInit)
		CkAbort("TCharm> Can only get arguments from in init!\n");
	return CmiGetArgc(cookie.getArgv());
}

CDECL int TCharmGetNumChunks(void)
{
	int nChunks=CkNumPes();
	char **argv=TCharmArgv();
	CmiGetArgInt(argv,"-vp",&nChunks);
	CmiGetArgInt(argv,"+vp",&nChunks);
	return nChunks;
}
FDECL int FTN_NAME(TCHARM_GET_NUM_CHUNKS,tcharm_get_num_chunks)(void)
{
	return TCharmGetNumChunks();
}


/***********************************
Callable from worker thread
*/
CDECL int TCharmElement(void)
{ return TCharm::get()->getElement();}
CDECL int TCharmNumElements(void)
{ return TCharm::get()->getNumElements();}

FDECL int FTN_NAME(TCHARM_ELEMENT,tcharm_element)(void) 
{ return TCharmElement();}
FDECL int FTN_NAME(TCHARM_NUM_ELEMENTS,tcharm_num_elements)(void) 
{ return TCharmNumElements();}

CDECL int TCharmRegister(void *data,TCharmPupFn pfn)
{ 
	if (!CmiIsomallocInRange(data))
		CkAbort("The UserData you register must be allocated on the stack!\n");
	return TCharm::get()->add(TCharm::UserData(pfn,data));
}
FDECL int FTN_NAME(TCHARM_REGISTER,tcharm_register)
	(void *data,TCpupUserDataF pfn)
{ 
	if (!CmiIsomallocInRange(data))
		CkAbort("The UserData you register must be allocated on the stack!\n");
	return TCharm::get()->add(TCharm::UserData(
		pfn,data,TCharm::UserData::isFortran()));
}

CDECL void *TCharmGetUserdata(int id)
{
	return TCharm::get()->lookupUserData(id);
}
FDECL void *FTN_NAME(TCHARM_GET_USERDATA,tcharm_get_userdata)(int *id)
{ return TCharmGetUserdata(*id); }

CDECL void TCharmMigrate(void)
{
	TCharm::get()->migrate();
}
FDECL void FTN_NAME(TCHARM_MIGRATE,tcharm_migrate)(void)
{
	TCharm::get()->migrate();
}

CDECL void TCharmDone(void)
{
	TCharm::get()->done();
}
FDECL void FTN_NAME(TCHARM_DONE,tcharm_done)(void)
{
	TCharmDone();
}



#include "tcharm.def.h"
