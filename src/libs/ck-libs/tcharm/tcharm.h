/*
Threaded Charm++ "Framework Framework"

Implements an array of migratable threads.
Provides utility routines for registering user
data, stopping, starting, and migrating threads.

Orion Sky Lawlor, olawlor@acm.org, 11/19/2001
*/
#ifndef __CHARM_TCHARM_H
#define __CHARM_TCHARM_H

#include "pup.h"
#include "pup_c.h"
#include "charm-api.h"
#include "tcharmc.h"
#include "cklists.h"
#include "memory-isomalloc.h"

//User's readonly global variables, set exactly once after initialization
class TCharmReadonlys {
	//There's only one (shared) list per node, so no CpvAccess here
	static CkVec<TCpupReadonlyGlobal> entries;
 public:
	static void add(TCpupReadonlyGlobal fn);
	//Pups all registered readonlys
	void pup(PUP::er &p);
};
PUPmarshall(TCharmReadonlys);

class TCharmTraceLibList;

#include "tcharm.decl.h"

class TCharmReadonlyGroup : public Group {
 public:
	//Just unpacking the parameter is enough:
	TCharmReadonlyGroup(const TCharmReadonlys &r) { /*nothing needed*/ }
};

class TCharm;

class TCharmInitMsg : public CMessage_TCharmInitMsg {
 public:
	//Function to start thread with:
	CthVoidFn threadFn;
	//Initial stack size, in bytes:
	int stackSize;
	//Array size (number of elements)
	int numElements;
	//Data to pass to thread:
	char *data;
	
	TCharmInitMsg(CthVoidFn threadFn_,int stackSize_)
		:threadFn(threadFn_), stackSize(stackSize_) {}
};

//Current computation location
typedef enum {inNodeSetup,inInit,inDriver,inFramework,inPup} inState;

//Thread-local variables:
CtvExtern(TCharm *,_curTCharm);
CpvExtern(inState,_stateTCharm);

CDECL {typedef void (*TCpupUserDataC)(pup_er p,void *data);};
FDECL {typedef void (*TCpupUserDataF)(pup_er p,void *data);};

class TCharm: public ArrayElement1D
{
 public:
	
	//User's heap-allocated/global data:
	class UserData {
		void *data; //user data pointer
		bool isC;
		TCpupUserDataC cfn;
		TCpupUserDataF ffn;
	public:
		UserData() {data=NULL; cfn=NULL; ffn=NULL;}
		UserData(TCpupUserDataC cfn_,void *data_)
			{cfn=cfn_; data=data_; isC=true;}
		class isFortran{};
		UserData(TCpupUserDataF ffn_,void *data_,isFortran tag)
			{ffn=ffn_; data=data_; isC=false;}
		void *getData(void) {return data;}
		void pup(PUP::er &p);
	};
	
	//One-time initialization
	static void nodeInit(void);
 private:
	//Informational data about the current thread:
	class ThreadInfo {
	public:
		CProxy_TCharm tProxy; //Our proxy
		int thisElement; //Index of current element
		int numElements; //Number of array elements
	};
	
	TCharmInitMsg *initMsg; //Thread initialization data
	CthThread tid; //Our migratable thread
	friend class TCharmAPIRoutine; //So he can get to heapBlocks:
	CmiIsomallocBlockList *heapBlocks; //Migratable heap data
	
	bool isStopped;
	ThreadInfo threadInfo;
	double timeOffset; //Value to add to CkWallTimer to get my clock

	enum {maxUserData=16};
	int nUd;
	UserData ud[maxUserData];

	void ResumeFromSync(void);

#ifdef CMK_OPTIMIZE
	static inline void check(void) {}
#else
	static void check(void);
#endif

 public:
	TCharm(TCharmInitMsg *initMsg);
	TCharm(CkMigrateMessage *);
	~TCharm();
	
	//Pup routine packs the user data and migrates the thread
	virtual void pup(PUP::er &p);

	//Start running the thread for the first time
	void run(void);

	inline double getTimeOffset(void) const { return timeOffset; }

//Client-callable routines:
	//One client is ready to run
	void ready(void);

	//Sleep till entire array is here
	void barrier(void);

	//Thread finished running
	void done(void);

	//Register user data to be packed with the thread
	int add(const UserData &d);
	void *lookupUserData(int ud);
	
	inline static TCharm *get(void) {check(); return CtvAccess(_curTCharm);}
	inline static inState getState(void) {return CpvAccess(_stateTCharm);}
	inline static void setState(inState to) {CpvAccess(_stateTCharm)=to;}
	inline CthThread getThread(void) {return tid;}
	inline const CProxy_TCharm &getProxy(void) const {return threadInfo.tProxy;}
	inline int getElement(void) const {return threadInfo.thisElement;}
	inline int getNumElements(void) const {return threadInfo.numElements;}

	//Start/stop load balancer measurements
	inline void stopTiming(void) {ckStopTiming();}
	inline void startTiming(void) {ckStartTiming();}

	//Block our thread, run the scheduler, and come back
	inline void schedule(void) {
		stopTiming();
		CthYield();
		startTiming();
	}

	//As above, but start/stop the thread itself, too.
	void stop(void); //Blocks; will not return until "start" called.
	void start(void);
	//Aliases:
	inline void suspend(void) {stop();}
	inline void resume(void) {start();}

	//Go to sync, block, possibly migrate, and then resume
	void migrate(void);

	//Make subsequent malloc's go into our list:
	inline void activateHeap(void) {
		CmiIsomallocBlockListActivate(heapBlocks);
	}
	//Disable migratable memory
	inline void deactivateHeap(void) {
		CmiIsomallocBlockListActivate(NULL);
	}
};

//Controls array startup, ready, run and shutdown
class TCharmCoordinator {
	static int nArrays; //Total number of running thread arrays
	static TCharmCoordinator *head; //List of coordinators

	TCharmCoordinator *next; //Next coordinator in list
	CProxy_TCharm threads;//The threads I coordinate
	int nThreads;//Number of threads (array elements)
	int nClients; //Number of bound client arrays
	int nReady; //Number of ready clients
public:
	TCharmCoordinator(CkArrayID threads,int nThreads);
	~TCharmCoordinator();
	void addClient(const CkArrayID &client);
	void clientReady(void);
	void clientBarrier(void);
	void clientDone(void);
	
	static int getTotal(void) {
		return nArrays;
	}
};

//Controls initial setup (main::main & init routines)
class TCharmSetupCookie {
	enum {correctMagic=0x5432abcd};
	int magic; //To make sure this is actually a cookie
	
	int stackSize; //Thread stack size, in bytes
	char **argv; //Command-line arguments
	
	CkArrayID tc; //Handle to last-created TCharm array
	int numElements; //Number of elements in last-created TCharm
	TCharmCoordinator *coord; 

	//Points to the active cookie
	static TCharmSetupCookie *theCookie;
	friend class TCharmMain;
 public:
	TCharmSetupCookie(char **argv_);
	
#ifdef CMK_OPTIMIZE //Skip check, for speed
	inline TCharmSetupCookie &check(void) {return *this;}
#else
	TCharmSetupCookie &check(void);
#endif
	void setStackSize(int sz) {stackSize=sz;}
	int getStackSize(void) const {return stackSize;}
	char **getArgv(void) {return argv;}
	
	bool hasThreads(void) const {return 0!=coord;}
	const CkArrayID &getThreads(void) const {return tc;}
	TCharmCoordinator *getCoordinator(void) {return coord;}
	int getNumElements(void) const {return numElements;}

	void addClient(const CkArrayID &client) {coord->addClient(client);}
	
	void setThreads(const CkArrayID &aid,int nel);

	static TCharmSetupCookie *get(void) {return theCookie;}
};

//Created in all API routines: disables/enables migratable malloc
class TCharmAPIRoutine {
 public:
	TCharmAPIRoutine() { //Entering Charm++ from user code
		//Disable migratable memory allocation while in Charm++:
		CmiIsomallocBlockListActivate(NULL);
	}
	~TCharmAPIRoutine() { //Returning to user code from Charm++:
		//Reenable migratable memory allocation
		TCharm *tc=CtvAccess(_curTCharm);
		if (tc!=NULL) tc->activateHeap();
	}
};

#ifndef CMK_OPTIMIZE
#  define TCHARM_API_TRACE(routineName,libraryName) \
	TCharmAPIRoutine apiRoutineSentry;\
	TCharmApiTrace(routineName,libraryName)
#else
#  define TCHARM_API_TRACE(routineName,libraryName) \
	TCharmAPIRoutine apiRoutineSentry
#endif
void TCharmApiTrace(const char *routineName,const char *libraryName);

/*The pattern:
  TCHARMAPI("routineName");
should be put at the start of every
user-callable library routine.  The string name is
used for debugging printouts, and eventually for
tracing (once tracing is generalized).
*/
#define TCHARMAPI(routineName) TCHARM_API_TRACE(routineName,"tcharm");


//Node setup callbacks: called at startup on each node
FDECL void FTN_NAME(TCHARM_USER_NODE_SETUP,tcharm_user_node_setup)(void);
FDECL void FTN_NAME(TCHARM_USER_SETUP,tcharm_user_setup)(void);

//Library fallback setup routine:
typedef void (*TCharmFallbackSetupFn)(void);
void TCharmSetFallbackSetup(TCharmFallbackSetupFn f);

#endif


