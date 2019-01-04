/*
Threaded Charm++ "Framework Framework" Implementation header

Implements an array of migratable threads.
Provides utility routines for registering user
data, stopping, starting, and migrating threads.

Orion Sky Lawlor, olawlor@acm.org, 11/19/2001
*/
#ifndef __CHARM_TCHARM_IMPL_H
#define __CHARM_TCHARM_IMPL_H

#include "pup.h"
#include "pup_c.h"
#include "charm-api.h"
#include "tcharmc.h"
#include "cklists.h"
#include "memory-isomalloc.h"

#include "cmitls.h"

#if 0
     /*Many debugging statements:*/
#    define DBG(x) ckout<<"["<<thisIndex<<","<<CkMyPe()<<"] TCHARM> "<<x<<endl;
#    define DBGX(x) ckout<<"PE("<<CkMyPe()<<") TCHARM> "<<x<<endl;
#else
     /*No debugging statements*/
#    define DBG(x) /*empty*/
#    define DBGX(x) /*empty*/
#endif

class TCharmTraceLibList;

/// Used to ship around system calls.
class callSystemStruct {
public:
	const char *cmd; ///< Shell command to execute.
	int *ret; ///< Place to store command's return value.
};
PUPbytes(callSystemStruct)


#include "tcharm.decl.h"

class TCharm;

// This little class holds values between a call to TCHARM_Set_* 
//   and the subsequent TCHARM_Create_*.  It should be moved
//   into a parameter to TCHARM_Create.
class TCHARM_Thread_options {
public:
	int stackSize; /* size of thread execution stack, in bytes */
	bool exitWhenDone; /* flag: call CkExit when thread is finished. */
	// Fill out the default thread options:
	TCHARM_Thread_options(int doDefault);
	TCHARM_Thread_options() {}

	void sanityCheck(void);
};

class TCharmInitMsg : public CMessage_TCharmInitMsg {
 public:
	//Function to start thread with:
	//CthVoidFn threadFn;
	int threadFn;
	//Initial thread parameters:
	TCHARM_Thread_options opts;
	//Array size (number of elements)
	int numElements;
	//Data to pass to thread:
	char *data;

	TCharmInitMsg(int threadFn_,const TCHARM_Thread_options &opts_)
		:threadFn(threadFn_), opts(opts_) {}
};

extern bool tcharm_nothreads;

//Thread-local variables:
CtvExtern(TCharm *,_curTCharm);

class TCharm: public CBase_TCharm
{
 private:
	friend class TCharmAPIRoutine; //So he can get to heapBlocks:

	CthThread tid; //Our migratable thread
	CmiIsomallocBlockList *heapBlocks; //Migratable heap data
#if CMI_SWAPGLOBALS
	CtgGlobals threadGlobals; //Global data
#endif

	//isSelfDone is added for out-of-core emulation in BigSim
	//when thread is brought back into core, ResumeFromSync is called
	//so if the thread has finished its stuff, it should not start again
	bool isStopped, exitWhenDone, isSelfDone, asyncMigrate;

	//Informational data about the current thread:
	class ThreadInfo {
	public:
		CProxy_TCharm tProxy; //Our proxy
		int thisElement; //Index of current element
		int numElements; //Number of array elements
	};
	ThreadInfo threadInfo;

	TCharmInitMsg *initMsg; //Thread initialization data
	double timeOffset; //Value to add to CkWallTimer to get my clock

 public:
	//User's heap-allocated/global data:
	class UserData {
		CthThread t;
		size_t    pos;
		char mode;
		TCHARM_Pup_fn cfn;
		TCHARM_Pup_global_fn gfn;
	public:
		UserData(int i=0) {pos=0; mode='?'; cfn=NULL; gfn=NULL;}
		UserData(TCHARM_Pup_fn cfn_,CthThread t_,void *p)
			{cfn=cfn_; t=t_; pos=CthStackOffset(t, (char *)p); mode='c';}
		UserData(TCHARM_Pup_global_fn gfn_,CthThread t_,void *p)
			{gfn=gfn_; t=t_; pos=CthStackOffset(t, (char *)p); mode='g';}
		inline void *getData() const {return pos==0?NULL:CthPointer(t, pos);}
		void pup(PUP::er &p);
		void update(CthThread t_) { t=t_; }
		friend inline void operator|(PUP::er &p,UserData &d) {d.pup(p);}
	};
	//New interface for user data:
	CkVec<UserData> sud;

	//Tiny semaphore-like pointer producer/consumer
	class TCharmSemaphore {
	public:
		int id; //User-defined identifier
		void *data; //User-defined data
		CthThread thread; //Waiting thread, or 0 if none

		TCharmSemaphore() { id=-1; data=NULL; thread=NULL; }
		TCharmSemaphore(int id_) { id=id_; data=NULL; thread=NULL; }
	};
	/// Short, unordered list of waiting semaphores.
	CkVec<TCharmSemaphore> sema;

 private:
	//Old interface for user data:
	enum {maxUserData=16};
	int nUd;
	UserData ud[maxUserData];

	void pupThread(PUP::er &p);
	void ResumeFromSync();

 public:
	TCharm(TCharmInitMsg *initMsg);
	TCharm(CkMigrateMessage *);
	~TCharm();

	virtual void ckJustMigrated();
	virtual void ckJustRestored();
	virtual void ckAboutToMigrate();

	void migrateDelayed(int destPE);
	void atBarrier();
	void atExit(CkReductionMsg *msg) noexcept;
	void clear();

	//Pup routine packs the user data and migrates the thread
	virtual void pup(PUP::er &p);

	TCharmSemaphore *findSema(int id);
	TCharmSemaphore *getSema(int id);
	void freeSema(TCharmSemaphore *);

	/// Store data at the semaphore "id".
	///  The put can come before or after the get.
	void semaPut(int id,void *data);

	/// Retreive data from the semaphore "id", returning NULL if not there.
	void *semaPeek(int id);

	/// Retreive data from the semaphore "id".
	///  Blocks if the data is not immediately available.
	void *semaGets(int id);

	/// Retreive data from the semaphore "id".
	///  Blocks if the data is not immediately available.
	///  Consumes the data, so another put will be required for the next get.
	void *semaGet(int id);

	//One-time initialization
	static void nodeInit();
	static void procInit();

	//Start running the thread for the first time
	void run() noexcept;

	inline double getTimeOffset() const noexcept { return timeOffset; }

	//Client-callable routines:
	//Sleep till entire array is here
	void barrier() noexcept;

	//Block, migrate to destPE, and resume
	void migrateTo(int destPE) noexcept;

#if CMK_FAULT_EVAC
	void evacuate() noexcept;
#endif

	//Thread finished running
	void done(int exitcode) noexcept;

	//Register user data to be packed with the thread
	int add(const UserData &d) noexcept;
	void *lookupUserData(int ud) noexcept;

	inline static TCharm *get() noexcept {
		TCharm *c=getNULL();
#if CMK_ERROR_CHECKING
		if (!c) ::CkAbort("TCharm has not been initialized!\n");
#endif
		return c;
	}
	inline static TCharm *getNULL() noexcept {return CtvAccess(_curTCharm);}
	inline CthThread getThread() noexcept {return tid;}
	inline const CProxy_TCharm &getProxy() const noexcept {return threadInfo.tProxy;}
	inline int getElement() const noexcept {return threadInfo.thisElement;}
	inline int getNumElements() const noexcept {return threadInfo.numElements;}

	//Start/stop load balancer measurements
	inline void stopTiming() noexcept {ckStopTiming();}
	inline void startTiming() noexcept {ckStartTiming();}

	//Block our thread, run the scheduler, and come back
	void schedule() noexcept {
		DBG("thread schedule");
		start(); // Calls CthAwaken
		stop(); // Calls CthSuspend
	}


	//As above, but start/stop the thread itself, too.
	void stop() noexcept { //Blocks; will not return until "start" called.
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
		/* SUBTLE: We have to do the get() because "this" may have changed
		 * during a migration-suspend.  If you access *any* members
		 * from this point onward, you'll cause heap corruption if
		 * we're resuming from migration!  (OSL 2003/9/23) */
		TCharm *dis=TCharm::get();
		dis->isStopped=false;
		dis->startTiming();
	}

	void start() noexcept {
		isStopped=false; // do not migrate while running
		DBG("thread resuming soon");
		CthAwaken(tid);
	}

	//Aliases:
	inline void suspend() noexcept {stop();}
	inline void resume() noexcept {
		//printf("in thcarm::resume, isStopped=%d\n", isStopped);
		if (isStopped){
		    start();
		}
		/*else {
		    printf("[%d] TCharm resume called on already running thread pe %d \n",thisIndex,CkMyPe());
		}*/
	}

	//Go to sync, block, possibly migrate, and then resume
	void migrate() noexcept;
	void async_migrate() noexcept;
	void allow_migrate();

	//Entering thread context: turn stuff on
	static void activateThread(void) noexcept {
		// NOTE: if changing the body of this function, you also need to
		//       modify TCharmAPIRoutine's destructor, which has the body
		//       of this routine inlined into it to avoid extra branching.
		TCharm *tc=CtvAccess(_curTCharm);
		if (tc!=NULL) {
			if (tc->heapBlocks)
				CmiIsomallocBlockListActivate(tc->heapBlocks);
#if CMI_SWAPGLOBALS
			if (tc->threadGlobals)
				CtgInstall(tc->threadGlobals);
#endif
		}
	}
	//Leaving this thread's context: turn stuff back off
	static void deactivateThread() noexcept {
		CmiIsomallocBlockListActivate(NULL);
#if CMI_SWAPGLOBALS
		CtgInstall(NULL);
#endif
	}

	/// System() call emulation:
	int system(const char *cmd);
	void callSystem(const callSystemStruct &s);

	inline CthThread getTid() { return tid; }
};

void TCHARM_Api_trace(const char *routineName, const char *libraryName) noexcept;

#if CMK_TRACE_ENABLED
typedef std::unordered_map<std::string, int> funcmap;
CsvExtern(funcmap*, tcharm_funcmap);

static
int tcharm_routineNametoID(char const *routineName) noexcept
{
  funcmap::iterator it;
  it = CsvAccess(tcharm_funcmap)->find(routineName);

  if (it != CsvAccess(tcharm_funcmap)->end())
    return it->second;

  return -1;
}
#endif

// Constructed at entrance into all API routines, destructed at exit from API routines:
// - Disables/enables migratable malloc
// - Swap global variables privatized via -swapglobals
// - Swap TLS variables for -tlsglobals
// - Traces library code entry/exit with appropriate build flags
class TCharmAPIRoutine {
private:
	bool doIsomalloc; // Whether to enable/disable Isomalloc for Heap memory
	bool doSwapglobals; // Whether to swap sets of global variables
	tlsseg_t oldtlsseg; // for TLS globals
#if CMK_TRACE_ENABLED
	double start; // starting time of trace event
	int tcharm_routineID; // TCharm routine ID that is traced
#endif
#if CMK_BIGSIM_CHARM
	void *callEvent; // The BigSim-level event that called into the library
	int pe;          // in case thread migrates
#endif

public:
	// Entering Charm++ from user code
#if CMK_TRACE_ENABLED
	TCharmAPIRoutine(const char *routineName, const char *libraryName) noexcept {
#else
	TCharmAPIRoutine() noexcept {
#endif
#if CMK_BIGSIM_CHARM
		// Start a new event, so we can distinguish between client
		// execution and library execution
		_TRACE_BG_TLINE_END(&callEvent);
		_TRACE_BG_END_EXECUTE(0);
		pe = CmiMyPe();
		_TRACE_BG_BEGIN_EXECUTE_NOMSG(routineName, &callEvent, 0);
#endif
#if CMK_TRACE_ENABLED
		start = CmiWallTimer();
		tcharm_routineID = tcharm_routineNametoID(routineName);
#endif

		doIsomalloc = (CmiIsomallocBlockListCurrent() != NULL);
#if CMI_SWAPGLOBALS
		doSwapglobals = (CtgCurrentGlobals() != NULL);
#endif
		if (CmiThreadIs(CMI_THREAD_IS_TLS)) {
			CtgInstallMainThreadTLS(&oldtlsseg); //switch to main thread
		}
		//Disable migratable memory allocation while in Charm++:
		TCharm::deactivateThread();

#if CMK_TRACE_ENABLED
		TCHARM_Api_trace(routineName,libraryName);
#endif
	}

	// Returning to user code from Charm++
	~TCharmAPIRoutine() noexcept {
#if CMK_TRACE_ENABLED
		double stop = CmiWallTimer();
#endif

		TCharm *tc=CtvAccess(_curTCharm);
		if(tc != NULL){
			//Reenable migratable memory allocation
			// NOTE: body of TCharm::activateThread() is inlined here:
			if(doIsomalloc){
				CmiIsomallocBlockListActivate(tc->heapBlocks);
			}
#if CMI_SWAPGLOBALS
			if(doSwapglobals){
				CtgInstall(tc->threadGlobals);
			}
#endif
		}
		if (CmiThreadIs(CMI_THREAD_IS_TLS)) {
			tlsseg_t cur;
			CtgInstallCthTLS(&cur, &oldtlsseg); // switch back to user's CthThread
		}

#if CMK_BIGSIM_CHARM
		void *log;
		_TRACE_BG_TLINE_END(&log);
		_TRACE_BG_END_EXECUTE(0);
		_TRACE_BG_BEGIN_EXECUTE_NOMSG("user_code", &log, 0);
		if (CmiMyPe() == pe) _TRACE_BG_ADD_BACKWARD_DEP(callEvent);
#endif
#if CMK_TRACE_ENABLED
		if (tcharm_routineID > -1) // is it a routine we care about?
			traceUserBracketEventNestedID(tcharm_routineID, start, stop, TCHARM_Element());
#endif
	}
};


#define TCHARMAPI(routineName) TCHARM_API_TRACE(routineName,"tcharm");

//Node setup callbacks: called at startup on each node
FLINKAGE void FTN_NAME(TCHARM_USER_NODE_SETUP,tcharm_user_node_setup)(void);
FLINKAGE void FTN_NAME(TCHARM_USER_SETUP,tcharm_user_setup)(void);


#endif


