/*
Threaded Charm++ "Framework Framework" 

This is the interface used by library writers--
people who use TCharm to build their own library.

Orion Sky Lawlor, olawlor@acm.org, 7/17/2002
*/
#ifndef __CHARM_TCHARMLIB_H
#define __CHARM_TCHARMLIB_H

#include "tcharm_impl.h"

/*
Library "fallback setup" routine.
From an initcall, you register one of your routines 
here in case the user doesn't write a TCHARM_User_setup
routine.  Your fallback version sets up (only) your 
library by creating some TCharm threads and attaching to them,
like:

void myLibFallbackSetupFn(void) {
	TCHARM_Create(TCHARM_Get_num_chunks(),myLibUserStart);
        myLib_Attach();
}
*/
typedef void (*TCHARM_Fallback_setup_fn)(void);
void TCHARM_Set_fallback_setup(TCHARM_Fallback_setup_fn f);


/*
These two routines need to be called from your Attach routine.
The "start" call finds the last-created set of tcharm threads,
copies their arrayID into retTCharmArray, and returns an 
opts object bound the the tcharm threads.  You pass this opts
object to your array's ckNew.
The "finish" call registers your newly-created array as a
client of this set of TCharm threads.
*/
CkArrayOptions TCHARM_Attach_start(CkArrayID *retTCharmArray,int *retNumElts=0);
void TCHARM_Attach_finish(const CkArrayID &libraryArray);


/*
A simple client array that can be bound to a tcharm array.
You write a TCharm library by inheriting your array from this class.
You'll have to pass the TCharm arrayID to our constructor, and
then call tcharmClientInit().
As soon as your startup work is done, you MUST call "thread->ready();"
*/
class TCharmClient1D : public ArrayElement1D {
  CProxy_TCharm threadProxy; //Proxy for our bound TCharm array
 protected:
  /* This is the actual TCharm object that manages our thread.
     You use this like "thread->suspend();", "thread->resume();",
     etc.  
   */
  TCharm *thread; 
  inline void findThread(void) {
    thread=threadProxy[thisIndex].ckLocal();  
    if (thread==NULL) CkAbort("Can't locate TCharm thread!");
  }
  
  //Clients need to override this function to set their
  // thread-private variables.  You usually use something like:
  //  CtvAccessOther(forThread,_myFooPtr)=this;
  virtual void setupThreadPrivate(CthThread forThread) =0;
  
 public:
  TCharmClient1D(const CkArrayID &threadArrayID) 
    :threadProxy(threadArrayID)
  {
    //Argh!  Can't call setupThreadPrivate yet, because
    // virtual functions don't work within constructors!
    findThread();
  }
  TCharmClient1D(CkMigrateMessage *m) //Migration, etc. constructor
  {
    thread=NULL;
  }
  
  //You MUST call this from your constructor:
  inline void tcharmClientInit(void) {
    setupThreadPrivate(thread->getThread());
  }
  
  virtual void ckJustMigrated(void);
  virtual void pup(PUP::er &p);
};


/*
Library API Calls.  The pattern:
  TCHARM_API_TRACE("myRoutineName","myLibName");
MUST be put at the start of every
user-callable library routine.  The string name is
used for debugging printouts, and eventually for
tracing (once tracing is generalized).
*/
#ifndef CMK_OPTIMIZE
#  define TCHARM_API_TRACE(routineName,libraryName) \
	TCharmAPIRoutine apiRoutineSentry;\
	TCHARM_Api_trace(routineName,libraryName)
#else
#  define TCHARM_API_TRACE(routineName,libraryName) \
	TCharmAPIRoutine apiRoutineSentry
#endif
void TCHARM_Api_trace(const char *routineName,const char *libraryName);


/* Get the currently running TCharm threads: */
CkArrayID TCHARM_Get_threads(void);

#endif /*def(thisHeader)*/
