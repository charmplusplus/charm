/* 
RTH runtime: Return-based "Threads" package,
for multiple flows-of-control *without* any 
crazy threading work.

I'll probably regret this, but if I make this a 
C++-based API I can make local variable access much easier.

Orion Sky Lawlor, olawlor@acm.org, 2003/07/23
*/
#ifndef __CHARM_RTH_H
#define __CHARM_RTH_H

#include "pup.h"
#include <new> /* for in-place operator new */

/** All the local variables for RTH routines
   are stored in subclasses of this type.  Keeping
   the types allows us to have a PUP routine and 
   an actual destructor. 
*/
class RTH_Locals {
public:
	virtual ~RTH_Locals();
	virtual void pup(PUP::er &p);
};


/** An opaque handle to the RTH runtime data,
  which includes the call stack and local variables.
 */
class RTH_Runtime;

/** An RTH user subroutine, declared using the
RTH_Routine_* macros.
*/
typedef void (*RTH_Routine)(RTH_Runtime *runtime,
	void *object,RTH_Locals *locals,int pc);


/************************************************
RTH library interface: used for writing libraries
that call RTH routines and/or are called by RTH routines.
*/

/** Create a new RTH runtime that will run this function
    on its first "resume" call.
*/
RTH_Runtime *RTH_Runtime_create(RTH_Routine fn,int localsSize,void *obj);

/** Pup this RTH_Runtime.  This routine can be called on 
an uninitialized RTH_Runtime if the PUP::er is unpacking.
This does *not* free the RTH_Runtime, so call RTH_Runtime_free
in your destructor as usual.
*/
RTH_Runtime *RTH_Runtime_pup(RTH_Runtime *runtime,PUP::er &p,void *obj);

/** "Block" this RTH thread.  Saves the currently
  running function and nextPC for later "resume" call.
  Unlike a real thread, this routine returns, and you're
  expected to exit the RTH_fn immediately.
  Also see the RTH_Suspend macro, which can be used
  directly from inside a user RTH_Routine.
*/
void RTH_Runtime_suspend(RTH_Runtime *runtime,int nextPC);

/** Resume the blocked RTH thread immediately.
   This call will not return until the thread blocks again.  
*/
void RTH_Runtime_resume(RTH_Runtime *runtime);

/** Call this RTH subroutine inline. Returns 0 if the
   current thread should block. (i.e., the call "failed") */
int RTH_Runtime_call(RTH_Runtime *runtime,RTH_Routine fn,int localsSize,int nextPC);

/** Called at the end of an RTH routine.
    Allows the next routine to be started.
 */
void RTH_Runtime_done(RTH_Runtime *runtime);

/** Dispose of the RTH runtime. Does not free the user object. */
void RTH_Runtime_destroy(RTH_Runtime *runtime);


/************************************************
RTH "translator":
  Instruments user's source code with calls to
  the RTH runtime.
*/

/** Begins the definition of an RTH user routine.
You define a routine foo for a class bar using 
these macros like:
   RTH_Routine_locals(foo,bar)
     int i, j;
     double z;
   RTH_Routine_code(foo,bar)
     z=0.0; 
     for (i=0;i<3;i++) z+=1.0;
   RTH_Routine_end(foo,bar)

IMPLEMENTATION: Creates a small wrapper object to hold
  the routine and the routine's local variables.
*/
#define RTH_Routine_locals(object,name) \
class RTH_Routine_##object##_##name : public RTH_Locals { \
  typedef RTH_Locals super; \
  typedef RTH_Routine_##object##_##name locals_t; \
public:  /* user's local variables go here */

/** After listing any local variables (and a pup routine),
    use this macro to start defining the body code of your
    subroutine. 
IMPLEMENTATION: Opens the actual routine, and adds a 
 "switch(pc)" statement.
*/
#define RTH_Routine_code(object,name) \
  inline void name(RTH_Runtime *RTH_impl_runtime,\
              object *c,int RTH_impl_pc) { \
    switch(RTH_impl_pc) { \
    case 0: /* Program counter at routine start (PC_START): */ \
      new ((void *)this) locals_t; /* Call our locals' constructor in-place */

/** Ends the definition of an RTH routine. 
 IMPLEMENTATION: Ends the switch statement started by 
 RTH_Routine_code, and adds the callfn static function.
*/
#define RTH_Routine_end(object,name) \
    /* end-of-routine: check if we need to resume our calling routine */ \
      RTH_Runtime_done(RTH_impl_runtime); \
      return;  \
    default: \
      printf("Bad pc %d\n",RTH_impl_pc); exit(1);/* unrecognized program counter: */ \
    }; /* Switch end */ \
  } /* User routine end */ \
\
  /* C-style call-function for user routine-- applies appropriate typecasts. */ \
  static void RTH_Call_routine(RTH_Runtime *runtime,\
              void *c,RTH_Locals *locals,int pc) \
  {\
    ((locals_t *)locals)->name(runtime,(object *)c,pc);\
  }\
\
}; /* locals_t class end */ 

/** Block the currently running RTH thread.  Can only
  be called from inside an RTH_Routine.
IMPLEMENTATION: Uses the bizarre method of stashing the 
  program counter for the next statement and returning.
  On the next resume, the statements following the suspend
  will be executed.
 */
#define RTH_Suspend() do {\
	RTH_Runtime_suspend(RTH_impl_runtime,__LINE__);\
	return;\
case __LINE__: ; \
} while(0)

/** Call this RTH subroutine (blocking).  Can only be called
    from inside an RTH_Routine.
    Store any arguments for the routine inside your object. 
FIXME: add real argument-passing and return types, based 
  on pup (somehow).
*/
#define RTH_Call(object,name) do{ \
     if (!RTH_Runtime_call(RTH_impl_runtime,RTH_Routine_lookup(object,name),__LINE__)) \
        return;\
case __LINE__: ; \
} while(0)

/** Given a routine name, convert to a call-function and locals size.
  Can be called from inside or outside an RTH_Routine.
*/
#define RTH_Routine_lookup(object,name) \
   RTH_Routine_##object##_##name::RTH_Call_routine, sizeof(RTH_Routine_##object##_##name)

#endif

