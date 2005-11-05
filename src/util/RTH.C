/* 
RTH runtime: calls that actually implement
RTH's execution.

Orion Sky Lawlor, olawlor@acm.org, 2003/07/23
*/
#include "RTH.h"
#include <stdio.h>
#include <stdlib.h>

enum {
	PC_START=0 /* Value of program counter at start of routine */
};

/** Describes everything you need to resume
  execution: a function, a program counter, 
  and local variables.
*/
struct RTH_StackFrame {
public:
	RTH_Routine fn; /* Function to execute (FIXME: use an index, not a pointer) */
	int pc; /* Program counter in function */
	int localsSize; /* Size, in bytes, of local variables */
	RTH_Locals *locals; /* Local variables */
	
	RTH_StackFrame(void) 
		:fn(0), pc(-123), localsSize(0), locals(0) {}
	RTH_StackFrame(RTH_Routine f,int p,int ls) 
		:fn(f), pc(p), localsSize(ls), locals(0) {}
	void invoke(RTH_Runtime *runtime);
	
	void pup(PUP::er &p);
};

/* Allocate new storage for local data */
RTH_Locals *allocLocals(int localsSize) {
	char *ret=new char[localsSize];
	/* Appropriate constructor for Userdata subclass will be called
	   from the routine's start method. */
	return (RTH_Locals *)ret;
}

/* Release storage for local data */
void freeLocals(RTH_Locals *locals) {
	/* call an in-place destructor on the locals */
	locals->~RTH_Locals();
	delete[] (char *)locals;
} 

void RTH_StackFrame::pup(PUP::er &p)
{
	p((char *)&fn,sizeof(fn)); //< evil- bytes.  Should pack function index (or something).
	p|pc; // Fine, program counter is portable
	p|localsSize;
	if (p.isUnpacking()) {
		locals=allocLocals(localsSize);
	}
	p((char *)locals,localsSize); //< evil- bytes.  Should rely on pup routine for everything.
	locals->pup(p);
}

/** Describes everything associated with a flow of control--
  execution stack, objects, and other housekeeping.
*/
class RTH_Runtime {
	/* FIXME: figure out how to paste a runtime to an array element.
	 */
	int stackptr; /* Top of stack-- next guy to return to */
	enum {maxStack=16};
	RTH_StackFrame stack[maxStack]; /* Chain of calls */
	
public:
	/* Grab the stack frame at the top-of-stack (tos) */
	inline RTH_StackFrame &tos(void) {return stack[stackptr];}
	/* Add a new stack frame, which becomes the new top-of-stack */
	void push(const RTH_StackFrame &f) {
		/* FIXME: handle stack overflow */
		stack[++stackptr]=f;
		tos().locals=allocLocals(tos().localsSize);
	}
	/* Pop the current stack frame.  Returns false if that's the end. */
	bool pop(void) {
		freeLocals(tos().locals);
		if (stackptr==0) { /* Last stack frame just returned-- we're done. */
			terminated=true;
			return false;
		} else {
			--stackptr;
			return true;
		}
	}
	
	void *obj;  /* Object to pass to function */
	int terminated; /* If true, our main routine has ended. */
	
	RTH_Runtime(void *obj_,const RTH_StackFrame &start);
	~RTH_Runtime();
	
	void pup(PUP::er &p) {
		p|stackptr;
		for (int s=0;s<=stackptr;s++) {
			stack[s].pup(p);
		}
		p|terminated;
	}
};

RTH_Runtime::RTH_Runtime(void *obj_, const RTH_StackFrame &start) {
	obj=obj_;
	terminated=false;
	stackptr=-1; /* fake value, makes "push" use first frame. */
	push(start);
}
RTH_Runtime::~RTH_Runtime() {
	// Clear out the stack, which gets rid of our local variables.
	while (!terminated) pop();
}
inline void RTH_StackFrame::invoke(RTH_Runtime *runtime) {
	(fn)(runtime,runtime->obj,locals,pc);
}

RTH_Runtime *RTH_Runtime_create(RTH_Routine fn,int localsSize,void *obj) {
	return new RTH_Runtime(obj,RTH_StackFrame(fn,PC_START,localsSize));
}

RTH_Runtime *RTH_Runtime_pup(RTH_Runtime *runtime,PUP::er &p,void *obj) {
	if (p.isUnpacking()) {
		runtime=new RTH_Runtime(obj,RTH_StackFrame());
	}
	runtime->pup(p);
	return runtime;
}

/** "Block" this RTH thread.  Saves the currently
  running function and returnPC for later "resume" call.
  Unlike a real thread, this routine returns, and you're
  expected to then exit the calling RTH routine immediately.
*/
void RTH_Runtime_suspend(RTH_Runtime *runtime,int nextPC) {
	runtime->tos().pc=nextPC;
}

/** Resume the blocked RTH thread immediately.
   This call will not return until the thread blocks or finishes.  
 */
void RTH_Runtime_resume(RTH_Runtime *runtime) {
	if (!runtime->terminated)
		runtime->tos().invoke(runtime);
}

/** Called at the end of an RTH routine.
    Allows the next routine to be started.
 */
void RTH_Runtime_done(RTH_Runtime *runtime) {
	RTH_StackFrame &topFrame=runtime->tos();
	if (runtime->pop()) 
	{ /* Pop to the calling routine-- resume him. */
		if (topFrame.pc!=PC_START) /* Resume the suspended calling routine */
			runtime->tos().invoke(runtime);
		/* else our PC==0, so we never even stopped running */
	}
	/* else we're done running */
}

/** Dispose of the RTH runtime. Does not free the user object. */
void RTH_Runtime_destroy(RTH_Runtime *runtime) {
	delete runtime;
}


/** Call this RTH subroutine inline. */
int RTH_Runtime_call(RTH_Runtime *runtime,RTH_Routine fn,int localsSize,int nextPC) {
	/* Save the old frame */
	RTH_StackFrame &oldFrame=runtime->tos();
	oldFrame.pc==PC_START; /* pretend we've never suspended */
	
	/* Push new frame for execution */
	runtime->push(RTH_StackFrame(fn,PC_START,localsSize));
	RTH_StackFrame &newFrame=runtime->tos();
	
	/* Try running the new frame immediately */
	newFrame.invoke(runtime);
	
	/* Check if the call suspended */
	if (newFrame.pc==PC_START) 
	{ /* Routine never suspended-- keep right on running! */
		return 1;
	}
	else { /* Routine (or its kids) gave up in the middle-- block our thread */
		oldFrame.pc=nextPC;
		return 0;
	}
}


RTH_Locals::~RTH_Locals() {}
void RTH_Locals::pup(PUP::er &p) {}
