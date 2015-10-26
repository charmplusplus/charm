/**
 My own jump-buffer based SYSV set/get/makecontext routines.
 
 Do a "grep -R context glibc/sysdeps" to see how glibc does it.
 Orion Sky Lawlor, olawlor@acm.org, 2004/2/13
*/
#include <stdio.h> /* for debugging prints */
#include <stdlib.h> /* for NULL */
#if CMK_HAS_ALLOCA_H
#  include <alloca.h> 
#endif

#include "uJcontext.h"
#ifdef _MSC_VER
#  define alloca _alloca
#endif

#ifdef __MINGW_H
#error "not supported under cygwin and mingw!"
#endif

/* Enable this define to get lots of debugging printouts */
#define VERBOSE(x) /* x */

#if CMK_HAS_UNDERSCORE_SETJMP
#define SETJMP    _setjmp
#define LONGJMP   _longjmp
#else
#define SETJMP     setjmp
#define LONGJMP    longjmp
#endif

/* Return an approximation of the top of the stack */
static void *getStack(void) {
	int x = 0;
        void *p=&x;
	return p;
}

VERBOSE(
static void printStack(void) {
	printf(" from stack %p\n",getStack());
}
)

static void threadFatal(const char *why) {
	fprintf(stderr,"Fatal thread error in uJcontext> %s\n",why);
	exit(1);
}

/* Get user context and store it in variable pointed to by UCP.  */
int getJcontext (uJcontext_t *u)
{
	u->uc_link=0;
	u->uc_stack.ss_sp=getStack();
	u->uc_stack.ss_size=0;
	u->uc_stack.ss_flags=0;
	u->uc_flags=0;
	u->uc_sigmask=0;
	u->uc_swap=0;
	u->_uc_fn=NULL;
	u->_uc_args[0]=u->_uc_args[1]=NULL;
	return 0;
}

/* global variable used to defeat compiler over-optimization which gets rid of alloca */
char *_dummyAllocaSetJcontext;

/* Set user context from information of variable pointed to by UCP.  */
int setJcontext (const uJcontext_t *u)
{
	register uJcontext_t *mu=(uJcontext_t *)u;
	
	CmiAssert (mu != NULL);

	/* Call the user's swap function if needed (brings in thread stack) */
	if (mu->uc_swap) mu->uc_swap(mu);
	
	if (mu->_uc_fn==NULL)
	{ /* Start running an existing thread */
		LONGJMP(mu->_uc_jmp_buf,0);
		threadFatal("Fatal error performing longjmp"); 
	}
	else /* mu->_uc_fn != NULL, so thread hasn't started yet */
	{ /* first time through-- start up the new thread */
		/* Change stack pointer to point to new stack */
		char *new_sp=(char *)mu->uc_stack.ss_sp;
		
		/**
		  Some machines reference variables (e.g., arguments,
		  saved link) from their caller's stack frame, so 
		  we need to leave a little extra space before the 
		  new stack pointer.
		*/
		int caller_distance=8*sizeof(void *);
		
		uJcontext_fn_t mu_fn=mu->_uc_fn; /* save function before marking NULL */
		mu->_uc_fn=NULL;

		/* FIXME: only if stack grows down (they all do...) */
		new_sp+=mu->uc_stack.ss_size-caller_distance;
		
#ifndef CMK_BLUEGENEQ
		VERBOSE( printf("About to switch to stack %p ",new_sp); printStack(); )
		if (1) { /* change to new stack */
#ifdef _MSC_VER 
/**
 Microsoft's _alloca is too smart-- it checks the pages being
 allocated to see if they're all valid stack pages, so our 
 hack of using a huge alloca to change the stack pointer 
 instead results in an address error (from "chkstk").

 Manually changing the stack pointer seems to work properly;
 and because local variables are referenced via the frame 
 pointer (ebp), this is all that's needed.
*/
			__asm { mov esp, new_sp };
#elif defined(__CYGWIN__)
			asm ( "mov %0, %%esp\n"::"m"(new_sp));
#elif 0 && defined(__APPLE__) && CMK_64BIT
			asm ( "mov %0, %%rsp\n"::"m"(new_sp));
#elif 0 /* Blue Gene/Light gcc PPC assembly version: */
			asm __volatile__ ("mr 1,%0" :: "r"(new_sp));
#else /* Portable alloca version */
			char *old_sp = NULL;
			register CmiInt8 allocLen;
			old_sp =  (char *)&old_sp;
			allocLen=old_sp-new_sp;
                  
		        VERBOSE( printf("calling alloca with %lld", allocLen); printStack(); )
			_dummyAllocaSetJcontext = alloca(allocLen);  /* defeat the compiler optimization! */
#endif
		}
		VERBOSE( printf("After alloca"); printStack(); )
		/* Call the user function for the thread */
		mu_fn(mu->_uc_args[0],mu->_uc_args[1]);

		/* Back from user function-- jump to next thread */
		if (mu->uc_link!=0)
		        setJcontext(mu->uc_link);
		else
			threadFatal("uc_link not set-- thread should never return");
#else
		//Start the thread by changing the stack pointer and calling the start function
		uint64_t startiar = *((uint64_t*)mu_fn);
		//uint64_t sp  = ((uint64_t)(ptr->stackptr) + ptr->stacksize - 1024) & ~(0x1f);
		asm volatile("mr 3, %0;"
			     "mtlr 3;"
			     "mr 1, %1;"
			     "mr 3, %2;"
			     "mr 4, %3;"
			     "blr;"
			     : : "r" (startiar), "r" (new_sp), "r" (mu->_uc_args[0]), "r" (mu->_uc_args[1]) : "r1", "r3", "r4", "memory");
		
		//mu->uc_link cannot be null. Call setJcontext to start next thread or return to master
		startiar = *((uint64_t*)setJcontext);
		asm volatile("mr 3, %0;"
			     "mtlr 3;"
			     "mr 3, %1;"
			     "blr;"
			     : : "r" (startiar), "r" (mu->uc_link) : "r3", "memory");
#endif	
		
	}
	return 0;
}

/* Save current context in context variable pointed to by o and set
   context from variable pointed to by u.  */
int swapJcontext (uJcontext_t *o,
                        const uJcontext_t *u)
{
	register uJcontext_t *mu=(uJcontext_t *)u;
	VERBOSE( printf("swapJcontext(%p,%p)",o,u); printStack(); )
	if (0==SETJMP(o->_uc_jmp_buf))
		setJcontext(mu); /* direct path-- switch to new thread */
	else { /* old thread resuming-- */
		VERBOSE( printf("swapJcontext returning to %p",mu); printStack(); )
	}
	return 0;
}

/* Manipulate user context UCP to continue with calling functions FUNC
   and the ARGC-1 parameters following ARGC when the context is used
   the next time in `setJcontext' or `swapJcontext'.
   
   UGLY NONSTANDARD HACK: always take two arguments, a and b.
*/
void makeJcontext (uJcontext_t *u, uJcontext_fn_t __func,
                         int __argc, void *a,void *b)
{
	VERBOSE( printf("makeJcontext(%p)",u); printStack(); )
	u->_uc_fn=__func;
	u->_uc_args[0]=a;
	u->_uc_args[1]=b;
}



