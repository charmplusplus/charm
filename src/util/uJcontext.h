/**
 A drop-in replacement for the SYSV set/get/makecontext routines,
 implemented using setjmp and longjmp.
*/
#ifndef __UIUC_UJCONTEXT_H
#define __UIUC_UJCONTEXT_H

#include <setjmp.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Describes the stack for a uJcontext flow of control. */
typedef struct uJcontext_stack_t {
        void *ss_sp;
        int ss_flags;
        size_t ss_size;
} uJcontext_stack_t;

/** Start a uJcontext flow of control. */
typedef void (*uJcontext_fn_t)(void *a1,void *a2);

/** Represents a uJcontext flow of control. */
typedef struct uJcontext_t {
    uJcontext_stack_t uc_stack;
    struct uJcontext_t *uc_link; /* not implemented */
    int uc_flags, uc_sigmask; /* not implemented */
 
/* Non-standard fields */
    void (*uc_swap)(struct uJcontext_t *newContext);
    
    /* Storage for processor registers, etc. when thread not running */
    jmp_buf _uc_jmp_buf;
    
    /* Only used to start the thread; useless afterwards */
    uJcontext_fn_t _uc_fn; /* user function to call */
    void *_uc_args[2]; /* user arguments to pass in */
} uJcontext_t;

/** Called before switching to a uJcontext flow of control. */
typedef void (*uJcontext_swap_fn_t)(struct uJcontext_t *newContext);

/* Get user context and store it in variable pointed to by UCP.  */
extern int getJcontext (uJcontext_t *__ucp) ;

/* Set user context from information of variable pointed to by UCP.  */
extern int setJcontext (const uJcontext_t *__ucp) ;

/* Save current context in context variable pointed to by OUCP and set
   context from variable pointed to by UCP.  */
extern int swapJcontext (uJcontext_t * __oucp,
                        const uJcontext_t * __ucp) ;

/* Manipulate user context UCP to continue with calling functions FUNC
   and the ARGC-1 parameters following ARGC when the context is used
   the next time in `setcontext' or `swapcontext'.
   
   UGLY NONSTANDARD HACK: always take two arguments, a1 and a2.
*/
extern void makeJcontext (uJcontext_t *__ucp, uJcontext_fn_t func,
                         int __argc, void *a1,void *a2);

#ifdef __cplusplus
};
#endif

#endif
