#ifndef QT_H
#define QT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <qtmd.h>

/* A QuickThreads thread is represented by it's current stack pointer.
   To restart a thread, you merely need pass the current sp (qt_t*) to
   a QuickThreads primitive.  `qt_t*' is a location on the stack.  To
   improve type checking, represent it by a particular struct. */

typedef struct qt_t {
  char dummy;
} qt_t;


/* Alignment is guaranteed to be a power of two. */
#ifndef QT_STKALIGN
  #error "Need to know the machine-dependent stack alignment."
#endif

#define QT_STKROUNDUP(bytes) \
  (((bytes)+QT_STKALIGN) & ~(QT_STKALIGN-1))


/* Find ``top'' of the stack, space on the stack. */
#ifndef QT_SP
#ifdef QT_GROW_DOWN
#define QT_SP(sto, size)	((qt_t *)(&((char *)(sto))[(size)]))
#endif
#ifdef QT_GROW_UP
#define QT_SP(sto, size)	((void *)(sto))
#endif
#if !defined(QT_SP)
  #error "QT_H: Stack must grow up or down!"
#endif
#endif


/* The type of the user function:
   For non-varargs, takes one void* function.
   For varargs, takes some number of arguments. */
typedef void *(qt_userf_t)(void *pu);
typedef void *(qt_vuserf_t)(int arg0, ...);

/* For non-varargs, just call a client-supplied function,
   it does all startup and cleanup, and also calls the user's
   function. */
typedef void (qt_only_t)(void *pu, void *pt, qt_userf_t *userf);

/* For varargs, call `startup', then call the user's function,
   then call `cleanup'. */
typedef void (qt_startup_t)(void *pt);
typedef void (qt_cleanup_t)(void *pt, void *vuserf_return);


/* Internal helper for putting stuff on stack. */
#ifndef QT_SPUT
#define QT_SPUT(top, at, val)	\
    (((qt_word_t *)(top))[(at)] = (qt_word_t)(val))
#endif


/* Push arguments for the non-varargs case. */
#ifndef QT_ARGS

#ifndef QT_ARGS_MD
#define QT_ARGS_MD (0)
#endif

#ifndef QT_STKBASE
  #error "Need to know the machine-dependent stack allocation."
#endif

/* All things are put on the stack relative to the final value of
   the stack pointer. */
#ifdef QT_GROW_DOWN
#define QT_ADJ(sp)	(((char *)sp) - QT_STKBASE)
#else
#define QT_ADJ(sp)	(((char *)sp) + QT_STKBASE)
#endif

#define QT_ARGS(sp, pu, pt, userf, only) \
    (QT_ARGS_MD (QT_ADJ(sp)), \
     QT_SPUT (QT_ADJ(sp), QT_ONLY_INDEX, only), \
     QT_SPUT (QT_ADJ(sp), QT_USER_INDEX, userf), \
     QT_SPUT (QT_ADJ(sp), QT_ARGT_INDEX, pt), \
     QT_SPUT (QT_ADJ(sp), QT_ARGU_INDEX, pu), \
     ((qt_t *)QT_ADJ(sp)))

#endif


/* Push arguments for the varargs case.
   Has to be a function call because initialization is an expression
   and we need to loop to copy nbytes of stuff on to the stack.
   But that's probably OK, it's not terribly cheap, anyway. */

#ifdef QT_VARGS_DEFAULT
#ifndef QT_VARGS_MD0
#define QT_VARGS_MD0(sp, vasize)	(sp)
#endif
#ifndef QT_VARGS_MD1
#define QT_VARGS_MD1(sp)	do { ; } while (0)
#endif

#ifndef QT_VSTKBASE
  #error "Need base stack size for varargs functions."
#endif

/* Sometimes the stack pointer needs to munged a bit when storing
   the list of arguments. */
#ifndef QT_VARGS_ADJUST
#define QT_VARGS_ADJUST(sp)	(sp)
#endif

/* All things are put on the stack relative to the final value of
   the stack pointer. */
#ifdef QT_GROW_DOWN
#define QT_VADJ(sp)	(((char *)sp) - QT_VSTKBASE)
#else
#define QT_VADJ(sp)	(((char *)sp) + QT_VSTKBASE)
#endif

extern qt_t *qt_vargs (qt_t *sp, int nbytes, void *vargs,
		       void *pt, qt_startup_t *startup,
		       qt_vuserf_t *vuserf, qt_cleanup_t *cleanup);

#ifndef QT_VARGS
#define QT_VARGS(sp, nbytes, vargs, pt, startup, vuserf, cleanup) \
      (qt_vargs (sp, nbytes, vargs, pt, startup, vuserf, cleanup))
#endif

#endif


/* Save the state of the thread and call the helper function
   using the stack of the new thread. */
typedef void *(qt_helper_t)(qt_t *old, void *a0, void *a1);
typedef void *(qt_block_t)(qt_helper_t *helper, void *a0, void *a1,
			  qt_t *newthread);

/* Rearrange the parameters so that things passed to the helper
   function are already in the right argument registers. */
#ifndef QT_ABORT
extern qt_abort (qt_helper_t *h, void *a0, void *a1, qt_t *newthread);
/* The following does, technically, `return' a value, but the
   user had better not rely on it, since the function never
   returns. */ 
#define QT_ABORT(h, a0, a1, newthread) \
    do { qt_abort (h, a0, a1, newthread); } while (0)
#endif

#ifndef QT_BLOCK
extern void *qt_block (qt_helper_t *h, void *a0, void *a1,
		       qt_t *newthread);
#define QT_BLOCK(h, a0, a1, newthread) \
    (qt_block (h, a0, a1, newthread))
#endif

#ifndef QT_BLOCKI
extern void *qt_blocki (qt_helper_t *h, void *a0, void *a1,
			qt_t *newthread);
#define QT_BLOCKI(h, a0, a1, newthread) \
    (qt_blocki (h, a0, a1, newthread))
#endif

#ifdef __cplusplus
}		/* Match `extern "C" {' at top. */
#endif

#endif /* ndef QT_H */
