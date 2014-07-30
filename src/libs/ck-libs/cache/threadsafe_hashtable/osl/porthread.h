/**
 * PorThread:
 *  Portable, trivial threading library.
 * Orion Sky Lawlor, olawlor@acm.org, 2003/4/2
 */
#ifndef __OSL_PORTHREAD_H
#define __OSL_PORTHREAD_H

/**
 * This is the routine executed in the thread.
 */
typedef void (*porthread_fn_t)(void *arg);

/** This is a handle to a running thread */
typedef void *porthread_t;

/**
 * Calls fn(arg) from within a new kernel thread.
 */
porthread_t porthread_create(porthread_fn_t fn,void *arg);

/** Wait until this thread has finished running. */
void porthread_wait(porthread_t p);


/**
 * Suspend the current thread for up to 
 *  this many milliseconds, letting other threads
 *  or processes run.
 */
void porthread_yield(int msec);

/**************** Locks ***************
	From Hovik Melikyan's http://www.melikyan.com/ptypes/ 
	(pasync.h)
*/
#ifdef _WIN32 /* Windows implementation */
#include <windows.h>

class porlock
{
protected:
	CRITICAL_SECTION critsec;
public:
	inline porlock()	{ InitializeCriticalSection(&critsec); }
	inline ~porlock()	{ DeleteCriticalSection(&critsec); }
	inline void lock() volatile { EnterCriticalSection(&critsec); }
        inline void unlock() volatile { LeaveCriticalSection(&critsec); }
};


#else /* Portable UNIX pthread version */
#include <pthread.h>

class porlock
{
protected:
	pthread_mutex_t mtx;
public:
	inline porlock()	{ pthread_mutex_init(&mtx, 0); }
	inline ~porlock()	{ pthread_mutex_destroy(&mtx); }
	inline void lock() volatile { pthread_mutex_lock(const_cast<pthread_mutex_t *>(&mtx)); }
        inline void unlock() { pthread_mutex_unlock(&mtx); }
};

#endif

/**
  C++ "scoped" lock.  Locks the lock on creation,
  unlocks the lock on deletion, which is guaranteed
  to happen when the lock goes out of scope.
  Just declare the lock, and the locking and unlocking
  happen automatically.
*/
class porlock_scoped {
	porlock *p;
public:
	inline porlock_scoped(porlock *p_) :p(p_) {p->lock();}
	inline ~porlock_scoped() {p->unlock();}
};


#endif
