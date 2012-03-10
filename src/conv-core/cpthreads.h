
#ifndef CPTHREADS_H
#define CPTHREADS_H

#include <converse.h>
/* for size_t */
#include <sys/types.h>

#define CPTHREAD_THREADS_MAX     1000000000
#define CPTHREAD_KEYS_MAX        1000000000
#define CPTHREAD_STACK_MIN       32768
#define CPTHREAD_CREATE_DETACHED 1
#define CPTHREAD_CREATE_JOINABLE 0

struct Cpthread_attr_s
{
  int magic;
  int detached;
  int stacksize;
};

struct Cpthread_mutexattr_s
{
  int magic;
  int pshared;
};

struct Cpthread_mutex_s
{
  int magic;
  int onpe;
  void *users;
};

struct Cpthread_condattr_s
{
  int magic;
  int pshared;
};

struct Cpthread_cond_s
{
  int magic;
  int onpe;
  void *users;
};

typedef struct { int flag[32]; } Cpthread_once_t;

#define CPTHREAD_ONCE_INIT {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}

typedef struct Cpthread_attr_s      Cpthread_attr_t;
typedef struct Cpthread_key_s      *Cpthread_key_t;
typedef struct Cpthread_cleanup_s  *Cpthread_cleanup_t;
typedef struct Cpthread_mutexattr_s Cpthread_mutexattr_t;
typedef struct Cpthread_condattr_s  Cpthread_condattr_t;
typedef struct Cpthread_mutex_s     Cpthread_mutex_t;
typedef struct Cpthread_cond_s      Cpthread_cond_t;
typedef struct Cpthread_s          *Cpthread_t;

#ifdef CPTHREAD_IS_HERE
CtvDeclare(Cpthread_t,  Cpthread_current);
CtvDeclare(int,         Cpthread_errcode);
#else
CtvExtern(Cpthread_t,  Cpthread_current);
CtvExtern(int,         Cpthread_errcode);
#endif

#define Cpthread_self() (CtvAccess(Cpthread_current))
#define Cpthread_errno (CtvAccess(Cpthread_errcode))

int   Cpthread_attr_init(Cpthread_attr_t *attr);
int   Cpthread_attr_destroy(Cpthread_attr_t *attr);
int   Cpthread_attr_getstacksize(Cpthread_attr_t *attr, size_t *size);
int   Cpthread_attr_setstacksize(Cpthread_attr_t *attr, size_t size);
int   Cpthread_attr_getdetachstate(Cpthread_attr_t *attr, int *state);
int   Cpthread_attr_setdetachstate(Cpthread_attr_t *attr, int state);
int   Cpthread_key_create(Cpthread_key_t *keyp, void (*destructo)(void *));
int   Cpthread_key_delete(Cpthread_key_t key);
int   Cpthread_setspecific(Cpthread_key_t key, void *val);
void *Cpthread_getspecific(Cpthread_key_t key);
void  Cpthread_cleanup_push(void (*routine)(void*), void *arg);
void  Cpthread_cleanup_pop(int execute);
void  Cpthread_exit(void *status);
void  Cpthread_top(Cpthread_t pt);
int   Cpthread_create(Cpthread_t *thread, Cpthread_attr_t *attr,
		      void *(*fn)(void *), void *arg);
int   Cpthread_equal(Cpthread_t t1, Cpthread_t t2);
int   Cpthread_detach(Cpthread_t pt);
int   Cpthread_join(Cpthread_t pt, void **status);
int   Cpthread_mutexattr_init(Cpthread_mutexattr_t *mattr);
int   Cpthread_mutexattr_destroy(Cpthread_mutexattr_t *mattr);
int   Cpthread_mutexattr_getpshared(Cpthread_mutexattr_t *mattr,int *pshared);
int   Cpthread_mutexattr_setpshared(Cpthread_mutexattr_t *mattr,int  pshared);
int   Cpthread_mutex_init(Cpthread_mutex_t *mutex,Cpthread_mutexattr_t *mattr);
int   Cpthread_mutex_destroy(Cpthread_mutex_t *mutex);
int   Cpthread_mutex_lock(Cpthread_mutex_t *mutex);
int   Cpthread_mutex_trylock(Cpthread_mutex_t *mutex);
int   Cpthread_mutex_unlock(Cpthread_mutex_t *mutex);
int   Cpthread_condattr_init(Cpthread_condattr_t *cattr);
int   Cpthread_condattr_destroy(Cpthread_condattr_t *cattr);
int   Cpthread_condattr_getpshared(Cpthread_condattr_t *cattr, int *pshared);
int   Cpthread_condattr_setpshared(Cpthread_condattr_t *cattr, int pshared);
int   Cpthread_cond_init(Cpthread_cond_t *cond, Cpthread_condattr_t *cattr);
int   Cpthread_cond_destroy(Cpthread_cond_t *cond);
int   Cpthread_cond_wait(Cpthread_cond_t *cond, Cpthread_mutex_t *mutex);
int   Cpthread_cond_signal(Cpthread_cond_t *cond);
int   Cpthread_cond_broadcast(Cpthread_cond_t *cond);
int   Cpthread_once(Cpthread_once_t *once, void (*fn)(void));

int Cpthread_init();

void Cpthread_initialize();
void Cpthread_start_main(CmiStartFn fn, int argc, char **argv);

#define Cpthread_yield() (CthYield())

#ifndef SUPPRESS_PTHREADS

#define _POSIX_THREADS
#define _POSIX_THREAD_ATTR_STACKSIZE
/* #define _POSIX_THREAD_ATTR_STACKADDR */
/* #define _POSIX_THREAD_PRIORITY_SCHEDULING */
/* #define _POSIX_THREAD_PRIO_INHERIT */
/* #define _POSIX_THREAD_PRIO_PROTECT */
/* #define _POSIX_THREAD_PROCESS_SHARED */

#define PTHREAD_THREADS_MAX		CPTHREAD_THREADS_MAX
#define PTHREAD_KEYS_MAX		CPTHREAD_KEYS_MAX
#define PTHREAD_STACK_MIN		CPTHREAD_STACK_MIN
#define PTHREAD_CREATE_DETACHED		CPTHREAD_CREATE_DETACHED
#define PTHREAD_CREATE_JOINABLE		CPTHREAD_CREATE_JOINABLE

#define PTHREAD_ONCE_INIT               CPTHREAD_ONCE_INIT

#define pthread_once_t                  Cpthread_once_t
#define pthread_attr_t                  Cpthread_attr_t
#define pthread_key_t                   Cpthread_key_t
#define pthread_cleanup_t               Cpthread_cleanup_t
#define pthread_mutexattr_t             Cpthread_mutexattr_t
#define pthread_condattr_t              Cpthread_condattr_t
#define pthread_mutex_t                 Cpthread_mutex_t
#define pthread_cond_t                  Cpthread_cond_t
#define pthread_t                       Cpthread_t
#define pthread_attr_init               Cpthread_attr_init
#define pthread_attr_destroy            Cpthread_attr_destroy
#define pthread_attr_getstacksize       Cpthread_attr_getstacksize
#define pthread_attr_setstacksize       Cpthread_attr_setstacksize
#define pthread_attr_getdetachstate     Cpthread_attr_getdetachstate
#define pthread_attr_setdetachstate     Cpthread_attr_setdetachstate
#define pthread_key_create              Cpthread_key_create
#define pthread_key_delete              Cpthread_key_delete
#define pthread_setspecific             Cpthread_setspecific
#define pthread_getspecific             Cpthread_getspecific
#define pthread_cleanup_push            Cpthread_cleanup_push
#define pthread_cleanup_pop             Cpthread_cleanup_pop
#define pthread_exit                    Cpthread_exit
#define pthread_top                     Cpthread_top
#define pthread_create                  Cpthread_create
#define pthread_equal                   Cpthread_equal
#define pthread_self                    Cpthread_self
#define pthread_detach                  Cpthread_detach
#define pthread_join                    Cpthread_join
#define pthread_mutexattr_init          Cpthread_mutexattr_init
#define pthread_mutexattr_destroy       Cpthread_mutexattr_destroy
#define pthread_mutexattr_getpshared    Cpthread_mutexattr_getpshared
#define pthread_mutexattr_setpshared    Cpthread_mutexattr_setpshared
#define pthread_mutex_init              Cpthread_mutex_init
#define pthread_mutex_destroy           Cpthread_mutex_destroy
#define pthread_mutex_lock              Cpthread_mutex_lock
#define pthread_mutex_trylock           Cpthread_mutex_trylock
#define pthread_mutex_unlock            Cpthread_mutex_unlock
#define pthread_condattr_init           Cpthread_condattr_init
#define pthread_condattr_destroy        Cpthread_condattr_destroy
#define pthread_condattr_getpshared     Cpthread_condattr_getpshared
#define pthread_condattr_setpshared     Cpthread_condattr_setpshared
#define pthread_cond_init               Cpthread_cond_init
#define pthread_cond_destroy            Cpthread_cond_destroy
#define pthread_cond_wait               Cpthread_cond_wait
#define pthread_cond_signal             Cpthread_cond_signal
#define pthread_cond_broadcast          Cpthread_cond_broadcast
#define pthread_once                    Cpthread_once
#define pthread_init                    Cpthread_init

#define pthread_yield                   (CthYield())

#endif /* SUPPRESS_PTHREADS */

#endif /* CPTHREAD_H */

