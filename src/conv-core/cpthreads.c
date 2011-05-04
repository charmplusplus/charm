/*
 * TO-DO:
 *
 * what about shared memory machines?
 *
 * what about the fact that posix threads programs exit when all
 * threads have completed?
 *
 * write errcode, errspan.  Figure out errno thing.
 *
 * there's obviously something I don't understand about cond... what's
 * the mutex for?
 *
 * how are we going to implement pthread_cond_timedwait?
 *
 * what about shared-memory locks?
 *
 * implement inheritance of processor-private data (in Cth)
 *
 */

#define CPTHREAD_IS_HERE
#define SUPPRESS_PTHREADS
#include "cpthreads.h"
#include <stdlib.h>
#include <errno.h>

/******************************************************************************
 *
 * Magic Numbers
 *
 * Each of the structures we use in this file has a magic number associated
 * with it for error-checking purposes.
 *
 *****************************************************************************/

#define PT_MAGIC    0x8173292a
#define ATTR_MAGIC  0x783a2004
#define KEY_MAGIC   0x99934315
#define FKEY_MAGIC  0x99934315
#define MATTR_MAGIC 0x12673434
#define CATTR_MAGIC 0xA865B812

#undef MUTEX_MAGIC
#undef COND_MAGIC
#define MUTEX_MAGIC 0x13237770
#define COND_MAGIC  0x99431664

/******************************************************************************
 *
 * The Thread Structure Definition.
 *
 *****************************************************************************/
 
typedef void *(*voidfn)();

struct Cpthread_s
{
  int magic;
  voidfn startfn;
  void *startarg1;
  void *startarg2;
  void *startarg3;
  int detached;
  void *joinstatus;
  Cpthread_cleanup_t cleanups;
  CthThread waiting;
  CthThread thread;
};

#define errcode(n) { Cpthread_errno=(n); return -1; }


/******************************************************************************
 *
 * POSIX Thread private data.
 *
 * Posix references thread_private data by keys.  For each key created, we
 * use CthRegister to allocate another 4 bytes of Cth thread-private space.  
 * The offset returned by CthRegister and the key's destructor (if any)
 * are stored in the POSIX thread-private key structure.  We keep a list
 * of all the active keys so that when a thread exits, we can execute
 * all the thread-private destructors.  Since we can't really destroy
 * a key (there's no CthUnRegister), we simply put ``destroyed'' keys onto
 * a list of inactive keys and reuse them later.
 *
 *****************************************************************************/

struct Cpthread_key_s
{
  int magic;
  int offset;
  void (*destructo)(void *);
  Cpthread_key_t next;
};

Cpthread_key_t keys_active = 0;
Cpthread_key_t keys_inactive = 0;

int Cpthread_key_create(Cpthread_key_t *keyp, void (*destructo)(void *))
{
  Cpthread_key_t key;
  key = keys_inactive;
  if (key) {
    keys_inactive = key->next;
  } else {
    key = (Cpthread_key_t)malloc(sizeof(struct Cpthread_key_s));
    _MEMCHECK(key);
    key->offset = CthRegister(sizeof(void *));
  }
  key->magic = KEY_MAGIC;
  key->destructo = destructo;
  key->next = keys_active;
  keys_active = key;
  *keyp = key;
  return 0;
}

int Cpthread_key_delete(Cpthread_key_t key)
{
  Cpthread_key_t active = keys_active;
  if (key->magic != KEY_MAGIC) errcode(EINVAL);
  if (active==key) {
    keys_active = key->next;
  } else {
    while (active) {
      if (active->next == key) {
	active->next = key->next;
	goto deleted;
      }
      active = active->next;
    }
    return -1;
  }
deleted:
  key->magic = FKEY_MAGIC;
  key->next = keys_inactive;
  keys_inactive = key;
  return 0;
}

int Cpthread_setspecific(Cpthread_key_t key, void *val)
{
  char *data;
  data = CthCpvAccess(CthData);
  if (key->magic != KEY_MAGIC) errcode(EINVAL);
  *((void **)(data+(key->offset))) = val;
  return 0;
}

void *Cpthread_getspecific(Cpthread_key_t key)
{
  char *data = CthCpvAccess(CthData);
  if (key->magic != KEY_MAGIC) return 0;
  return *((void **)(data+(key->offset)));
}

/******************************************************************************
 *
 * Cleanup routines.
 *
 * Every thread has a hook for cleanup routines that are to be automatically
 * called at exit-time.  These functions add cleanup-routines to a thread.
 *
 *****************************************************************************/

struct Cpthread_cleanup_s
{
  void (*routine)(void *);
  void *argument;
  Cpthread_cleanup_t next;
};

void Cpthread_cleanup_push(void (*routine)(void*), void *arg)
{
  Cpthread_t pt = CtvAccess(Cpthread_current);
  Cpthread_cleanup_t c =
    (Cpthread_cleanup_t)malloc(sizeof(struct Cpthread_cleanup_s));
  _MEMCHECK(c);
  c->routine = routine;
  c->argument = arg;
  c->next = pt->cleanups;
  pt->cleanups = c;
}

void Cpthread_cleanup_pop(int execute)
{
  Cpthread_t pt = CtvAccess(Cpthread_current);
  Cpthread_cleanup_t c = pt->cleanups;
  if (c) {
    pt->cleanups = c->next;
    if (execute) (c->routine)(c->argument);
    free(c);
  }
}

/******************************************************************************
 *
 * Thread Attributes
 *
 * Threads have two attributes set at creation time:
 *
 *    1. stack size.
 *    2. detached or not.
 *
 * These attributes must be put into an attribute structure before
 * calling the thread creation function.
 *
 *****************************************************************************/

int Cpthread_attr_init(Cpthread_attr_t *attr)
{
  attr->magic = ATTR_MAGIC;
  attr->detached = 0;
  attr->stacksize = 0;
  return 0;
}

int Cpthread_attr_destroy(Cpthread_attr_t *attr)
{
  if (attr->magic != ATTR_MAGIC) errcode(EINVAL);
  attr->magic = 0;
  return 0;
}

int Cpthread_attr_getstacksize(Cpthread_attr_t *attr, size_t *size)
{
  if (attr->magic != ATTR_MAGIC) errcode(EINVAL);
  *size = attr->stacksize;
  return 0;
}

int Cpthread_attr_setstacksize(Cpthread_attr_t *attr, size_t size)
{
  if (attr->magic != ATTR_MAGIC) errcode(EINVAL);
  attr->stacksize = size;
  return 0;
}

int Cpthread_attr_getdetachstate(Cpthread_attr_t *attr, int *state)
{
  if (attr->magic != ATTR_MAGIC) errcode(EINVAL);
  *state = attr->detached;
  return 0;
}

int Cpthread_attr_setdetachstate(Cpthread_attr_t *attr, int state)
{
  if (attr->magic != ATTR_MAGIC) errcode(EINVAL);
  attr->detached = state;
  return 0;
}

/******************************************************************************
 *
 * Thread primary operations: create, destroy, equal, self, detach, join
 *
 * Every thread is associated with a CthThread and a pthread_t (which are
 * separate from each other).  The pthread_t contains a field pointing to the
 * CthThread, and the CthThread has a thread-private variable ``Cpthread_current''
 * pointing to the pthread_t.
 *
 *****************************************************************************/

void Cpthread_top(Cpthread_t pt)
{
  Cpthread_key_t k; char *data; void *result; 

  data = CthCpvAccess(CthData);
  for (k=keys_active; k; k=k->next)
    *(void **)(data+(k->offset)) = 0;
  CtvAccess(Cpthread_errcode) = 0;
  CtvAccess(Cpthread_current) = pt;
  result = (pt->startfn)(pt->startarg1, pt->startarg2, pt->startarg3);
  Cpthread_exit(result);
}

int Cpthread_create3(Cpthread_t *thread, Cpthread_attr_t *attr,
		     voidfn fn, void *a1, void *a2, void *a3)
{
  Cpthread_t pt;
  if (attr->magic != ATTR_MAGIC) errcode(EINVAL);
  pt = (Cpthread_t)malloc(sizeof(struct Cpthread_s));
  _MEMCHECK(pt);
  pt->magic = PT_MAGIC;
  pt->startfn = fn;
  pt->startarg1 = a1;
  pt->startarg2 = a2;
  pt->startarg3 = a3;
  pt->detached = attr->detached;
  pt->joinstatus = 0;
  pt->cleanups = 0;
  pt->waiting = 0;
  pt->thread = CthCreate((CthVoidFn)Cpthread_top, (void *)pt, attr->stacksize);
  CthSetStrategyDefault(pt->thread);
  CthAwaken(pt->thread);
  *thread = pt;
  return 0;
}

int Cpthread_create(Cpthread_t *thread, Cpthread_attr_t *attr,
		     voidfn fn, void *arg)
{
  return Cpthread_create3(thread, attr, fn, arg, 0, 0);
}

void Cpthread_exit(void *status)
{
  Cpthread_t pt; Cpthread_cleanup_t c, cn; Cpthread_key_t k;
  void *priv; char *data; CthThread t;

  pt = CtvAccess(Cpthread_current);
  t = pt->thread;
  c = pt->cleanups;
  data = CthCpvAccess(CthData);

  /* perform all cleanup functions */
  while (c) {
    (c->routine)(c->argument);
    cn = c->next;
    free(c); c=cn;
  }
  /* execute destructors for thread-private data */
  k = keys_active;
  while (k) {
    if (k->destructo) {
      priv = *(void **)(data+(k->offset));
      if (priv) (k->destructo)(priv);
    }
    k=k->next;
  }
  /* handle the join-operation */
  if (pt->detached) {
    pt->magic = 0;
    free(pt);
  } else {
    pt->joinstatus = status;
    pt->thread = 0;
    if (pt->waiting) CthAwaken(pt->waiting);
  }
  CthFree(t);
  CthSuspend();
}

int Cpthread_equal(Cpthread_t t1, Cpthread_t t2)
{
  return (t1==t2);
}

int Cpthread_detach(Cpthread_t pt)
{
  if (pt->magic != PT_MAGIC) errcode(EINVAL);
  if (pt->thread==0) {
    pt->magic = 0;
    free(pt);
  } else {
    pt->detached = 1;
  }
  return 0;
}

int Cpthread_join(Cpthread_t pt, void **status)
{
  if (pt->magic != PT_MAGIC) errcode(EINVAL);
  if (pt->thread) {
    pt->waiting = CthSelf();
    CthSuspend();
  }
  *status = pt->joinstatus;
  free(pt);
  return 0;
}

int Cpthread_once(Cpthread_once_t *once, void (*fn)(void))
{
  int rank = CmiMyRank();
  if (rank>=32) {
    CmiPrintf("error: cpthreads current implementation limited to 32 PE's per node.\n");
    exit(1);
  }
  if (once->flag[rank]) return 0;
  once->flag[rank] = 1;
  fn();
  return 1;
}

/******************************************************************************
 *
 * Synchronization Structure: MUTEX
 *
 * Caution, when updating this code: the COND routines also access
 * the internals of the MUTEX structures. 
 *
 *****************************************************************************/

static void errspan()
{
  CmiPrintf("Error: Cpthreads sync primitives do not work across processor boundaries.\n");
  exit(1);
}

int Cpthread_mutexattr_init(Cpthread_mutexattr_t *mattr)
{
  mattr->magic = MATTR_MAGIC;
  return 0;
}

int Cpthread_mutexattr_destroy(Cpthread_mutexattr_t *mattr)
{
  if (mattr->magic != MATTR_MAGIC) errcode(EINVAL);
  mattr->magic = 0;
  return 0;
}

int Cpthread_mutexattr_getpshared(Cpthread_mutexattr_t *mattr, int *pshared)
{
  if (mattr->magic != MATTR_MAGIC) errcode(EINVAL);
  *pshared = mattr->pshared;
  return 0;
}

int Cpthread_mutexattr_setpshared(Cpthread_mutexattr_t *mattr, int pshared)
{
  if (mattr->magic != MATTR_MAGIC) errcode(EINVAL);
  mattr->pshared = pshared;
  return 0;
}

int Cpthread_mutex_init(Cpthread_mutex_t *mutex, Cpthread_mutexattr_t *mattr)
{
  if (mattr->magic != MATTR_MAGIC) errcode(EINVAL);
  mutex->magic = MUTEX_MAGIC;
  mutex->onpe = CmiMyPe();
  mutex->users = CdsFifo_Create();
  return 0;
}

int Cpthread_mutex_destroy(Cpthread_mutex_t *mutex)
{
  if (mutex->magic != MUTEX_MAGIC) errcode(EINVAL);
  if (mutex->onpe != CmiMyPe()) errspan();
  if (!CdsFifo_Empty(mutex->users)) errcode(EBUSY);
  mutex->magic = 0;
  CdsFifo_Destroy(mutex->users);
  return 0;
}

int Cpthread_mutex_lock(Cpthread_mutex_t *mutex)
{
  CthThread self = CthSelf();
  if (mutex->magic != MUTEX_MAGIC) errcode(EINVAL);
  if (mutex->onpe != CmiMyPe()) errspan();
  CdsFifo_Enqueue(mutex->users, self);
  if (CdsFifo_Peek(mutex->users) != self) CthSuspend();
  return 0;
}

int Cpthread_mutex_trylock(Cpthread_mutex_t *mutex)
{
  CthThread self = CthSelf();
  if (mutex->magic != MUTEX_MAGIC) errcode(EINVAL);
  if (mutex->onpe != CmiMyPe()) errspan();
  if (!CdsFifo_Empty(mutex->users)) errcode(EBUSY);
  CdsFifo_Enqueue(mutex->users, self);
  return 0;
}

int Cpthread_mutex_unlock(Cpthread_mutex_t *mutex)
{
  CthThread self = CthSelf();
  CthThread sleeper;
  if (mutex->magic != MUTEX_MAGIC) errcode(EINVAL);
  if (mutex->onpe != CmiMyPe()) errspan();
  if (CdsFifo_Peek(mutex->users) != self) errcode(EPERM);
  CdsFifo_Pop(mutex->users);
  sleeper = CdsFifo_Peek(mutex->users);
  if (sleeper) CthAwaken(sleeper);
  return 0;
}

/******************************************************************************
 *
 * Synchronization Structure: COND
 *
 *****************************************************************************/

int Cpthread_condattr_init(Cpthread_condattr_t *cattr)
{
  cattr->magic = CATTR_MAGIC;
  return 0;
}

int Cpthread_condattr_destroy(Cpthread_condattr_t *cattr)
{
  if (cattr->magic != CATTR_MAGIC) errcode(EINVAL);
  return 0;
}

int Cpthread_condattr_getpshared(Cpthread_condattr_t *cattr, int *pshared)
{
  if (cattr->magic != CATTR_MAGIC) errcode(EINVAL);
  *pshared = cattr->pshared;
  return 0;
}

int Cpthread_condattr_setpshared(Cpthread_condattr_t *cattr, int pshared)
{
  if (cattr->magic != CATTR_MAGIC) errcode(EINVAL);
  cattr->pshared = pshared;
  return 0;
}

int Cpthread_cond_init(Cpthread_cond_t *cond, Cpthread_condattr_t *cattr)
{
  if (cattr->magic != CATTR_MAGIC) errcode(EINVAL);
  cond->magic = COND_MAGIC;
  cond->onpe = CmiMyPe();
  cond->users = CdsFifo_Create();
  return 0;
}

int Cpthread_cond_destroy(Cpthread_cond_t *cond)
{
  if (cond->magic != COND_MAGIC) errcode(EINVAL);
  if (cond->onpe != CmiMyPe()) errspan();
  cond->magic = 0;
  CdsFifo_Destroy(cond->users);
  return 0;
}

int Cpthread_cond_wait(Cpthread_cond_t *cond, Cpthread_mutex_t *mutex)
{
  CthThread self = CthSelf();
  CthThread sleeper;

  if (cond->magic != COND_MAGIC) errcode(EINVAL);
  if (mutex->magic != MUTEX_MAGIC) errcode(EINVAL);
  if (cond->onpe != CmiMyPe()) errspan();
  if (mutex->onpe != CmiMyPe()) errspan();

  if (CdsFifo_Peek(mutex->users) != self) errcode(EPERM);
  CdsFifo_Pop(mutex->users);
  sleeper = CdsFifo_Peek(mutex->users);
  if (sleeper) CthAwaken(sleeper);
  CdsFifo_Enqueue(cond->users, self);
  CthSuspend();
  CdsFifo_Enqueue(mutex->users, self);
  if (CdsFifo_Peek(mutex->users) != self) CthSuspend();
  return 0;
}

int Cpthread_cond_signal(Cpthread_cond_t *cond)
{
  CthThread sleeper;
  if (cond->magic != COND_MAGIC) errcode(EINVAL);
  if (cond->onpe != CmiMyPe()) errspan();
  sleeper = CdsFifo_Dequeue(cond->users);
  if (sleeper) CthAwaken(sleeper);
  return 0;
}

int Cpthread_cond_broadcast(Cpthread_cond_t *cond)
{
  CthThread sleeper;
  if (cond->magic != COND_MAGIC) errcode(EINVAL);
  if (cond->onpe != CmiMyPe()) errspan();
  while (1) {
    sleeper = CdsFifo_Dequeue(cond->users);
    if (sleeper==0) break;
    CthAwaken(sleeper);
  }
  return 0;
}

/******************************************************************************
 *
 * Module initialization
 *
 *****************************************************************************/

typedef void (*mainfn)(int argc, char **argv);

int Cpthread_init()
{
  return 0;
}

void CpthreadModuleInit()
{
  CtvInitialize(Cpthread_t, Cpthread_current);
  CtvInitialize(int,        Cpthread_errcode);
}

void Cpthread_start_main(mainfn fn, int argc, char **argv)
{
  Cpthread_t pt;
  Cpthread_attr_t attrib;
  CmiIntPtr pargc = argc;
  if (CmiMyRank()==0) {
    Cpthread_attr_init(&attrib);
    Cpthread_attr_setdetachstate(&attrib, 1);
    Cpthread_create3(&pt, &attrib, (voidfn)fn, (void *)pargc, argv, 0);
  }
}
