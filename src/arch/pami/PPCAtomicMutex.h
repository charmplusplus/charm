
#ifndef __PPC_ATOMIC_MUTEX__
#define __PPC_ATOMIC_MUTEX__

#include <pthread.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#if CMK_PPC_ATOMIC_DEFAULT_IMPL
#include "default_ppcq.h"
#else
//define new ppc atomics in the pami instance directory
#include "ppc_atomicq_impl.h"
#endif

typedef struct _ppc_atomic_mutex_t
{
  volatile ppc_atomic_t     counter;
  volatile ppc_atomic_t     bound;
} PPCAtomicMutex;

PPCAtomicMutex *PPCAtomicMutexInit (void           * atomic_mem,
                                    size_t           atomic_size)
{
  //Verify counter array is 64-byte aligned
  assert( (((uintptr_t) atomic_mem) & (0x0F)) == 0 );
  assert (sizeof(PPCAtomicMutex) <= atomic_size);

  PPCAtomicMutex *mutex = (PPCAtomicMutex*) atomic_mem;
  PPC_AtomicStore(&mutex->counter, 0);
  PPC_AtomicStore(&mutex->bound, 1);

  return mutex;
}

/**
 *  \brief Try to acquire a mutex
 *  \param[in]   mutex pointer
 *  \return 0    Lock successfully acquired
 *  \return 1    Lock was not acquired
 */
static inline int PPCAtomicMutexTryAcquire (PPCAtomicMutex *mutex)
{
  size_t rc = PPC_AtomicLoadIncrementBounded(&mutex->counter);
  if (rc == CMI_PPC_ATOMIC_FAIL)
    return 1;

  PPC_AtomicReadFence();
  return rc;
}

/**
 *  \brief Acquire a mutex
 *  \param[in]   mutex pointer
 *  \return 0    Lock successfully acquired
 */
static inline void PPCAtomicMutexAcquire (PPCAtomicMutex *mutex)
{
  size_t rc = 0;
  do {
    rc = PPC_AtomicLoadIncrementBounded(&mutex->counter);
  } while (rc == CMI_PPC_ATOMIC_FAIL);

  PPC_AtomicReadFence();
}

/**
 *  \brief Release a mutex
 *  \param[in]   mutex pointer
 */
static inline void PPCAtomicMutexRelease(PPCAtomicMutex *mutex)
{
  //Flush outstanding loads/stores
  PPC_AtomicWriteFence();

  /* Release the lock */
  PPC_AtomicStore(&(mutex->counter), 0);
}


#endif
