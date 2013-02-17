
#ifndef __L2_ATOMIC_MUTEX__
#define __L2_ATOMIC_MUTEX__

#include <pthread.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include "spi/include/l2/atomic.h"
#include "spi/include/l1p/flush.h"

#define L2_ATOMIC_MUTEX_FAIL        0x8000000000000000UL

typedef struct
{
  volatile uint64_t     counter;
  volatile uint64_t     bound;
} L2AtomicMutex;

L2AtomicMutex *L2AtomicMutexInit (void           * l2mem, 
				  size_t           l2memsize)
{
  //Verify counter array is 64-byte aligned 
  assert( (((uintptr_t) l2mem) & (0x0F)) == 0 );  
  assert (sizeof(L2AtomicMutex) <= l2memsize);

  L2AtomicMutex *mutex = (L2AtomicMutex*)l2mem;  
  L2_AtomicStore(&mutex->counter, 0);
  L2_AtomicStore(&mutex->bound, 1);
  
  return mutex;
}

/**
 *  \brief Try to acquire a mutex 
 *  \param[in]   mutex pointer
 *  \return 0    Lock successfully acquired
 *  \return 1    Lock was not acquired
 */
static inline int L2AtomicMutexTryAcquire (L2AtomicMutex *mutex)
{
  size_t rc = L2_AtomicLoadIncrementBounded(&mutex->counter);
  return (rc == L2_ATOMIC_MUTEX_FAIL) ? (1) : (0);
}

/**
 *  \brief Acquire a mutex 
 *  \param[in]   mutex pointer
 *  \return 0    Lock successfully acquired
 */
static inline void L2AtomicMutexAcquire (L2AtomicMutex *mutex)
{
  size_t rc = 0;
  do {
    rc = L2_AtomicLoadIncrementBounded(&mutex->counter);
  } while (rc == L2_ATOMIC_MUTEX_FAIL);
}

/**
 *  \brief Release a mutex 
 *  \param[in]   mutex pointer 
 *  \return 0    Lock successfully released
 *  \return 1    Fail
 */
static inline void L2AtomicMutexRelease(L2AtomicMutex *mutex)
{
  //Flush outstanding loads/stores
  ppc_msync();
  
  /* Release the lock */
  L2_AtomicStore(&(mutex->counter), 0);  
}


#endif
