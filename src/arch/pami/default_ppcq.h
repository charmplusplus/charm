
#ifndef  __DEFAULT_PPCQ_H__
#define  __DEFAULT_PPCQ_H__

#include "pami.h"

/////////////////////////////////////////////////////
// \brief Basic atomic operations should to defined
// PPC_AtomicStore : store a value to the atomic counter
// PPC_AtomicLoadIncrementBounded : bounded increment
// PPC_AtomicWriteFence : a producer side write fence
// PPC_AtomicReadFence  : consumer side read fence
// PPC_AtomicCounterAllocate : allocate atomic counters
/////////////////////////////////////////////////////

#define CMI_PPC_ATOMIC_FAIL  0x8000000000000000UL

typedef uint64_t ppc_atomic_type_t;

typedef struct _ppc_atomic_t {
  volatile uint64_t   val;
  char                _pad[56];
} ppc_atomic_t;

#define PPC_AQVal(x) ((x).val)

static inline void PPC_AtomicCounterAllocate (void **atomic_mem,
                                              size_t  atomic_memsize)
{
  posix_memalign(atomic_mem, 64, atomic_memsize);
}

// Load Reserved: 64bit atom
static inline ppc_atomic_type_t PPC_AtomicLoadReserved ( volatile ppc_atomic_t *ptr )
{
  ppc_atomic_type_t val;
  __asm__ __volatile__ ("ldarx %[val],0,%[ptr]"
                        : [val] "=r" (val)
                        : [ptr] "r" (&ptr->val)
                        : "cc");

  return( val );
}

static inline int PPC_AtomicStoreConditional( volatile ppc_atomic_t *ptr, ppc_atomic_type_t val )
{
  register int rc = 1; // assume success
  __asm__ __volatile__ ("stdcx. %[val],0,%[ptr];\n"
                        "beq 1f;\n"
                        "li %[rc], 0;\n"
                        "1: ;\n"
                        : [rc] "=r" (rc)
                        : [ptr] "r" (&ptr->val), [val] "r" (val), "0" (rc)
                        : "cc", "memory");
  return( rc );
}

static inline ppc_atomic_type_t PPC_AtomicLoadIncrementBounded (volatile ppc_atomic_t *counter)
{
  register ppc_atomic_type_t old_val, tmp_val, bound;
  bound = counter[1].val;
  do
  {
    old_val = PPC_AtomicLoadReserved( counter );
    tmp_val = old_val + 1;

    if (tmp_val > bound)
      return CMI_PPC_ATOMIC_FAIL;
  }
  while ( !PPC_AtomicStoreConditional( counter, tmp_val ) );

  return( old_val );
}

static inline void PPC_AtomicStore(volatile ppc_atomic_t *counter, ppc_atomic_type_t val)
{
  //Counter perpetually increments, so stale value is always smaller
  //__asm__ __volatile__ ("lwsync":::"memory");
  counter->val = val;
}

static inline void PPC_AtomicReadFence()
{
#if !CMK_BLUEGENEQ  //full memory barrier executed on Producer
  __asm__ __volatile__ ("isync":::"memory");
#endif
}

static inline void PPC_AtomicWriteFence()
{
#if CMK_BLUEGENEQ //execute full memory barrier
  __asm__ __volatile__ ("sync":::"memory");
#else
  __asm__ __volatile__ ("lwsync":::"memory");
#endif
}

#endif
