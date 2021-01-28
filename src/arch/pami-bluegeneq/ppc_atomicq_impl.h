
#ifndef __L2_ATOMIC_PPCQ_H__
#define __L2_ATOMIC_PPCQ_H__

#include "spi/include/l2/atomic.h"
#include "spi/include/l1p/flush.h"
#include "pami.h"

typedef pami_result_t (*pamix_proc_memalign_fn) (void**, size_t, size_t, const char*);

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
typedef uint64_t ppc_atomic_t;

#define PPC_AQVal(x) x

static inline void PPC_AtomicCounterAllocate (void **atomic_mem,
                                              size_t  atomic_memsize)
{
  pami_extension_t l2;
  pamix_proc_memalign_fn PAMIX_L2_proc_memalign;
  size_t size = atomic_memsize;
  pami_result_t rc = PAMI_SUCCESS;

  rc = PAMI_Extension_open(NULL, "EXT_bgq_l2atomic", &l2);
  CmiAssert (rc == 0);
  PAMIX_L2_proc_memalign = (pamix_proc_memalign_fn)PAMI_Extension_symbol(l2, "proc_memalign");
  rc = PAMIX_L2_proc_memalign(atomic_mem, 64, size, NULL);
  CmiAssert (rc == 0);
}

#define PPC_AtomicLoadIncrementBounded(counter) L2_AtomicLoadIncrementBounded(counter);

#define PPC_AtomicStore(counter, val) L2_AtomicStore(counter, val)

#define PPC_AtomicReadFence()     ppc_msync()

#define PPC_AtomicWriteFence()    L1P_FlushRequests()

#endif
