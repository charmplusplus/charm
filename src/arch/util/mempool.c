
/** 

Memory pool implementation. It is used in three places-
1 - For UGNI management of pinned memory
2 - Isomalloc allocation
3 - GPU manager pinned memory pool

Memory is allocated in terms of blocks from the OS and the user
is given back memory after rounding to nearest power of 2.

Written by Yanhua Sun 08-27-2011
Generalized by Gengbin Zheng  10/5/2011
Heavily modified by Nikhil Jain 11/28/2011
Extended to GPUs by Michael Robson 3/13/2017
*/

#define MEMPOOL_DEBUG   0
#define DEFAULT_BASE_POWER      6 // Smallest block of memory returned, 64 B = 2^6
#define DEFAULT_CUTOFF_POWER   30 // Largest  block of memory returned, 1 GB = 2^30

#include "converse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if CMK_HAS_MALLOC_H
#include <malloc.h>
#endif

#if CMK_C_INLINE
#define INLINE_KEYWORD inline static
#else
#define INLINE_KEYWORD static
#endif

#include "mempool.h"

INLINE_KEYWORD int which_pow2(size_t size, mempool_type* mptr)
{
  int i;
  for (i = 0; i < mptr->cutOffNum; i++) {
    if (size <= mptr->cutOffPoints[i]) {
      return i;
    }
  }
  return i;
}

//method to initialize the freelists of a newly allocated block
INLINE_KEYWORD void fillblock(mempool_type *mptr,block_header *block_head,size_t pool_size,int expansion)
{
  int         i,power;
  size_t      loc,left,prev;
  slot_header *head;
  void        *pool;

  block_head->freelists = calloc(mptr->cutOffNum, sizeof(size_t));

  pool = block_head;
  if(expansion) {
    left = pool_size-sizeof(block_header);
    loc = (char*)pool+sizeof(block_header)-(char*)mptr;
  } else {
    left = pool_size-sizeof(mempool_type);
    loc = (char*)pool+sizeof(mempool_type)-(char*)mptr;
  }
  power = which_pow2(left, mptr);
  if (left < mptr->cutOffPoints[power]) {
    power--;
  }
    
  if (power == mptr->cutOffNum) {
    CmiAbort("Mempool-requested slot is more than what mempool can provide as\
    one chunk, increase cutoffPwr in mempool\n");
  }

#if MEMPOOL_DEBUG
  CmiPrintf("Left is %d, Max power obtained is %d\n",left,power);
#endif

  for(i=power; i>=0; i--) {
    if (left >= mptr->cutOffPoints[i]) {
      block_head->freelists[i] = loc;
      loc += mptr->cutOffPoints[i];
      left -= mptr->cutOffPoints[i];
    }
  }

  prev = 0;
  for(i=power; i>=0; i--) {
    if(block_head->freelists[i]) {
      head = (slot_header*)((char*)mptr+block_head->freelists[i]);
      head->size = i;
      head->status = 1;
      head->block_ptr = block_head;
      head->prev = head->next = 0;
      head->gprev = prev;
      if(i!=power) {
        ((slot_header*)((char*)mptr+prev))->gnext = block_head->freelists[i];
      }
      prev = block_head->freelists[i];
    }
  }
  head->gnext = 0;
}

//method to check if a request can be met by this block
//if yes, alter the block free list appropiately 
int checkblock(mempool_type *mptr,block_header *current,int power)
{
  int         i,powiter;
  size_t      prev,loc,gnext;
  slot_header *head,*head_free,*head_move,*head_next;
  head_free = current->freelists[power]?(slot_header*)((char*)mptr+current->freelists[power]):NULL;

  //if the freelist of required size is empty, check if free
  //list of some larger size is non-empty and break a slot from it
  powiter = power+1;
  while (head_free == NULL && powiter < mptr->cutOffNum) {
    if(current->freelists[powiter]) {
      head_move = (slot_header*)((char*)mptr+current->freelists[powiter]);
      gnext = head_move->gnext;
      loc = current->freelists[powiter];
      current->freelists[powiter] = head_move->next;
      current->freelists[power] = loc;
      //we get 2 entries for smallest size required
      loc = loc + mptr->cutOffPoints[power];
      for(i=power+1; i<powiter; i++) { 
        loc = loc + mptr->cutOffPoints[i-1];
        current->freelists[i] = loc;
      }

      head_move->size = power;
      prev = current->freelists[power];
      head_move->next = prev + mptr->cutOffPoints[power];
      head = (slot_header*)((char*)head_move + mptr->cutOffPoints[power]);
      for(i=power; i<powiter; i++) {
        if(i!=power) {
          head = (slot_header*)((char*)head + mptr->cutOffPoints[i-1]);
        }
        head->size = i;
        head->status = 1;
        head->block_ptr = current;
        head->prev = head->next = 0;
        head->gprev = prev;
        ((slot_header*)((char*)mptr+prev))->gnext = (char*)head-(char*)mptr;
        if(i!=power) {
          prev = prev + mptr->cutOffPoints[i - 1];
        } else {
          prev = prev + mptr->cutOffPoints[i];
        }
      }
      ((slot_header*)((char*)head_move + mptr->cutOffPoints[power]))->prev =
      current->freelists[power];
      head->gnext = gnext;
      if(gnext!= 0) {
        ((slot_header*)((char*)mptr+gnext))->gprev = prev;
      }
      if(current->freelists[powiter]) {
        head_next = (slot_header*)((char*)mptr+current->freelists[powiter]);
        head_next->prev = 0;
      }
      head_free = (slot_header*)((char*)mptr+current->freelists[power]);
    }
    powiter++;
  }
  if(head_free == NULL) {
    return 0;
  } else {
    return 1;
  }
}

void removeblocks(mempool_type *mptr)
{
  block_header *current,*prev,*tofree,*tail;

  mempool_freeblock freefn;
  if(mptr == NULL) return;
  freefn = mptr->freeblockfn;
  tail = (block_header*)((char*)mptr+mptr->block_tail);
  current = prev = &(mptr->block_head);
  current = current->block_next?(block_header *)((char*)mptr+current->block_next):NULL;

  while(current != NULL) {
    if(current->used <= 0) {
      tofree = current;
      current = current->block_next?(block_header *)((char*)mptr+current->block_next):NULL;
      if(tail == tofree) {
        mptr->block_tail = tofree->block_prev;
      }
      prev->block_next = tofree->block_next;
      if(current != NULL) {
        current->block_prev = tofree->block_prev;
      }
      mptr->size -= tofree->size;
      freefn(tofree, tofree->mem_hndl);
      if(mptr->size < mptr->limit) return;
    } else {
      prev = current;
      current = current->block_next?(block_header *)((char*)mptr+current->block_next):NULL;
    }
  }
}

mempool_type *mempool_init(size_t pool_size, mempool_newblockfn allocfn,
    mempool_freeblock freefn, size_t limit)
{
  return mempool_init_all(pool_size, allocfn, freefn, limit, MEMPOOL_USELOCK,
      DEFAULT_CUTOFF_POWER, DEFAULT_BASE_POWER);
}

mempool_type *mempool_init_lock(size_t pool_size, mempool_newblockfn allocfn,
    mempool_freeblock freefn, size_t limit, mempool_lock_flag useLock)
{
  return mempool_init_all(pool_size, allocfn, freefn, limit, useLock,
      DEFAULT_CUTOFF_POWER, DEFAULT_BASE_POWER);
}

mempool_type *mempool_init_all(size_t pool_size, mempool_newblockfn allocfn,
    mempool_freeblock freefn, size_t limit, mempool_lock_flag useLock,
    int cutOffPwr, int basePwr)
{
  mempool_type *mptr;
  mem_handle_t  mem_hndl;
  int i;

  void *pool = allocfn(&pool_size, &mem_hndl, 0);
  mptr = (mempool_type*)pool;
  mptr->newblockfn = allocfn;
  mptr->freeblockfn = freefn;
  mptr->block_tail = 0;
  mptr->limit = limit;
  mptr->size = pool_size;
  mptr->useMempoolLock = useLock;
  mptr->mempoolLock = CmiCreateLock();
  mptr->block_head.mptr = (struct mempool_type *)pool;
  mptr->block_head.mem_hndl = mem_hndl;
  mptr->block_head.size = pool_size;
  mptr->block_head.used = 0;
  mptr->block_head.block_prev = 0;
  mptr->block_head.block_next = 0;
#if CMK_CONVERSE_UGNI
  mptr->block_head.msgs_in_send= 0;
  mptr->block_head.msgs_in_recv= 0;
#endif

  if (cutOffPwr < basePwr) {
    CmiAbort("cutOffPwr must be greater than or equal to basePwr\n");
  }
  mptr->cutOffNum = cutOffPwr - basePwr + 1;

  mptr->cutOffPoints = malloc(sizeof(size_t) * mptr->cutOffNum);
  size_t power = 1 << basePwr;
  for (i = 0; i < mptr->cutOffNum; i++) {
    mptr->cutOffPoints[i] = power;
    power <<= 1;
  }

  fillblock(mptr,&mptr->block_head,pool_size,0);
  return mptr;
}

void mempool_destroy(mempool_type *mptr)
{
  block_header *current,*tofree;

  mempool_freeblock freefn;
  if(mptr == NULL) return;
  freefn= mptr->freeblockfn;
  current = tofree = &(mptr->block_head);

  free(mptr->cutOffPoints);

  while(current != NULL) {
    tofree = current;
    current = current->block_next?(block_header *)((char*)mptr+current->block_next):NULL;
    freefn(tofree, tofree->mem_hndl);
  }
}

// append slot_header size before the real memory buffer
void*  mempool_malloc(mempool_type *mptr, size_t size, int expand)
{
    void          *pool;
    int           i;
    size_t        expand_size, bestfit_size;
    int           power; //closest power of cutoffpoint
    block_header  *current,*tail;
    slot_header   *head_free,*head_next;
    mem_handle_t  mem_hndl;

    if (mptr->useMempoolLock) {
      CmiLock(mptr->mempoolLock);
    }

    bestfit_size = size + sizeof(used_header);
    power = which_pow2(bestfit_size, mptr);
    if (power == mptr->cutOffNum) {
      CmiAbort("Mempool-requested slot is more than what mempool can provide as\
      one chunk, increase cutoffPwr in mempool\n");
    }
    bestfit_size = mptr->cutOffPoints[power];
#if MEMPOOL_DEBUG
    CmiPrintf("Request size is %d, power value is %d, size is %d\n", size,
        power, mptr->cutOffPoints[power]);
#endif

    head_free = NULL;
    current = &mptr->block_head;
    while(current != NULL) {
     if(checkblock(mptr,current,power)) {
        head_free = current->freelists[power]?(slot_header*)((char*)mptr+current->freelists[power]):NULL;
        break;
      } else {
        current = current->block_next?(block_header *)((char*)mptr+current->block_next):NULL;
      }
    }

    //no space in current blocks, get a new one
    if(head_free==NULL) {
      if (!expand) return NULL;

#if MEMPOOL_DEBUG
      CmiPrintf("Expanding size %lld limit %lld\n",mptr->size,mptr->limit);
#endif
      //free blocks which are not being used
      if((mptr->size > mptr->limit) && (mptr->limit > 0)) {
        removeblocks(mptr);
      }

      tail = (block_header*)((char*)mptr+mptr->block_tail);
      expand_size = 2*bestfit_size + sizeof(block_header); 
      pool = mptr->newblockfn(&expand_size, &mem_hndl, 1);
      if(pool==NULL) {
        CmiPrintf("Mempool-Did not get memory while expanding\n");
        return NULL;
      }
    
      mptr->size += expand_size;
      current = (block_header*)pool; 
      tail->block_next = ((char*)current-(char*)mptr);
      current->block_prev = mptr->block_tail;
      mptr->block_tail = tail->block_next;

      current->mptr = mptr;
      current->mem_hndl = mem_hndl;
      current->used = 0;
      current->size = expand_size;
      current->block_next = 0;
#if CMK_CONVERSE_UGNI
      current->msgs_in_send= 0;
      current->msgs_in_recv = 0;
#endif

      fillblock(mptr,current,expand_size,1);
      if(checkblock(mptr,current,power)) {
        head_free = current->freelists[power]?(slot_header*)((char*)mptr+current->freelists[power]):NULL;
      } else {
        CmiPrintf("Mempool-No free block after expansion, something is broken in mempool\n");
	return NULL;
      }
    }

    if(head_free!=NULL) {
      head_free->status = 0;
      current->freelists[power] = head_free->next;
      head_next = current->freelists[power]?(slot_header*)((char*)mptr+current->freelists[power]):NULL;
      if(head_next != NULL) {
        head_next->prev = 0;
      }

      head_free->block_ptr = current;
      current->used += power;

    if (mptr->useMempoolLock) {
      CmiUnlock(mptr->mempoolLock);
    }

      return (char*)head_free + sizeof(used_header);
    }
    
    CmiPrintf("Mempool-Reached a location which I should never have reached\n");
    return NULL;
}

void mempool_free_thread( void *ptr_free)
{
    slot_header *to_free;
    mempool_type *mptr;

    to_free = (slot_header *)((char*)ptr_free - sizeof(used_header));
    mptr = (mempool_type*)(((block_header*)(to_free->block_ptr))->mptr);
    CmiLock(mptr->mempoolLock);
    mempool_free(mptr,  ptr_free);
    CmiUnlock(mptr->mempoolLock);
}

void mempool_free(mempool_type *mptr, void *ptr_free)
{
    int           i,power;
    size_t        prev,loc,size,left;
    block_header  *block_head;
    slot_header   *to_free, *first, *current;
    slot_header   *used_next,*temp;

#if MEMPOOL_DEBUG
    CmiPrintf("Free request for %lld\n",
              ((char*)ptr_free - (char*)mptr - sizeof(used_header)));
#endif

    to_free = (slot_header *)((char*)ptr_free - sizeof(used_header));
    to_free->status = 1;
    block_head = to_free->block_ptr;
    block_head->used -= to_free->size;

    //find the neighborhood of to_free which is also free and
    //can be merged to get larger free slots 
    size = 0;
    current = to_free;
    while(current->status == 1) {
      size += mptr->cutOffPoints[current->size];
      first = current;
      current = current->gprev?(slot_header*)((char*)mptr+current->gprev):NULL;
      if(current == NULL)
        break;
    }

    size -= mptr->cutOffPoints[to_free->size];
    current = to_free;
    while(current->status == 1) {
      size += mptr->cutOffPoints[current->size];
      current = current->gnext?(slot_header*)((char*)mptr+current->gnext):NULL;
      if(current == NULL)
        break;
    }
    used_next = current;

    //remove the free slots in neighbor hood from their respective
    //free lists
    current = first;
    while(current!=used_next) {
      if(current!=to_free) {
        power = current->size;
        temp = current->prev?(slot_header*)((char*)mptr+current->prev):NULL;
        if(temp!=NULL) {
          temp->next = current->next;
        } else {
          block_head->freelists[power] = current->next;
        }
        temp = current->next?(slot_header*)((char*)mptr+current->next):NULL;
        if(temp!=NULL) {
          temp->prev = current->prev;
        }
      }
      current = current->gnext?(slot_header*)((char*)mptr+current->gnext):NULL;
    }

    //now create the new free slots of as large a size as possible
    power = which_pow2(size, mptr);
    if (size < mptr->cutOffPoints[power]) {
      power--;
    }
    left = size;

#if MEMPOOL_DEBUG
    if(CmiMyPe() == 0)
      printf("free was for %lld, merging for %lld, power %lld\n",to_free->size,size,power);
#endif

    loc = (char*)first - (char*)mptr;
    for(i=power; i>=0; i--) {
      if (left >= mptr->cutOffPoints[i]) {
        current = (slot_header*)((char*)mptr+loc);
        current->size = i;
        current->status = 1;
      	current->block_ptr = block_head;
        if(i!=power) {
          current->gprev = prev;
        }
        current->gnext = loc + mptr->cutOffPoints[i];
        current->prev = 0;
        if(block_head->freelists[i] == 0) {
          current->next = 0;
        } else {
          current->next = block_head->freelists[i];
          temp = (slot_header*)((char*)mptr+block_head->freelists[i]);
          temp->prev = loc;
        }
        block_head->freelists[i] = loc;
        prev = loc;
        loc += mptr->cutOffPoints[i];
        left -= mptr->cutOffPoints[i];
      }
    }
   if(used_next!=NULL) {
      used_next->gprev = (char*)current - (char*)mptr;
    } else {
      current->gnext = 0;
    }
#if MEMPOOL_DEBUG
    CmiPrintf("Free done\n");
#endif
}

#if CMK_CONVERSE_UGNI
inline void* getNextRegisteredPool(void *current)
{
    
}
#endif
