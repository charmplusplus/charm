
/** 

Memory pool implementation. It is used in two places-
1 - For UGNI management of pinned memory
2 - Isomalloc allocation

Memory is allocated in terms of blocks from the OS and the user
is given back memory after rounding to nearest power of 2.

Written by Yanhua Sun 08-27-2011
Generalized by Gengbin Zheng  10/5/2011
Heavily modified by Nikhil Jain 11/28/2011
*/

#define MEMPOOL_DEBUG   0

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
int cutOffPoints[] = {64,128,256,512,1024,2048,4096, 8192,16384,32768,
                      65536,131072,262144,524288,1048576,2097152,4194304,
                      8388608,16777216,33554432,67108864,134217728,268435456,
                      536870912,1073741824};

INLINE_KEYWORD int which_pow2(size_t size)
{
  int i;
  for(i=0; i<cutOffNum; i++) {
    if(size <= cutOffPoints[i]) {
      return i;
    }
  }
  return i;
}

//method to initialize the freelists of a newly allocated block
INLINE_KEYWORD void fillblock(mempool_type *mptr,block_header *block_head,int pool_size,int expansion) 
{
  int         i,power;
  size_t      loc,left,prev;
  slot_header *head;
  void        *pool;

  for(i=0; i<cutOffNum;i++) {
    block_head->freelists[i] = 0;
  }

  pool = block_head;
  if(expansion) {
    left = pool_size-sizeof(block_header);
    loc = (char*)pool+sizeof(block_header)-(char*)mptr;
  } else {
    left = pool_size-sizeof(mempool_type);
    loc = (char*)pool+sizeof(mempool_type)-(char*)mptr;
  }
  power = which_pow2(left);
  if(left < cutOffPoints[power]) {
    power--;
  }
    
  if(power == cutOffNum) {
    CmiAbort("Mempool-requested slot is more than what mempool can provide as\
    one chunk, increase cutOffNum and cutoffPoints in mempool\n");
  }

#if MEMPOOL_DEBUG
  CmiPrintf("Left is %d, Max power obtained is %d\n",left,power);
#endif

  for(i=power; i>=0; i--) {
    if(left>=cutOffPoints[i]) {
      block_head->freelists[i] = loc;
      loc += cutOffPoints[i];
      left -= cutOffPoints[i];
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
  while(head_free==NULL && powiter<cutOffNum) {
    if(current->freelists[powiter]) {
      head_move = (slot_header*)((char*)mptr+current->freelists[powiter]);
      gnext = head_move->gnext;
      loc = current->freelists[powiter];
      current->freelists[powiter] = head_move->next;
      current->freelists[power] = loc;
      //we get 2 entries for smallest size required
      loc = loc+cutOffPoints[power];
      for(i=power+1; i<powiter; i++) { 
        loc = loc+cutOffPoints[i-1];
        current->freelists[i] = loc;
      }

      head_move->size = power;
      prev = current->freelists[power];
      head_move->next = prev+cutOffPoints[power]; 
      head = (slot_header*)((char*)head_move+cutOffPoints[power]);
      for(i=power; i<powiter; i++) {
        if(i!=power) {
          head = (slot_header*)((char*)head+cutOffPoints[i-1]);
        }
        head->size = i;
        head->status = 1;
        head->block_ptr = current;
        head->prev = head->next = 0;
        head->gprev = prev;
        ((slot_header*)((char*)mptr+prev))->gnext = (char*)head-(char*)mptr;
        if(i!=power) {
          prev = prev+cutOffPoints[i-1];
        } else {
          prev = prev+cutOffPoints[i];
        }
      }
      ((slot_header*)((char*)head_move+cutOffPoints[power]))->prev = 
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

mempool_type *mempool_init(size_t pool_size, mempool_newblockfn allocfn, mempool_freeblock freefn, size_t limit)
{
  int i,power;
  size_t end,left,prev,next;
  mempool_type *mptr;
  slot_header *head;
  mem_handle_t  mem_hndl;

  void *pool = allocfn(&pool_size, &mem_hndl, 0);
  mptr = (mempool_type*)pool;
  mptr->newblockfn = allocfn;
  mptr->freeblockfn = freefn;
  mptr->block_tail = 0;
  mptr->limit = limit;
  mptr->size = pool_size;
#if CMK_USE_MEMPOOL_ISOMALLOC || (CMK_SMP && CMK_CONVERSE_UGNI)
  mptr->mempoolLock = CmiCreateLock();
#endif
  mptr->block_head.mptr = pool;
  mptr->block_head.mem_hndl = mem_hndl;
  mptr->block_head.size = pool_size;
  mptr->block_head.used = 0;
  mptr->block_head.block_prev = 0;
  mptr->block_head.block_next = 0;
#if CMK_CONVERSE_UGNI
  mptr->block_head.msgs_in_send= 0;
  mptr->block_head.msgs_in_recv= 0;
#endif
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

  while(current != NULL) {
    tofree = current;
    current = current->block_next?(block_header *)((char*)mptr+current->block_next):NULL;
    freefn(tofree, tofree->mem_hndl);
  }
}

// append slot_header size before the real memory buffer
void*  mempool_malloc(mempool_type *mptr, int size, int expand)
{
    void          *pool;
    int           i;
    size_t        expand_size;
    int           power, bestfit_size; //closest power of cutoffpoint 
    block_header  *current,*tail;
    slot_header   *head_free,*head_next;
    mem_handle_t  mem_hndl;

#if CMK_USE_MEMPOOL_ISOMALLOC || (CMK_SMP && CMK_CONVERSE_UGNI)
    CmiLock(mptr->mempoolLock);
#endif

    bestfit_size = size + sizeof(used_header);
    power = which_pow2(bestfit_size);
    if(power == cutOffNum) {
      CmiAbort("Mempool-requested slot is more than what mempool can provide as\
      one chunk, increase cutOffNum and cutoffPoints in mempool\n");
    }
    bestfit_size = cutOffPoints[power];
#if MEMPOOL_DEBUG
    CmiPrintf("Request size is %d, power value is %d, size is %d\n",size,power,cutOffPoints[power]);
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
#if CMK_USE_MEMPOOL_ISOMALLOC || (CMK_SMP && CMK_CONVERSE_UGNI)
      CmiUnlock(mptr->mempoolLock);
#endif
      return (char*)head_free + sizeof(used_header);
    }
    
    CmiPrintf("Mempool-Reached a location which I should never have reached\n");
    return NULL;
}

#if CMK_USE_MEMPOOL_ISOMALLOC || (CMK_SMP && CMK_CONVERSE_UGNI)
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
#endif

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
      size += cutOffPoints[current->size];
      first = current;
      current = current->gprev?(slot_header*)((char*)mptr+current->gprev):NULL;
      if(current == NULL)
        break;
    }

    size -= cutOffPoints[to_free->size];
    current = to_free;
    while(current->status == 1) {
      size += cutOffPoints[current->size];
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
    power = which_pow2(size);
    if(size < cutOffPoints[power]) {
      power--;
    }
    left = size;

#if MEMPOOL_DEBUG
    if(CmiMyPe() == 0)
      printf("free was for %lld, merging for %lld, power %lld\n",to_free->size,size,power);
#endif

    loc = (char*)first - (char*)mptr;
    for(i=power; i>=0; i--) {
      if(left>=cutOffPoints[i]) {
        current = (slot_header*)((char*)mptr+loc);
        current->size = i;
        current->status = 1;
      	current->block_ptr = block_head;
        if(i!=power) {
          current->gprev = prev;
        }
        current->gnext = loc + cutOffPoints[i];
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
        loc += cutOffPoints[i];
        left -= cutOffPoints[i];
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
