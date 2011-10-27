
/** 

Memory pool implementation , It is only good for Charm++ usage. The first 64 bytes provides additional information. sizeof(int)- size of this block(free or allocated), next mem_handle_t, then void** point to the next available block. 

Written by Yanhua Sun 08-27-2011
Generalized by Gengbin Zheng  10/5/2011

*/

#define MEMPOOL_DEBUG   0

#define POOLS_NUM       2
#define MAX_INT        2147483647

#include "converse.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if CMK_HAS_MALLOC_H
#include <malloc.h>
#endif

#include "mempool.h"


mempool_type *mempool_init(size_t pool_size, mempool_newblockfn allocfn, mempool_freeblock freefn)
{
    mempool_type *mptr;
    mempool_header *header;
    mem_handle_t  mem_hndl;

    void *pool = allocfn(&pool_size, &mem_hndl, 0);
    mptr = (mempool_type*)pool;
    mptr->newblockfn = allocfn;
    mptr->freeblockfn = freefn;
#if CMK_SMP
    mptr->mempoolLock = CmiCreateLock();
#endif
    mptr->mempools_head.mempool_ptr = pool;
    mptr->mempools_head.mem_hndl = mem_hndl;
    mptr->mempools_head.size = pool_size;
    mptr->mempools_head.memblock_next = 0;
    header = (mempool_header *) ((char*)pool+sizeof(mempool_type));
    mptr->freelist_head = sizeof(mempool_type);
    mptr->memblock_tail = 0;
#if MEMPOOL_DEBUG
    printf("[%d] pool: %p  free: %p\n", myrank, pool, header);
#endif
    header->size = pool_size-sizeof(mempool_type)-sizeof(mempool_header);
    header->mem_hndl = mem_hndl;
    header->next_free = 0;
    return mptr;
}

void mempool_destroy(mempool_type *mptr)
{
    mempool_block *current, *mempools_head;
    mempool_freeblock   freefn = mptr->freeblockfn;

    current = mempools_head = &(mptr->mempools_head);

    while(mempools_head!= NULL)
    {
#if MEMPOOL_DEBUG
        printf("[%d] free mempool:%p\n", CmiMyPe(), mempools_head->mempool_ptr);
#endif
        current=mempools_head;
        mempools_head = mempools_head->memblock_next?(mempool_block *)((char*)mptr+mempools_head->memblock_next):NULL;
        freefn(current->mempool_ptr, current->mem_hndl);
    }
}

// append size before the real memory buffer
void*  mempool_malloc(mempool_type *mptr, int size, int expand)
{
    int     bestfit_size = MAX_INT; //most close size 
    size_t    *freelist_head = &mptr->freelist_head;
    
#if CMK_SMP
    CmiLock(mptr->mempoolLock);
#endif
    mempool_header    *freelist_head_ptr = mptr->freelist_head?(mempool_header*)((char*)mptr+mptr->freelist_head):NULL;
    mempool_header    *current = freelist_head_ptr;
    mempool_header    *previous = NULL;
    mempool_header    *bestfit = NULL;
    mempool_header    *bestfit_previous = NULL;
    mempool_block     *mempools_head = &(mptr->mempools_head);

#if  MEMPOOL_DEBUG
    CmiPrintf("[%d] request malloc from pool: %p  free_head: %p %d for size %d, \n", CmiMyPe(), mptr, freelist_head_ptr, mptr->freelist_head, size);
#endif

    size += sizeof(mempool_header);

#if 1
    while(current!= NULL)     /* best fit */
    {
#if  MEMPOOL_DEBUG
        CmiPrintf("[%d] current=%p size:%d \n", CmiMyPe(), current, current->size);
#endif
        if(current->size >= size && current->size < bestfit_size)
        {
            bestfit_size = current->size;
            bestfit = current;
            bestfit_previous = previous;
        }
        previous = current;
        current = current->next_free?(mempool_header*)((char*)mptr + current->next_free):NULL;
    }
#else
    while(current!= NULL)             /*  first fit */
    {
#if  MEMPOOL_DEBUG
        CmiPrintf("[%d] current=%p size:%d ->%p \n", CmiMyPe(), current, current->size, (char*)current+current->size);
#endif
        CmiAssert(current->size != 0);
        if(current->size >= size)
        {
            bestfit_size = current->size;
            bestfit = current;
            bestfit_previous = previous;
            break;
        }
        previous = current;
        current = current->next_free?(mempool_header*)((char*)mptr + current->next_free):NULL;
    }
#endif

    if(bestfit == NULL)
    {
        void *pool;
        mempool_block   *expand_pool, *memblock_tail;
        size_t   expand_size;
        mem_handle_t  mem_hndl;

        if (!expand) return NULL;

         /* set minimum size, newblockfn checks against the default size */
        expand_size = 2*size; 
        pool = mptr->newblockfn(&expand_size, &mem_hndl, 1);
        expand_pool = (mempool_block*)pool;
        expand_pool->mempool_ptr = pool;
        expand_pool->mem_hndl = mem_hndl;
        expand_pool->size = expand_size;
        expand_pool->memblock_next = 0;
#if MEMPOOL_DEBUG
        printf("[%d] No memory has such free empty chunck of %d. expanding %p with new size %ld\n", CmiMyPe(), size, expand_pool->mempool_ptr, expand_size);
#endif
         /* insert new block to memblock tail */
        memblock_tail = (mempool_block*)((char*)mptr + mptr->memblock_tail);
        memblock_tail->memblock_next = mptr->memblock_tail = (char*)expand_pool - (char*)mptr;

        bestfit = (mempool_header*)((char*)expand_pool->mempool_ptr + sizeof(mempool_block));
        bestfit->size = expand_size-sizeof(mempool_block);
        bestfit->mem_hndl = expand_pool->mem_hndl;
        bestfit->next_free = 0;
        bestfit_size = expand_size-sizeof(mempool_block);
#if 1
         /* insert bestfit to the sorted free list */
        previous = NULL;
        current = freelist_head_ptr;
        while (current) 
        {
            if (current > bestfit) break;
            previous = current;
            current = current->next_free?(mempool_header*)((char *)mptr + current->next_free):NULL;
        };
        bestfit->next_free = current!=NULL? (char*)current-(char*)mptr:0;
#else
        CmiAssert(bestfit > previous);
        previous->next_free = (char*)bestfit-(char*)mptr;
#endif
        bestfit_previous = previous;
        if (previous == NULL) {
           *freelist_head = (char*)bestfit - (char*)mptr;
           freelist_head_ptr =  bestfit;
        }
        else
           previous->next_free = (char*)bestfit-(char*)mptr;
    }

    bestfit->size = size;
    if(bestfit_size > size + sizeof(mempool_header)) //deduct this entry 
    {
        mempool_header *ptr = (mempool_header *)((char*)bestfit + size);
        ptr->size = bestfit_size - size;
        ptr->mem_hndl = bestfit->mem_hndl;
        ptr->next_free = bestfit->next_free;
        if(bestfit == freelist_head_ptr)
           *freelist_head = (char*)ptr - (char*)mptr;
        if(bestfit_previous != NULL)
           bestfit_previous->next_free = (char*)ptr - (char*)mptr;
    }
    else {  
          //delete this free entry
        if (bestfit_size > size) {
           bestfit->size = bestfit_size;
        }
        if(bestfit == freelist_head_ptr)
            *freelist_head = freelist_head_ptr->next_free;
        else
            bestfit_previous->next_free = bestfit->next_free;
    }
#if  MEMPOOL_DEBUG
    printf("[%d] ++MALLOC served: %d, ptr:%p\n", CmiMyPe(), size, bestfit);
printf("[%d] freelist_head in malloc  offset:%d free_head: %ld %ld %d %d\n", myrank, (char*)bestfit-(char*)mptr, *freelist_head, ((mempool_header*)((char*)mptr+*freelist_head))->next_free, bestfit_size, size);
#endif
    CmiAssert(*freelist_head >= 0);
#if CMK_SMP
    CmiUnlock(mptr->mempoolLock);
#endif
#if CMK_SMP
    ((mempool_header *)bestfit)->pool_addr = mptr;
#endif
    return (char*)bestfit + sizeof(mempool_header);
}

#if CMK_SMP
void mempool_free_thread( void *ptr_free)
{
    mempool_header *to_free;
    mempool_type *mptr;

    to_free = (mempool_header *)((char*)ptr_free - sizeof(mempool_header));
    mptr = (mempool_type*)(to_free->pool_addr); 
    CmiLock(mptr->mempoolLock);
    mempool_free(mptr,  ptr_free); 
    CmiUnlock(mptr->mempoolLock);
}
#endif
//sorted free_list and merge it if it become continous 
void mempool_free(mempool_type *mptr, void *ptr_free)
{
    int i;
    int merged = 0;
    int free_size;
    void *free_lastbytes_pos;
    mempool_block     *mempools_head;
    size_t    *freelist_head;
    mempool_header    *freelist_head_ptr;
    mempool_header    *current;
    mempool_header *previous = NULL;
    mempool_header *to_free;

    to_free = (mempool_header *)((char*)ptr_free - sizeof(mempool_header));

    mempools_head = &(mptr->mempools_head);
    freelist_head = &mptr->freelist_head;
    freelist_head_ptr = mptr->freelist_head?(mempool_header*)((char*)mptr+mptr->freelist_head):NULL;
    current = freelist_head_ptr;

    free_size = to_free->size;
    free_lastbytes_pos = (char*)to_free +free_size;

#if  MEMPOOL_DEBUG
    printf("[%d] INSIDE FREE ptr=%p, size=%d freehead=%p mutex: %p\n", CmiMyPe(), to_free, free_size, freelist_head, mptr->mutex);
#endif
    
    while(current!= NULL && current < to_free )
    {
#if  MEMPOOL_DEBUG
        CmiPrintf("[%d] previous=%p, current=%p size:%d %p\n", CmiMyPe(), previous, current, current->size, (char*)current+current->size);
#endif
        previous = current;
        current = current->next_free?(mempool_header*)((char*)mptr + current->next_free):NULL;
    }
#if  MEMPOOL_DEBUG
    if (current) CmiPrintf("[%d] previous=%p, current=%p size:%d %p\n", CmiMyPe(), previous, current, current->size, free_lastbytes_pos);
#endif
    //continuos with previous free space 
    if(previous!= NULL && (char*)previous+previous->size == (char*)to_free &&  memcmp(&previous->mem_hndl, &to_free->mem_hndl, sizeof(mem_handle_t))==0 )
    {
        previous->size +=  free_size;
        merged = 1;
    }
    else if(current!= NULL && free_lastbytes_pos == current && memcmp(&current->mem_hndl, &to_free->mem_hndl, sizeof(mem_handle_t))==0)
    {
        to_free->size += current->size;
        to_free->next_free = current->next_free;
        current = to_free;
        merged = 1;
        if(previous == NULL)
            *freelist_head = (char*)current - (char*)mptr;
        else
            previous->next_free = (char*)to_free - (char*)mptr;
    }
    //continous, merge
    if(merged) {
       if (previous!= NULL && current!= NULL && (char*)previous + previous->size  == (char *)current && memcmp(&previous->mem_hndl, &current->mem_hndl, sizeof(mem_handle_t))==0)
      {
         previous->size += current->size;
         previous->next_free = current->next_free;
      }
    }
    else {
          // no merge to previous, current, create new entry
        to_free->next_free = current?(char*)current - (char*)mptr: 0;
        if(previous == NULL)
            *freelist_head = (char*)to_free - (char*)mptr;
        else
            previous->next_free = (char*)to_free - (char*)mptr;
    }
#if  MEMPOOL_DEBUG
    printf("[%d] Memory free done %p, freelist_head=%p\n", CmiMyPe(), to_free,  freelist_head);
#endif

    CmiAssert(*freelist_head >= 0);
}

