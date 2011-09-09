/*****************************************************************************
 * $Source$
 * $Author$ Yanhua Sun 
 * $Date$   08-27-2011
 * $Revision$
 *****************************************************************************/

/** Memory pool implementation , It is only good for Charm++ usage. The first 64 bytes provides additional information. sizeof(int)- size of this block(free or allocated), next gni_mem_handle_t, then void** point to the next available block, */
#define MEMPOOL_DEBUG   0

#define POOLS_NUM       2
#define MAX_INT        2147483647

#define  GetMemHndl(x)  ((mempool_header*)((char*)x-ALIGNBUF))->mem_hndl

static      size_t     expand_mem = 1024ll*1024*16;

// multiple mempool for different size allocation
typedef struct mempool_block_t
{
    void                *mempool_ptr;
    gni_mem_handle_t    mem_hndl;
    struct              mempool_block_t *next;
} mempool_block;

mempool_block       *mempools_head = 0;

typedef struct mempool_header
{
  int size;
  gni_mem_handle_t  mem_hndl;
  struct mempool_header *next;
} mempool_header;

mempool_header    *freelist_head = NULL;

void init_mempool(size_t pool_size)
{
    gni_return_t status;
    mempool_header *header;
    //CmiPrintf("[%d] mempool ask for init size %d\n", CmiMyPe(), pool_size);
    mempools_head = (mempool_block*)malloc(sizeof(mempool_block));
    mempools_head->mempool_ptr = memalign(ALIGNBUF, pool_size);
    freelist_head = header = (mempool_header *) mempools_head->mempool_ptr;
    status = MEMORY_REGISTER(onesided_hnd, nic_hndl, header, pool_size,  &(mempools_head->mem_hndl), &omdh);
    mempools_head->next = NULL;
    GNI_RC_CHECK("Mempool register", status);
    header->size = pool_size;
    header->mem_hndl = mempools_head->mem_hndl;
    header->next = NULL;

#if  MEMPOOL_DEBUG
    CmiPrintf("[%d] mempool init, MEM:%p, next=%p\n", CmiMyPe(), freelist_head, freelist_head->next);
#endif
}

void kill_allmempool()
{
    gni_return_t status;
    
    mempool_block *current = mempools_head;
    while(mempools_head!= NULL)
    {
        status = GNI_MemDeregister(nic_hndl, &(mempools_head->mem_hndl));
        GNI_RC_CHECK("Mempool de-register", status);
        //printf("[%d] free mempool:%p\n", CmiMyPe(), mempools_head->mempool_ptr);
        free( mempools_head->mempool_ptr);
        current=mempools_head;
        mempools_head = mempools_head->next;
        free(current);
    }
}

// append size before the real memory buffer
void*  syh_mempool_malloc(int size)
{
    int     bestfit_size = MAX_INT; //most close size 
    mempool_block   *expand_pool;
    gni_return_t    status;
    mempool_header    *current = freelist_head;
    mempool_header    *previous = NULL;
    mempool_header    *bestfit = NULL;
    mempool_header    *bestfit_previous = NULL;
    int     expand_size;

#if  MEMPOOL_DEBUG
    CmiPrintf("[%d] request malloc for size %d, freelist:%p\n", CmiMyPe(), size, freelist_head);
#endif
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
        current = current->next;
    }
#else
    while(current!= NULL)             /*  first fit */
    {
#if  MEMPOOL_DEBUG
        CmiPrintf("[%d] current=%p size:%d ->%p \n", CmiMyPe(), current, current->size, (char*)current+current->size);
#endif
        if(current->size >= size)
        {
            bestfit_size = current->size;
            bestfit = current;
            bestfit_previous = previous;
            break;
        }
        previous = current;
        current = current->next;
    }
#endif

    if(bestfit == NULL)
    {
        expand_size = expand_mem>size ? expand_mem:2*size; 
        expand_pool         = (mempool_block*)malloc(sizeof(mempool_block));
        expand_pool->mempool_ptr = memalign(ALIGNBUF, expand_size);
        printf("[%d] No memory has such free empty chunck of %d. expanding %p (%d)\n", CmiMyPe(), size, expand_pool->mempool_ptr, expand_size);
        status = MEMORY_REGISTER(onesided_hnd, nic_hndl, expand_pool->mempool_ptr, expand_size,  &(expand_pool->mem_hndl), &omdh);
        GNI_RC_CHECK("Mempool register", status);
        expand_pool->next = mempools_head;
        mempools_head = expand_pool;
        
        bestfit = expand_pool->mempool_ptr;
        bestfit->size = expand_size;
        bestfit->mem_hndl = expand_pool->mem_hndl;
        bestfit->next = NULL;
        bestfit_size = expand_size;
#if 0
        current = freelist_head;
        while(current!= NULL && current < bestfit )
        {
          previous = current;
          current = current->next;
        }
#else
        CmiAssert(bestfit > previous);
#endif
        bestfit_previous = previous;
        if (previous == NULL)
           freelist_head = bestfit;
        else
           previous->next = bestfit;
    }

    bestfit->size = size;
    if(bestfit_size > size) //deduct this entry 
    {
        mempool_header *ptr = (mempool_header *)((char*)bestfit + size);
        ptr->size = bestfit_size - size;
        ptr->mem_hndl = bestfit->mem_hndl;
        ptr->next = bestfit->next;
        if(bestfit == freelist_head)
            freelist_head = ptr;
        if(bestfit_previous != NULL)
            bestfit_previous->next= ptr;
    }
    else {  
          //delete this free entry
        if(bestfit == freelist_head)
            freelist_head = freelist_head->next;
        else
            bestfit_previous->next = bestfit->next;
    }
#if  MEMPOOL_DEBUG
    printf("[%d] ++MALLOC served: %d, ptr:%p\n", CmiMyPe(), size, bestfit);
#endif
    return (char*)bestfit;
}

//sorted free_list and merge it if it become continous 
void syh_mempool_free(void *ptr_free)
{
    int i;
    int merged = 0;
    int free_size;
    void *free_lastbytes_pos;
    mempool_header *current = freelist_head;
    mempool_header *previous = NULL;
    mempool_header *to_free = (mempool_header *)ptr_free;

    free_size = to_free->size;
    free_lastbytes_pos = (char*)ptr_free +free_size;
#if  MEMPOOL_DEBUG
    printf("[%d] INSIDE FREE ptr=%p, size=%d freehead=%p\n", CmiMyPe(), ptr_free, free_size, freelist_head);
#endif
    
    while(current!= NULL && current < to_free )
    {
#if  MEMPOOL_DEBUG
        CmiPrintf("[%d] previous=%p, current=%p size:%d %p\n", CmiMyPe(), previous, current, current->size, (char*)current+current->size);
#endif
        previous = current;
        current = current->next;
    }
#if  MEMPOOL_DEBUG
    if (current) CmiPrintf("[%d] previous=%p, current=%p size:%d %p\n", CmiMyPe(), previous, current, current->size, free_lastbytes_pos);
#endif
    //continuos with previous free space 
    if(previous!= NULL && (char*)previous+previous->size == ptr_free &&  memcmp(&previous->mem_hndl, &to_free->mem_hndl, sizeof(gni_mem_handle_t))==0 )
    {
        previous->size +=  free_size;
        merged = 1;
    }
    else if(current!= NULL && free_lastbytes_pos == current && memcmp(&current->mem_hndl, &to_free->mem_hndl, sizeof(gni_mem_handle_t))==0)
    {
        to_free->size += current->size;
        to_free->next = current->next;
        current = to_free;
        merged = 1;
        if(previous == NULL)
            freelist_head = current;
        else
            previous->next = to_free;
    }
    //continous, merge
    if(merged) {
       if (previous!= NULL && current!= NULL && (char*)previous + previous->size  == (char *)current && memcmp(&previous->mem_hndl, &current->mem_hndl, sizeof(gni_mem_handle_t))==0)
      {
         previous->size += current->size;
         previous->next = current->next;
      }
    }
    else {
          // no merge to previous, current, create new entry
        to_free->next = current;
        if(previous == NULL)
            freelist_head = to_free;
        else
            previous->next = to_free;
    }
#if  MEMPOOL_DEBUG
    printf("[%d] Memory free done %p, freelist_head=%p\n", CmiMyPe(), ptr_free,  freelist_head);
#endif
}

