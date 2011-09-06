/*****************************************************************************
 * $Source$
 * $Author$ Yanhua Sun 
 * $Date$   08-27-2011
 * $Revision$
 *****************************************************************************/

/** Memory pool implementation , It is only good for Charm++ usage. The first 64 bytes provides additional information. sizeof(int)- size of this block(free or allocated), next gni_mem_handle_t, then void** point to the next available block, */
#define MEMPOOL_DEBUG   0
#define SIZEOFINT       sizeof(int)
#define POOLS_NUM       2
#define MAX_INT        2147483647

#define  GetMemHndl(x)  (x-ALIGNBUF+SIZEOFINT)

#define    POS_NEXT(ptr)    (ptr+SIZEOFINT+sizeof(gni_mem_handle_t))

static      int     expand_mem = 1024*1024*16;

// multiple mempool for different size allocation
typedef struct mempool_block_t
{
    void                *mempool_ptr;
    gni_mem_handle_t    mem_hndl;
    struct              mempool_block_t *next;
}mempool_block;

mempool_block       *mempools_head = 0;

void    *freelist_head;

void init_mempool( int pool_size)
{
    void *ptr;
    gni_return_t status;
    //CmiPrintf("[%d] mempool ask for init size %d\n", CmiMyPe(), pool_size);
    mempools_head = (mempool_block*)malloc(sizeof(mempool_block));
    ptr = mempools_head->mempool_ptr = memalign(ALIGNBUF, pool_size);
    status = MEMORY_REGISTER(onesided_hnd, nic_hndl, ptr, pool_size,  &(mempools_head->mem_hndl), &omdh);
    mempools_head->next = NULL;
    GNI_RC_CHECK("Mempool register", status);
    freelist_head = ptr;
    *((int*)ptr) = pool_size;
    ptr += SIZEOFINT;
    memcpy(ptr, &(mempools_head->mem_hndl), sizeof(gni_mem_handle_t));
    ptr += sizeof(gni_mem_handle_t);
    *((void**)ptr) = NULL;
#if  MEMPOOL_DEBUG
    CmiPrintf("[%d] mempool init, MEM:%p, next=%p\n", CmiMyPe(), freelist_head, *((void**)POS_NEXT(freelist_head)));
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
    void    *alloc_ptr;
    int     bestfit_size = MAX_INT; //most close size 
    mempool_block   *expand_pool;
    gni_return_t    status;
    void    *current = freelist_head;
    void    *previous = NULL;
    
    void    *bestfit = NULL;
    void    *bestfit_previous = NULL;
    int     expand_size;

#if  MEMPOOL_DEBUG
    CmiPrintf("request malloc for size %d, freelist:%p\n", size, freelist_head);
#endif
    while(current!= NULL)
    {
        if(*((int*)current) >= size && *((int*)current)<bestfit_size)
        {
            bestfit_size = *((int*)current);
            bestfit = current;
            bestfit_previous = previous;
        }
        //printf(" free entr:%p, size=%d\n", current->mempool_ptr, current->size);
        previous = current;
        current = *((void**)(POS_NEXT(current)));
#if  MEMPOOL_DEBUG
        CmiPrintf("next=%p\n", current);
#endif
    }

    if(bestfit == NULL)
    {
      
        expand_size = expand_mem>size ? expand_mem:2*size; 
        expand_pool         = (mempool_block*)malloc(sizeof(mempool_block));
        expand_pool->mempool_ptr = memalign(ALIGNBUF, expand_size);
        printf("No memory has such free empty chunck of %d. expanding %p (%d)\n", size, expand_pool->mempool_ptr, expand_size);
        status = MEMORY_REGISTER(onesided_hnd, nic_hndl, expand_pool->mempool_ptr, expand_size,  &(expand_pool->mem_hndl), &omdh);
        GNI_RC_CHECK("Mempool register", status);
        expand_pool->next = mempools_head;
        mempools_head = expand_pool;
        
        alloc_ptr = expand_pool->mempool_ptr;
        *((int*)alloc_ptr) = size;
        memcpy(alloc_ptr+SIZEOFINT, &(expand_pool->mem_hndl), sizeof(gni_mem_handle_t));
        
        bestfit = alloc_ptr+size;
        *((int*)bestfit) = expand_size - size;
        memcpy(bestfit+SIZEOFINT, &(expand_pool->mem_hndl), sizeof(gni_mem_handle_t));
        *((void**)POS_NEXT(bestfit))= NULL;
        if(previous == NULL)
            freelist_head = bestfit;
        else
           *( (void**)POS_NEXT(previous)) = bestfit;
#if  MEMPOOL_DEBUG
        CmiPrintf("freelist_head=%p bestfit=%p, size=%d \n", freelist_head, bestfit, *((int*)bestfit));
#endif
        
        return alloc_ptr;
    }

    alloc_ptr = bestfit;
    *((int*)alloc_ptr) = size;
    if(bestfit_size > size) //deduct this entry 
    {
        if(bestfit == freelist_head)
            freelist_head += size;
        bestfit += size;
        if(bestfit_previous != NULL)
            *((void**)POS_NEXT(bestfit_previous)) = bestfit;
        *((int*)bestfit) = bestfit_size -size;
        memcpy(bestfit+SIZEOFINT, alloc_ptr+SIZEOFINT, sizeof(gni_mem_handle_t));
        memcpy(POS_NEXT(bestfit), POS_NEXT(alloc_ptr), sizeof(void*));
    }else   //delete this free entry
    {
        if(bestfit == freelist_head)
            freelist_head = *((void**)POS_NEXT(freelist_head));
        else
            memcpy(POS_NEXT(bestfit_previous), POS_NEXT(bestfit), sizeof(void*));
    }
#if  MEMPOOL_DEBUG
    printf("++MALLOC served: %d, ptr:%p\n", size, alloc_ptr);
#endif
    return alloc_ptr;
}

//sorted free_list and merge it if it become continous 
void syh_mempool_free(void *ptr_free)
{
    int i;
    int merged = 0;
    int free_size;
    void *free_lastbytes_pos;
    void *current = freelist_head;
    void *previous = NULL;

#if  MEMPOOL_DEBUG
    printf("INSIDE FREE ptr=%p, freehead=%p\n", ptr_free, freelist_head);
#endif
    free_size = *((int*)ptr_free);
    free_lastbytes_pos = ptr_free +free_size;
    
    while(current!= NULL && current < ptr_free )
    {
        previous = current;
        current = *((void**)POS_NEXT(current));
#if  MEMPOOL_DEBUG
        CmiPrintf("previous=%p, current=%p\n", previous, current);
#endif
    }
    //continuos with previous free space 
    if(previous!= NULL && previous+ *((int*)previous)   == ptr_free &&  memcmp(previous+SIZEOFINT, ptr_free+SIZEOFINT, sizeof(gni_mem_handle_t))==0 )
    {
        *((int*)previous) +=  free_size;
        merged = 1;
    }
    else if(current!= NULL && free_lastbytes_pos == current && memcmp(current+SIZEOFINT, ptr_free+SIZEOFINT, sizeof(gni_mem_handle_t))==0)
    {
        *((int*)ptr_free) += *((int*)current);
        *((void**)POS_NEXT(ptr_free)) = *((void**)POS_NEXT(current));
        current = ptr_free;
        merged = 1;
        if(previous == NULL)
            freelist_head = current;
    }
    //continous, merge
    if(previous!= NULL && current!= NULL && previous + *((int*)previous)  == current && memcmp(previous+SIZEOFINT, current+SIZEOFINT, sizeof(gni_mem_handle_t))==0)
    {
       *((int*)previous) += *((int*)current);
       *((void**)(POS_NEXT(previous))) = *((void**)POS_NEXT(current));
    }
    // no merge to previous, current, create new entry
#if  MEMPOOL_DEBUG
        CmiPrintf("++previous=%p, current=%p\n", previous, current);
#endif
    if(merged == 0)
    {
        *((void**)POS_NEXT(ptr_free)) = current;
        if(previous == NULL)
            freelist_head = ptr_free;
        else
            *((void**)POS_NEXT(previous)) = ptr_free;
    }
#if  MEMPOOL_DEBUG
    printf("Memory free done %p, freelist_head=%p\n", ptr_free,  freelist_head);
#endif
}

