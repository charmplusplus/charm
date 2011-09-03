/*****************************************************************************
 * $Source$
 * $Author$ Yanhua Sun 
 * $Date$   08-27-2011
 * $Revision$
 *****************************************************************************/

/** Memory pool implementation */
#define MEMPOOL_DEBUG   0
#define SIZE_BYTES       4
#define POOLS_NUM       2
#define MAX_INT        2147483647

#define  GetMemHndl(x)  (x-ALIGNBUF+SIZE_BYTES)
// for small memory allocation, large allocation
int MEMPOOL_SIZE[POOLS_NUM] = {536870912, 536870912};

typedef struct free_block_t 
{
    int     size;
    void    *mempool_ptr;   //where this entry points to
    gni_mem_handle_t    mem_hndl;
    struct  free_block_t *next;
} free_block_entry;

// multiple mempool for different size allocation
typedef struct mempool_block_t
{
    void                *mempool_base_addr;
    gni_mem_handle_t    mempool_hndl;
    free_block_entry    *freelist_head;
}mempool_block;

mempool_block       mempools_data[2];

free_block_entry    *freelist_head;
//free_block_entry    **freelist_head;
void init_mempool( int pool_size)
{
    gni_return_t status;
    freelist_head           = (free_block_entry*)malloc(sizeof(free_block_entry));
    (freelist_head)->size     = pool_size;
    (freelist_head)->mempool_ptr = memalign(ALIGNBUF, pool_size);
    status = MEMORY_REGISTER(onesided_hnd, nic_hndl, freelist_head->mempool_ptr, pool_size,  &(freelist_head->mem_hndl), &omdh);
    GNI_RC_CHECK("Mempool register", status);
#if  MEMPOOL_DEBUG
    printf("Mempool init with base_addr=%p\n\n", freelist_head->mempool_ptr);
#endif
    (freelist_head)->next     = NULL;
}

void kill_allmempool()
{
    gni_return_t status;
    
    free_block_entry *current = freelist_head;
    while(freelist_head!= NULL)
    {
        status = GNI_MemDeregister(nic_hndl, &(freelist_head->mem_hndl));
        GNI_RC_CHECK("Mempool de-register", status);
        //printf("[%d] free mempool:%p\n", CmiMyPe(), freelist_head->mempool_ptr);
        free( freelist_head->mempool_ptr);
        current=freelist_head;
        freelist_head = freelist_head->next;
        free(current);
    }
    //all free entry
}

// append size before the real memory buffer
void*  syh_mempool_malloc(int size)
{
    int     real_size = size;
    void    *alloc_ptr;
    int     bestfit_size = MAX_INT; //most close size  
    gni_return_t status;
    free_block_entry *current = freelist_head;
    free_block_entry *previous = NULL;
    
    free_block_entry *bestfit = NULL;
    free_block_entry *bestfit_previous = NULL;
#if  MEMPOOL_DEBUG
    printf("+MALLOC request :%d, freehead=%p, current=%p\n", size, current->mempool_ptr, (freelist_head)->mempool_ptr);
#endif
    while(current!= NULL)
    {
        if(current->size >= real_size && current->size<bestfit_size)
        {
            bestfit_size = current->size;
            bestfit = current;
            bestfit_previous = previous;
        }
        //printf(" free entr:%p, size=%d\n", current->mempool_ptr, current->size);
        previous = current;
        current = current->next;

    }
    if(bestfit == NULL)
    {
        
        bestfit           = (free_block_entry*)malloc(sizeof(free_block_entry));
        bestfit ->size     = size*8;
        bestfit ->mempool_ptr = memalign(ALIGNBUF, bestfit->size);
        //printf("No memory has such free empty chunck of %d. Expanding mempool %p\n", size,  bestfit->mempool_ptr);
        status = MEMORY_REGISTER(onesided_hnd, nic_hndl, bestfit->mempool_ptr, bestfit->size,  &(bestfit->mem_hndl), &omdh);
        GNI_RC_CHECK("Mempool register", status);
        bestfit->next     = NULL;

        alloc_ptr = bestfit->mempool_ptr;//+SIZE_BYTES;
        memcpy(bestfit->mempool_ptr, &size, SIZE_BYTES);
        memcpy(bestfit->mempool_ptr+SIZE_BYTES, &(bestfit->mem_hndl), sizeof(gni_mem_handle_t));
        bestfit->size -= size;
        bestfit->mempool_ptr += size;
        if(previous == NULL)
            freelist_head = bestfit;
        else
            previous->next = bestfit;
        return alloc_ptr;
    }

    alloc_ptr = bestfit->mempool_ptr;//+SIZE_BYTES;
    memcpy(bestfit->mempool_ptr, &size, SIZE_BYTES);
    memcpy(bestfit->mempool_ptr+SIZE_BYTES, &(bestfit->mem_hndl), sizeof(gni_mem_handle_t));
    if(bestfit->size > real_size) //deduct this entry 
    {
        bestfit->size -= real_size;
        bestfit->mempool_ptr += real_size;
    }else   //delete this free entry
    {
        if(bestfit == freelist_head)
            freelist_head = freelist_head->next;
        else
            bestfit_previous ->next = bestfit->next;
        free(bestfit);
    }
#if  MEMPOOL_DEBUG
    printf("++MALLOC served: %d, ptr:%p\n", size, alloc_ptr);

    //memset(alloc_ptr+SIZE_BYTES, ((long int)(alloc_ptr+size))%126, size);
    //printf("Memset, %p, %d vs=%ld, vr=%ld\n", alloc_ptr, size, ((long int)(alloc_ptr+size))%126,  (*((char*)alloc_ptr)));
#endif
    return alloc_ptr;
}

//sorted free_list and merge it if it become continous 
void syh_mempool_free(void *ptr_free)
{
    int i;
    int merged = 0;
    int free_size;
    void *free_firstbytes_pos = ptr_free;
    void *free_lastbytes_pos;
    free_block_entry *new_entry; 
    free_block_entry *current = freelist_head;
    free_block_entry *previous = NULL;
    gni_mem_handle_t    mem_hndl;

#if  MEMPOOL_DEBUG
    printf("INSIDE FREE ptr=%p, freehead=%p\n", ptr_free, freelist_head);
    /*for(i=0; i<free_size; i++)
    {
        if( (long int)(*((char*)ptr_free+i)) != ((long int)(ptr_free+free_size))%126)
        {
            printf("verifying fails, %p, %d vs=%ld, vr=%ld\n", ptr_free, free_size, ((long int)(ptr_free+free_size))%126,  (long int)(*((char*)ptr_free+i)));
            exit(2);
        }
    }*/
#endif
    memcpy(&free_size, free_firstbytes_pos, SIZE_BYTES);
    free_lastbytes_pos = ptr_free +free_size;
    
    while(current!= NULL && current->mempool_ptr < ptr_free )
    {
        previous = current;
        current = current->next;
    }
    //printf("pre=%p, current=%p, current->ptr=%p\n", previous, current->mempool_ptr);
    //continuos with previous free space 
    if(previous!= NULL && previous->mempool_ptr + previous->size  == free_firstbytes_pos &&  memcmp(&(previous->mem_hndl), ptr_free+SIZE_BYTES, sizeof(gni_mem_handle_t))==0 )
    {
        previous->size += free_size;
        merged = 1;
    }
    else
    if(current!= NULL && free_lastbytes_pos == current->mempool_ptr && memcmp(&(current->mem_hndl), ptr_free+SIZE_BYTES, sizeof(gni_mem_handle_t))==0)
    {
        current->mempool_ptr = free_firstbytes_pos;
        current->size +=  free_size ;
        merged = 1;
    }
    //continous, merge
    if(previous!= NULL && current!= NULL && previous->mempool_ptr + previous->size  == current->mempool_ptr && memcmp(&(previous->mem_hndl), &(current->mem_hndl), sizeof(gni_mem_handle_t))==0)
    {
       previous->size += current->size;
       previous->next = current->next;
       free(current);
    }
    // no merge to previous, current, create new entry
    if(merged == 0)
    {
        new_entry = malloc(sizeof(free_block_entry));
        new_entry->mempool_ptr = free_firstbytes_pos;
        memcpy(&(new_entry->mem_hndl), ptr_free+SIZE_BYTES, sizeof( gni_mem_handle_t));
        new_entry->size = free_size ;
        new_entry->next = current;
        if(previous == NULL)
            freelist_head = new_entry;
        else
            previous->next = new_entry;
        //printf(" create new entry, freehead=%p, %p\n", (freelist_head)->mempool_ptr, free_firstbytes_pos);
    }
#if  MEMPOOL_DEBUG
    printf("Memory free done %p\n", ptr_free);
#endif
}

