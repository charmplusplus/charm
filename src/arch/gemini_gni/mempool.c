/*****************************************************************************
 * $Source$
 * $Author$ Yanhua Sun 
 * $Date$   08-27-2011
 * $Revision$
 *****************************************************************************/

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define SIZE_BYTES       4
#define POOLS_NUM       2
#define MAX_INT        2147483647

#define         MEMPOOL_DEBUG   0

int malloc_count = 0;
int free_count = 0;

// for small memory allocation, large allocation
int MEMPOOL_SIZE[POOLS_NUM] = {536870912, 536870912};


typedef struct free_block_t 
{
    int     size;
    void    *mempool_ptr;   //where this entry points to
    struct  free_block_t *next;
} free_block_entry;

// multiple mempool for different size allocation
typedef struct mempool_block_t
{
    void                *mempool_base_addr;
    free_block_entry    *freelist_head;
}mempool_block;

mempool_block       mempools_data[2];

void                *mempool;
free_block_entry    *freelist_head;

void init_mempool( int pool_size)
{
    mempool = malloc(pool_size);
#if  MEMPOOL_DEBUG
    printf("Mempool init with base_addr=%p\n\n", mempool);
#endif
    freelist_head           = (free_block_entry*)malloc(sizeof(free_block_entry));
    freelist_head->size     = pool_size;
    freelist_head->mempool_ptr = mempool;
    freelist_head->next     = NULL;
}

void kill_allmempool()
{
    int i;
    for(i=0; i<POOLS_NUM; i++)
    {
        if(mempools_data[i].mempool_base_addr != NULL)
            free(mempools_data[i].mempool_base_addr);
    }
    //all free entry
}

// append size before the real memory buffer
void*  mempool_malloc(int size)
{
    int     real_size = size + SIZE_BYTES;
    void    *alloc_ptr;
    int     bestfit_size = MAX_INT; //most close size  
    free_block_entry *current = freelist_head;
    free_block_entry *previous = NULL;
    
    free_block_entry *bestfit = NULL;
    free_block_entry *bestfit_previous = NULL;
#if  MEMPOOL_DEBUG
    printf("+MALLOC request :%d\n", size);
#endif
    if(current == NULL)
    {
        //printf("Mempool overflow exit\n");
        return NULL;
    }
    while(current!= NULL)
    {
        if(current->size >= real_size && current->size<bestfit_size)
        {
            bestfit_size = current->size;
            bestfit = current;
            bestfit_previous = previous;
        }
        previous = current;
        current = current->next;
    }
    if(bestfit == NULL)
    {
        //printf("No memory has such free empty chunck of %d\n", size);
        return NULL;
    }

    alloc_ptr = bestfit->mempool_ptr+SIZE_BYTES;
    memcpy(bestfit->mempool_ptr, &size, SIZE_BYTES);
    if(bestfit->size > real_size) //deduct this entry 
    {
        bestfit->size -= real_size;
        bestfit->mempool_ptr += real_size;
    }else   //delete this free entry
    {
        if(bestfit == freelist_head)
            freelist_head = NULL;
        else
            bestfit_previous ->next = bestfit->next;
        free(bestfit);
    }
#if  MEMPOOL_DEBUG
    printf("++MALLOC served: %d, ptr:%p\n", size, alloc_ptr);

    memset(alloc_ptr, ((long int)(alloc_ptr+size))%126, size);
    printf("Memset, %p, %d vs=%ld, vr=%ld\n", alloc_ptr, size, ((long int)(alloc_ptr+size))%126,  (*((char*)alloc_ptr)));
#endif
    return alloc_ptr;
}

//sorted free_list and merge it if it become continous 
void mempool_free(void *ptr_free)
{
    int i;
    int merged = 0;
    int free_size;
    void *free_firstbytes_pos = ptr_free-SIZE_BYTES;
    void *free_lastbytes_pos;
    free_block_entry *new_entry; 
    free_block_entry *current = freelist_head;
    free_block_entry *previous = NULL;
    
    memcpy(&free_size, free_firstbytes_pos, SIZE_BYTES);
    free_lastbytes_pos = ptr_free +free_size;

#if  MEMPOOL_DEBUG
    printf("--FREE request :ptr=%p, size=%d\n", ptr_free, free_size); 
    for(i=0; i<free_size; i++)
    {
        if( (long int)(*((char*)ptr_free+i)) != ((long int)(ptr_free+free_size))%126)
        {
            printf("verifying fails, %p, %d vs=%ld, vr=%ld\n", ptr_free, free_size, ((long int)(ptr_free+free_size))%126,  (long int)(*((char*)ptr_free+i)));
            exit(2);
        }
    }
#endif
    while(current!= NULL && current->mempool_ptr < ptr_free )
    {
        previous = current;
        current = current->next;
    }
    //continuos with previous free space 
    if(previous!= NULL && previous->mempool_ptr + previous->size  == free_firstbytes_pos)
    {
        previous->size += (free_size + SIZE_BYTES);
        merged = 1;
    }
    
    if(current!= NULL && free_lastbytes_pos == current->mempool_ptr)
    {
        current->mempool_ptr = free_firstbytes_pos;
        current->size +=  (free_size + SIZE_BYTES);
        merged = 1;
    }
    //continous, merge
    if(previous!= NULL && current!= NULL && previous->mempool_ptr + previous->size  == current->mempool_ptr)
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
        new_entry->size = free_size + SIZE_BYTES;
        new_entry->next = current;
        if(previous == NULL)
            freelist_head = new_entry;
        else
            previous->next = new_entry;
    }
}

void* syh_memalign(int size, int align)
{
    int pool_index;
    void *align_ptr, *ptr;
    if(size < 1024*512)
    {
        pool_index = 0;
    }else 
        pool_index = 1;

    mempool = mempools_data[pool_index].mempool_base_addr;
    freelist_head = mempools_data[pool_index].freelist_head;
    
    if(mempool == NULL)
    {
        init_mempool(MEMPOOL_SIZE[pool_index]);
        mempools_data[pool_index].mempool_base_addr = mempool;
        mempools_data[pool_index].freelist_head = freelist_head;
    }   
    
    ptr = mempool_malloc(size+(align-1) + sizeof(void*));
 
    align_ptr = ptr + sizeof(void*);

}

void syh_alignfree(void *ptr)
{

}

// external interface 
void* syh_malloc(int size)
{
    int pool_index;
    if(size < 1024*512)
    {
        pool_index = 0;
    }else 
        pool_index = 1;

    mempool = mempools_data[pool_index].mempool_base_addr;
    freelist_head = mempools_data[pool_index].freelist_head;
    
    if(mempool == NULL)
    {
        init_mempool(MEMPOOL_SIZE[pool_index]);
        mempools_data[pool_index].mempool_base_addr = mempool;
        mempools_data[pool_index].freelist_head = freelist_head;
    }   
    malloc_count++;
    return mempool_malloc(size);
}

void syh_free(void *ptr)
{
    int i=0;
    for(i=0; i<2; i++)
    {
        if(ptr> mempools_data[i].mempool_base_addr && ptr< mempools_data[i].mempool_base_addr + MEMPOOL_SIZE[i])
        {
            mempool = mempools_data[i].mempool_base_addr;
            freelist_head = mempools_data[i].freelist_head;
            break;
        }
    }
    free_count++;
    mempool_free(ptr);
}

#if  0
#define MAX_BINS  1024*1024
void*  malloc_list[MAX_BINS];
int    empty_pos = 0;


int main(int argc, char* argv[])
{

    void *ptr;
    int mem_size = atoi(argv[2]);
    int iter = atoi(argv[1]);
    int i, size;
    struct timeval start, end;
    float timecost;
    //init_mempool();
    //srand(time(NULL));    
    if(argc<3)
    {
        printf("mempool iteration mem_size\n");
        return 1;
    }
    gettimeofday(&start, NULL); 
    for(i=0; i<iter; i++)
    {
        size = (rand()%(131)) + mem_size;
        if(empty_pos == MAX_BINS)
        {
            empty_pos--;
            syh_free(malloc_list[empty_pos]);
        }
        while( (malloc_list[empty_pos] = syh_malloc(size)) == NULL)
        {
            empty_pos--;
            syh_free(malloc_list[empty_pos]);
        }
        empty_pos++;
        if(rand()%3 == 0 && empty_pos>0)
        {
            empty_pos--;
            syh_free(malloc_list[empty_pos]);
        }
    }
    gettimeofday(&end, NULL); 
    
    timecost =  ((end.tv_sec * 1000000.0 + end.tv_usec)- (start.tv_sec * 1000000 + start.tv_usec))/iter;
    printf("Memsize:%d Malloc time:%f us\n",  mem_size, timecost); 
    kill_allmempool();
}

#endif
