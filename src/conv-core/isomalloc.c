/**************************************************************************
Isomalloc:
  A way to allocate memory at the same address on every processor.
This enables linked data structures, like thread stacks, to be migrated
to the same address range on other processors.  This is similar to an
explicitly managed shared memory system.

  The memory is used and released via the mmap()/mumap() calls, so unused
memory does not take any (RAM, swap or disk) space.

  The way it's implemented is that each processor claims some section 
of the available virtual address space, and statisfies all new allocations
from that space.  Migrating structures use whatever space they started with.

Written for migratable threads by Milind Bhandarkar around August 2000;
generalized by Orion Lawlor November 2001.
*/
#include "converse.h"

#include <stdlib.h>

/*Size in bytes of a single slot*/
static int slotsize;

/*Total number of slots per processor*/
static int numslots;

/*Start and end of isomalloc-managed addresses*/
static char *isomallocStart;
static char *isomallocEnd;

/*Utility conversion functions*/
static int addr2slot(void *addr) {
	return (((char *)addr)-isomallocStart)/slotsize;
}
static void *slot2addr(int slot) {
	return isomallocStart+slotsize*slot;
}
static int slot2pe(int slot) {
	return slot/numslots;
}
static int pe2slot(int pe) {
	return pe*numslots;
}
static int length2slots(int nBytes) {
	return (nBytes+slotsize-1)/slotsize;
}
static int slots2length(int nSlots) {
	return nSlots*slotsize;
}

typedef struct _slotblock
{
  int startslot;
  int nslots;
} slotblock;

typedef struct _slotset
{
  int maxbuf;
  slotblock *buf;
  int emptyslots;
} slotset;

/*
 * creates a new slotset of nslots entries, starting with all
 * empty slots. The slot numbers are [startslot,startslot+nslot-1]
 */
static slotset *
new_slotset(int startslot, int nslots)
{
  int i;
  slotset *ss = (slotset*) malloc(sizeof(slotset));
  _MEMCHECK(ss);
  ss->maxbuf = 16;
  ss->buf = (slotblock *) malloc(sizeof(slotblock)*ss->maxbuf);
  _MEMCHECK(ss->buf);
  ss->emptyslots = nslots;
  ss->buf[0].startslot = startslot;
  ss->buf[0].nslots = nslots;
  for (i=1; i<ss->maxbuf; i++)
    ss->buf[i].nslots = 0;
  return ss;
}

/*
 * returns new block of empty slots. if it cannot find any, returns (-1).
 */
static int
get_slots(slotset *ss, int nslots)
{
  int i;
  if(ss->emptyslots < nslots)
    return (-1);
  for(i=0;i<(ss->maxbuf);i++)
    if(ss->buf[i].nslots >= nslots)
      return ss->buf[i].startslot;
  return (-1);
}

/* just adds a slotblock to an empty position in the given slotset. */
static void
add_slots(slotset *ss, int sslot, int nslots)
{
  int pos, emptypos = -1;
  if (nslots == 0)
    return;
  for (pos=0; pos < (ss->maxbuf); pos++) {
    if (ss->buf[pos].nslots == 0) {
      emptypos = pos;
      break; /* found empty slotblock */
    }
  }
  if (emptypos == (-1)) /*no empty slotblock found */
  {
    int i;
    int newsize = ss->maxbuf*2;
    slotblock *newbuf = (slotblock *) malloc(sizeof(slotblock)*newsize);
    _MEMCHECK(newbuf);
    for (i=0; i<(ss->maxbuf); i++)
      newbuf[i] = ss->buf[i];
    for (i=ss->maxbuf; i<newsize; i++)
      newbuf[i].nslots  = 0;
    free(ss->buf);
    ss->buf = newbuf;
    emptypos = ss->maxbuf;
    ss->maxbuf = newsize;
  }
  ss->buf[emptypos].startslot = sslot;
  ss->buf[emptypos].nslots = nslots;
  ss->emptyslots += nslots;
  return;
}

/* grab a slotblock with specified range of blocks
 * this is different from get_slots, since it pre-specifies the
 * slots to be grabbed.
 */
static void
grab_slots(slotset *ss, int sslot, int nslots)
{
  int pos, eslot, e;
  eslot = sslot + nslots;
  for (pos=0; pos < (ss->maxbuf); pos++)
  {
    if (ss->buf[pos].nslots == 0)
      continue;
    e = ss->buf[pos].startslot + ss->buf[pos].nslots;
    if(sslot >= ss->buf[pos].startslot && eslot <= e)
    {
      int old_nslots;
      old_nslots = ss->buf[pos].nslots;
      ss->buf[pos].nslots = sslot - ss->buf[pos].startslot;
      ss->emptyslots -= (old_nslots - ss->buf[pos].nslots);
      add_slots(ss, sslot + nslots, old_nslots - ss->buf[pos].nslots - nslots);
      return;
    }
  }
  CmiAbort("requested a non-existent slotblock\n");
}

/*
 * Frees slot by adding it to one of the blocks of empty slots.
 * this slotblock is one which is contiguous with the slots to be freed.
 * if it cannot find such a slotblock, it creates a new slotblock.
 * If the buffer fills up, it adds up extra buffer space.
 */
static void
free_slots(slotset *ss, int sslot, int nslots)
{
  int pos;
  /* eslot is the ending slot of the block to be freed */
  int eslot = sslot + nslots;
  for (pos=0; pos < (ss->maxbuf); pos++)
  {
    int e = ss->buf[pos].startslot + ss->buf[pos].nslots;
    if (ss->buf[pos].nslots == 0)
      continue;
    /* e is the ending slot of pos'th slotblock */
    if (e == sslot) /* append to the current slotblock */
    {
	    ss->buf[pos].nslots += nslots;
      ss->emptyslots += nslots;
	    return;
    }
    if(eslot == ss->buf[pos].startslot) /* prepend to the current slotblock */
    {
	    ss->buf[pos].startslot = sslot;
	    ss->buf[pos].nslots += nslots;
      ss->emptyslots += nslots;
	    return;
    }
  }
  /* if we are here, it means we could not find a slotblock that the */
  /* block to be freed was combined with. */
  add_slots(ss, sslot, nslots);
}

/*
 * destroys slotset
 */
static void
delete_slotset(slotset* ss)
{
  free(ss->buf);
  free(ss);
}

#if CMK_THREADS_DEBUG
static void
print_slots(slotset *ss)
{
  int i;
  CmiPrintf("[%d] maxbuf = %d\n", CmiMyPe(), ss->maxbuf);
  CmiPrintf("[%d] emptyslots = %d\n", CmiMyPe(), ss->emptyslots);
  for(i=0;i<ss->maxbuf;i++) {
    if(ss->buf[i].nslots)
      CmiPrintf("[%d] (%d, %d) \n", CmiMyPe(), ss->buf[i].startslot, 
          ss->buf[i].nslots);
  }
}
#endif

#if CMK_THREADS_ARE_WIN32_FIBERS || ! CMK_HAS_MMAN_H || CMK_NO_ISO_MALLOC
/****************** Manipulate memory map (Win32 non-version) *****************/
static void *
map_slots(int slot, int nslots)
{
	CmiAbort("isomalloc: Cannot map/unmap memory");
}

static void
unmap_slots(int slot, int nslots)
{
	CmiAbort("isomalloc: Cannot map/unmap memory");
}

static void 
init_map(char **argv)
{
}
#else
/****************** Manipulate memory map (UNIX version) *****************/
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

CpvStaticDeclare(int, zerofd); /*File descriptor for /dev/zero, for mmap*/

/*
 * maps the virtual memory associated with slot using mmap
 */
static void *
map_slots(int slot, int nslots)
{
  void *pa;
  void *addr=slot2addr(slot);
  pa = mmap(addr, slotsize*nslots, 
            PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED,
            CpvAccess(zerofd), 0);
  if((pa==((void*)(-1))) || (pa != addr)) {
    CmiError("mmap call failed to allocate %d bytes at %p.\n", slotsize*nslots, addr);
    CmiAbort("Exiting\n");
  }
#if CMK_THREADS_DEBUG
  CmiPrintf("[%d] mmap'd slots %d-%d to address %p\n",CmiMyPe(),
	    slot,slot+nslots-1,addr);
#endif
  return pa;
}

/*
 * unmaps the virtual memory associated with slot using munmap
 */
static void
unmap_slots(int slot, int nslots)
{
  void *addr=slot2addr(slot);
  int retval = munmap(addr, slotsize*nslots);
  if (retval==(-1))
    CmiAbort("munmap call failed to deallocate requested memory.\n");
#if CMK_THREADS_DEBUG
  CmiPrintf("[%d] munmap'd slots %d-%d from address %p\n",CmiMyPe(),
	    slot,slot+nslots-1,addr);
#endif
}

static void 
init_map(char **argv)
{
  CpvInitialize(int, zerofd);  
  CpvAccess(zerofd) = open("/dev/zero", O_RDWR);
  if(CpvAccess(zerofd)<0)
    CmiAbort("Cannot open /dev/zero. Aborting.\n");	
}

#endif /* UNIX memory map */

/************ Address space voodoo: find free address range **********/
CpvStaticDeclare(slotset *, myss); /*My managed slots*/

/*This struct describes a range of virtual addresses*/
typedef unsigned long memRange_t;
typedef struct {
  char *start; /*First byte of region*/
  memRange_t len; /*Number of bytes in region*/
  const char *type; /*String describing memory in region (debugging only)*/
} memRegion_t;

/*Estimate the top of the current stack*/
static void *__cur_stack_frame(void)
{
  char __dummy;
  void *top_of_stack=(void *)&__dummy;
  return top_of_stack;
}
/*Estimate the location of the static data region*/
static void *__static_data_loc(void)
{
  static char __dummy;
  return (void *)&__dummy;
}

static char *pmin(char *a,char *b) {return (a<b)?a:b;}
static char *pmax(char *a,char *b) {return (a>b)?a:b;}

/*Find the first available memory region of at least the
  given size not touching any data in the used list.
 */
static memRegion_t find_free_region(memRegion_t *used,int nUsed,int atLeast) 
{
  memRegion_t max;
  int i,j;  

  /*Find the largest hole between regions*/
  for (i=0;i<nUsed;i++) {
    /*Consider a hole starting at the end of region i*/
    char *holeStart=used[i].start+used[i].len;
    char *holeEnd=(void *)(-1);
    
    /*Shrink the hole by all others*/ 
    for (j=0;j<nUsed && holeStart<holeEnd;j++) {
      if (used[j].start<holeStart) holeStart=pmax(holeStart,used[j].start+used[j].len);
      else if (used[j].start<holeEnd) holeEnd=pmin(holeEnd,used[j].start);
    } 
    if (holeStart<holeEnd) 
    { /*There is some room between these two regions*/
      memRange_t len=((memRange_t)holeEnd)-((memRange_t)holeStart);
#if CMK_THREADS_DEBUG
      CmiPrintf("[%d] Usable address space at %p - %p\n",CmiMyPe(),holeStart,holeEnd);
#endif
      if (len>=atLeast) { /*This is a large enough hole-- return it*/
	max.len=len;
	max.start=holeStart;
	max.type="Unused";
	return max;
      }
    }
  }

  /*If we get here, there are no large enough holes*/
  CmiAbort("ISOMALLOC cannot locate any free virtual address space!");
  return max; /*<- never executed*/
}

static void init_ranges(char **argv)
{
  /*Round slot size up to nearest page size*/
  slotsize=16*1024;
  slotsize=(slotsize+CMK_MEMORY_PAGESIZE-1) & ~(CMK_MEMORY_PAGESIZE-1);
#if CMK_THREADS_DEBUG
  CmiPrintf("[%d] Using slotsize of %d\n", CmiMyPe(), slotsize);
#endif

  /* Find the largest unused region of virtual address space */
  {
    char *staticData =(char *) __static_data_loc();
    char *code = (char *)&init_ranges;
    char *heapLil = (char*) malloc(1);
    char *heapBig = (char*) malloc(4*1024*1024);
    char *stack = (char *)__cur_stack_frame();

    memRange_t meg=1024*1024; /*One megabyte*/
    memRange_t gig=1024*meg; /*One gigabyte*/
    int i,nRegions=6;
    memRegion_t regions[6]; /*used portions of address space*/
    memRegion_t freeRegion; /*Largest unused block of address space*/

/*Mark off regions of virtual address space as ususable*/
    regions[0].type="NULL (inaccessible)";
    regions[0].start=NULL; regions[0].len=16u*meg;
    
    regions[1].type="Static program data";
    regions[1].start=staticData; regions[1].len=256u*meg;
    
    regions[2].type="Program executable code";
    regions[2].start=code; regions[2].len=256u*meg;
    
    regions[3].type="Heap (small blocks)";
    regions[3].start=heapLil; regions[3].len=2u*gig;

    regions[4].type="Heap (large blocks)";
    regions[4].start=heapBig; regions[4].len=2u*gig;
    
    regions[5].type="Stack space";
    regions[5].start=stack; regions[5].len=256u*meg;

    _MEMCHECK(heapBig); free(heapBig);
    _MEMCHECK(heapLil); free(heapLil); 
    
    /*Align each memory region*/
    for (i=0;i<nRegions;i++) {
      memRange_t p=(memRange_t)regions[i].start;
      p&=~(regions[i].len-1); /*Round down to a len-boundary (mask off low bits)*/
      regions[i].start=(char *)p;
#if CMK_THREADS_DEBUG
    CmiPrintf("[%d] Memory map: %p - %p  %s\n",CmiMyPe(),
	      regions[i].start,regions[i].start+regions[i].len,regions[i].type);
#endif
    }
    
    /*Find the largest unused region*/
    freeRegion=find_free_region(regions,nRegions,(512u+256u)*meg);

    /*If the unused region is very large, pad it on both ends for safety*/
    if (freeRegion.len/gig>64u) {
      freeRegion.start+=16u*gig;
      freeRegion.len-=20u*gig;
    }

#if CMK_THREADS_DEBUG
    CmiPrintf("[%d] Largest unused region: %p - %p (%d megs)\n",CmiMyPe(),
	      freeRegion.start,freeRegion.start+freeRegion.len,
	      freeRegion.len/meg);
#endif

    /*Allocate stacks in unused region*/
    isomallocStart=freeRegion.start;
    isomallocEnd=freeRegion.start+freeRegion.len;
    numslots=(freeRegion.len/slotsize)/CmiNumPes();

#if CMK_THREADS_DEBUG
    CmiPrintf("[%d] Can isomalloc up to %d megs per pe\n",CmiMyPe(),
	      numslots*slotsize/1024/1024);
#endif
  }

  CpvInitialize(slotset *, myss);
  CpvAccess(myss) = new_slotset(pe2slot(CmiMyPe()), numslots);
}


/************* Communication: for grabbing/freeing remote slots *********/
typedef struct _slotmsg
{
  char cmicore[CmiMsgHeaderSizeBytes];
  int pe; /*Source processor*/
  int slot; /*First requested slot*/
  int nslots; /*Number of requested slots*/
} slotmsg;

static slotmsg *prepare_slotmsg(int slot,int nslots)
{
	slotmsg *m=(slotmsg *)CmiAlloc(sizeof(slotmsg));
	m->pe=CmiMyPe();
	m->slot=slot;
	m->nslots=nslots;
	return m;
}

static void grab_remote(slotmsg *msg)
{
	grab_slots(CpvAccess(myss),msg->slot,msg->nslots);
	CmiFree(msg);
}

static void free_remote(slotmsg *msg)
{
	free_slots(CpvAccess(myss),msg->slot,msg->nslots);
	CmiFree(msg);
}
static int grab_remote_idx, free_remote_idx;

struct slotOP {
	/*Function pointer to perform local operation*/
	void (*local)(slotset *ss,int s,int n);
	/*Index to perform remote operation*/
	int remote;
};
typedef struct slotOP slotOP;
static slotOP grabOP,freeOP;

static void init_comm(char **argv)
{
	grab_remote_idx=CmiRegisterHandler((CmiHandler)grab_remote);
	free_remote_idx=CmiRegisterHandler((CmiHandler)free_remote);	
	grabOP.local=grab_slots;
	grabOP.remote=grab_remote_idx;
	freeOP.local=free_slots;
	freeOP.remote=free_remote_idx;	
}

/*Apply the given operation to the given slots which
  lie on the given processor.*/
static void one_slotOP(const slotOP *op,int pe,int s,int n)
{
/*Shrink range to only those covered by this processor*/
	/*First and last slot for this processor*/
	int p_s=pe2slot(pe), p_e=pe2slot(pe+1);
	int e=s+n;
	if (s<p_s) s=p_s;
	if (e>p_e) e=p_e;
	n=e-s;

/*Send off range*/
	if (pe==CmiMyPe()) 
		op->local(CpvAccess(myss),s,n);
	else 
	{/*Remote request*/
		slotmsg *m=prepare_slotmsg(s,e);
		CmiSyncSendAndFree(pe,sizeof(slotmsg),m);
	}
}

/*Apply the given operation to all slots in the range [s, s+n) 
After a restart from checkpoint, a slotset can cross an 
arbitrary set of processors.
*/
static void all_slotOP(const slotOP *op,int s,int n)
{
	int spe=slot2pe(s), epe=slot2pe(s+n-1);
	int pe;
	for (pe=spe; pe<=epe; pe++)
		one_slotOP(op,pe,s,n);
}

/************** External interface ***************/
void *CmiIsomalloc(int size,CmiIsomallocBlock *b)
{
	int s,n;
	n=length2slots(size);
	/*Always satisfy mallocs with local slots:*/
	s=get_slots(CpvAccess(myss),n);
	if (s==-1) {
		CmiError("Not enough address space left on processor %d to isomalloc %d bytes!\n",
			 CmiMyPe(),size);
		CmiAbort("Out of virtual address space for isomalloc");
	}
	grab_slots(CpvAccess(myss),s,n);
	b->slot=s;
	b->nslots=n;
	return map_slots(s,n);
}

void *CmiIsomallocPup(pup_er p,CmiIsomallocBlock *b)
{
	int s,n;
	pup_int(p,&b->slot);
	pup_int(p,&b->nslots);
	s=b->slot, n=b->nslots;
	if (pup_isUnpacking(p)) 
	{ /*Allocate block in its old location*/
		if (pup_isUserlevel(p))
			/*Checkpoint: must grab old slots (even remote!)*/
			all_slotOP(&grabOP,s,n);
		map_slots(s,n);
	}
	
	/*Pup the allocated data*/
	pup_bytes(p,slot2addr(s),slots2length(n));

	if (pup_isDeleting(p)) 
	{ /*Unmap old slots, but do not mark as free*/
		unmap_slots(s,n);
		b->nslots=0;/*Mark as unmapped*/
	}
	return slot2addr(s);
}

void CmiIsomallocFree(CmiIsomallocBlock *b)
{
	int s=b->slot, n=b->nslots;
	if (n==0) return;
	unmap_slots(s,n);
	/*Mark used slots as free*/
	all_slotOP(&freeOP,s,n);
}

/*Return true if this address is in the region managed by isomalloc*/
int CmiIsomallocInRange(void *addr)
{
	return (isomallocStart<=((char *)addr)) && (((char *)addr)<isomallocEnd);
}

void CmiIsomallocInit(char **argv)
{
  init_comm(argv);
  init_ranges(argv);
  init_map(argv);
}

