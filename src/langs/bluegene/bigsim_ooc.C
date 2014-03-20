#include "blue.h"
#include "blue_types.h"
#include "bigsim_ooc.h"
#include <assert.h>
#include "bigsim_debug.h"

#undef DEBUGLEVEL
#define DEBUGLEVEL 3


//default values for out-of-core execution
int bgUseOutOfCore = 0; //default is off

double bgOOCMaxMemSize = 512.0; //default 512MB

#if BIGSIM_OUT_OF_CORE && BIGSIM_OOC_PREFETCH
oocPrefetchStatus *thdsOOCPreStatus = NULL;
oocPrefetchBufSpace *oocPrefetchSpace = NULL;
oocWorkThreadQueue *schedWorkThds = NULL;
#endif

threadInMemEntry *tblThreadInMemHead;

int TBLCAPACITY=1;

#define OOCSIMPLEVER 1 

#if OOCSIMPLEVER
void initTblThreadInMem(){ 
  
  /* Initialize the table that tracks threads (workThread) in memory:
   * First, accordin g to current memory usage, decide an appropriate table size
   * which should be less than TBLTHDINMEMMAXSIZE.
   * NOTE: This has not been implemented. The initial step is only set the table
   * to a small constant!!!
   * Second, initialize the table entries.
   */  
  tblThreadInMemHead = new threadInMemEntry[TBLCAPACITY];  
  for(int i=1; i<TBLCAPACITY; i++){
      tblThreadInMemHead[i].initSelf();
  }
}

void deInitTblThreadInMem(){    
    delete tblThreadInMemHead;
}

//if found return the entry of this thread, otherwise return NULL
threadInMemEntry *checkThreadInCore(/*int threadID*/threadInfo *thd){
    for(int i=0; i<TBLCAPACITY; i++){
        if(tblThreadInMemHead[i].thd == thd)
            return tblThreadInMemHead+i;
    }
    return NULL;
}

//clear all entries
void clearThdInMemTbl(){
    for(int i=0; i<TBLCAPACITY; i++)
	tblThreadInMemHead[i].thd = NULL;
}

/* 
 * Gives one empty entry for the thread (thd)  which is going to be put into memory
 * This includes following actions:
 * 1. Check tblThreadInMem for an empty entry
 * 2. If none is found, evict one entry based on useFreq. 
 * Note the evicted thread will not be taken out of memory until function takeThdOutofMem is called!!
 */
threadInMemEntry *giveEmptyThdInMemEntry(){

    for(int i=0; i<TBLCAPACITY; i++){
        if(tblThreadInMemHead[i].thd == NULL)
            return tblThreadInMemHead+i;
    }

    //Here, we know there is no empty entry in the table   
    return findLeastUsedThdInMemEntry();
}

/* Just find the least used entry */
threadInMemEntry *findLeastUsedThdInMemEntry(){
    threadInMemEntry *leastAccessedEntry=tblThreadInMemHead;
    assert(leastAccessedEntry->thd);
    int leastFreq = leastAccessedEntry->useFreq;
    
    for(int i=1; i<TBLCAPACITY; i++){
        threadInMemEntry *p = tblThreadInMemHead + i;
        assert(p->thd);        
        if(p->useFreq < leastFreq){
            leastAccessedEntry = p;
            leastFreq = p->useFreq;
        }
    }
    
    return leastAccessedEntry;
}

/* Find and delete the least accessed entry */
threadInMemEntry *detachLeastUsedThdInMemEntry(){
    return NULL;
}

void bgOutOfCoreSchedule(threadInfo *curThd){
    int outThdID=-1;    

    if(!curThd->startOutOfCore){
        DEBUGM(4, ("to execute not in ooc mode\n"));
	clearThdInMemTbl();
        return;
    }

    if(!checkThreadInCore(curThd)){   

        DEBUGM(4, ("to execute in ooc mode\n"));

        threadInMemEntry *thdEntry = giveEmptyThdInMemEntry();

        threadInfo *toBeSwapped = thdEntry->thd;
        if(toBeSwapped!=NULL){ //the returned entry is not empty thus needing eviction

            if(toBeSwapped->startOOCChanged){
                //indicate AMPI_Init is called and before it is finished, out-of-core should not 
                //happen for this thread
                //just to track the 0->1 change phase (which means MPI_Init is finished)
                //the 1->0 phase is not tracked because "startOutOfCore" is unset so that
                //the next processing of a msg will not go into this part of code
                toBeSwapped->startOOCChanged=0;
            }else{                                
                assert(toBeSwapped->isCoreOnDisk==0);
                outThdID=toBeSwapped->globalId;
                CmiSwitchToPE(outThdID);
                toBeSwapped->takenOutofMem();
                CmiSwitchToPE(curThd->globalId);
                
                DEBUGM(4, ("Taking thread %d out, bring thread %d in\n", outThdID, curThd->globalId));                        
            }
        }
        thdEntry->thd = curThd;

        //if this thread's core has been dumped to disk, then we need to bring it back.
        //otherwise, it is the first time for this thread to process a message, thus
        //no need to resort to disk to find its core
        if(curThd->isCoreOnDisk){                        
            curThd->broughtIntoMem();            
        }
        updateThdInMemTable(curThd);

        //DEBUGF(("BG > current NOT in core\n"));
        //if(emptyEntry) printTblThdInMem();
        //printTblThdInMem();
    }else{
        DEBUGM(4, ("to execute not in ooc mode\n"));
    }

    /*//Output Stats
    printf("In thread %d\n", globalId);
    if(outThdID!=-1) printf("Take thread %d out\n", outThdID);
    printTblThdInMem();    
    */
}

/* Set the thread indicated by threadID to the most recently accessed one */
void updateThdInMemTable(/*int threadID*/threadInfo *thd){
    threadInMemEntry *p = tblThreadInMemHead;
    for(int i=0; i<TBLCAPACITY; i++, p++){
        if(p->thd==thd)
            p->useFreq = MOSTRECENTACCESSED;
        else
            (p->useFreq)--;        
    } 
}

void printTblThdInMem(){
    printf("====thread in memory table stats=======\n");
    threadInMemEntry *p = tblThreadInMemHead;
    for(int i=0; i<TBLCAPACITY; i++, p++){
        CmiPrintf("entry %d: thread global id: %d, usefreq: %d\n", i+1, (p->thd)->globalId, p->useFreq);
    }
    CmiPrintf("\n");
}

#else //more complex version
//This version decides the number of target processors images staying 
//in the memory according to the current memory available.    

void initTblThreadInMem(){ 
  tblThreadInMemHead = NULL;
}

//clearing all entries is the same with
//free the space for this table
void deInitTblThreadInMem(){
    threadInMemEntry *p = tblThreadInMemHead;
    while(p){
        threadInMemEntry *tmp = p;        
        p = p->nextEntry;
        delete tmp;
    }
    tblThreadInMemHead = NULL;    
}

//if found return the entry of this thread, otherwise return NULL
threadInMemEntry *checkThreadInCore(/*int threadID*/threadInfo *thd){
    threadInMemEntry *p = tblThreadInMemHead;
    while(p){
        if(p->thd == thd)
            return p;
        p = p->nextEntry;
    }
    return NULL;
}

/* 
 * Gives one empty entry for the thread (thd)  which is going to be put into memory
 * This includes following actions:
 * 1. Check tblThreadInMem for an empty entry
 * 2. If none is found, evict one entry based on useFreq. 
 * Note the evicted thread will not be taken out of memory until function takeThdOutofMem is called!!
 */
threadInMemEntry *giveEmptyThdInMemEntry(){

    if(tblThreadInMemHead==NULL){
        tblThreadInMemHead = new threadInMemEntry();        
        return tblThreadInMemHead;
    }
    
    threadInMemEntry *returnEntry = detachLeastUsedThdInMemEntry();
    /*    //add a new entry the table
        returnEntry = new threadInMemEntry();
        returnEntry->nextEntry = tblThreadInMemHead;
        tblThreadInMemHead = returnEntry;
   */

    return returnEntry;
}

threadInMemEntry *addNewThdInMemEntry(threadInfo *curThd){
    if(tblThreadInMemHead==NULL){
        tblThreadInMemHead = new threadInMemEntry();        
	tblThreadInMemHead->thd = curThd;
        return tblThreadInMemHead;
    }
    
    threadInMemEntry *returnEntry = new threadInMemEntry();
    returnEntry->nextEntry = tblThreadInMemHead;
    tblThreadInMemHead = returnEntry;
    returnEntry->thd = curThd;
    return returnEntry;

}

/* Just find the least used entry */
threadInMemEntry *findLeastUsedThdInMemEntry(){
    threadInMemEntry *leastAccessedEntry=tblThreadInMemHead;
    int leastFreq = leastAccessedEntry->useFreq;
    
    threadInMemEntry *cur = leastAccessedEntry->nextEntry;
    while(cur){        
        if(cur->useFreq < leastFreq){
            leastAccessedEntry = cur;
            leastFreq = cur->useFreq;            
        }        
        cur = cur->nextEntry;
    }    
    assert(leastAccessedEntry->thd);

    return leastAccessedEntry;
}

/* Find and delete the least accessed entry */
//If the least accessed thread is the curThd, then we don't detach it from the queue
threadInMemEntry *detachLeastUsedThdInMemEntry(){
    threadInMemEntry *leastAccessedEntry=tblThreadInMemHead;
    int leastFreq = leastAccessedEntry->useFreq;

    threadInMemEntry *leastPrev = NULL;
    threadInMemEntry *prev = leastAccessedEntry;
    threadInMemEntry *cur = leastAccessedEntry->nextEntry;
    while(cur){        
        if(cur->useFreq < leastFreq){
            leastAccessedEntry = cur;
            leastFreq = cur->useFreq;
            leastPrev = prev;
        }
        prev = cur;
        cur = cur->nextEntry;
    }    
    assert(leastAccessedEntry->thd);

    //detach the least accessed entry
    if(leastAccessedEntry == tblThreadInMemHead){        
        tblThreadInMemHead = tblThreadInMemHead->nextEntry;                
    }else{
        leastPrev->nextEntry = leastAccessedEntry->nextEntry;
    }

    return leastAccessedEntry;
}

void bgOutOfCoreSchedule(threadInfo *curThd){

//the memory eviction is based on the memory availabe now and
//the eviction uses LRU policy. The curThd may be the least accessed
//thd. In this case, if the out-of-core happens, curThd will be evicted
//out of the memory. To avoid this, we first update the access frequency of
//all threads in memory so that curThd must not be the least accessed.

    if(!curThd->startOutOfCore){
	DEBUGM(4, ("to execute not in ooc mode\n"));
	deInitTblThreadInMem();
	return;
    }    

    updateThdInMemTable(curThd);

    //double curMemUsage = CmiMemoryUsage()/1024.0/1024.0;
    double curMemUsage = bgGetProcessMemUsage();
    printf("Current physical memory used: %lf\n", curMemUsage);

    //the "evictThdCase" shows the two different cases:
    //0: the current thd is in core but finding the whole emulation
    //is going to run out of memory (in terms of bgOOCMaxMemSize), so 
    //another thd in core has to be evicted to free space
    //1: the current thd is not in core and finding not enough memory
    //to allow the current thd, so another thd in core has to be evicted..
    int evictThdCase = 0;
    if(checkThreadInCore(curThd)){
	assert(curThd->isCoreOnDisk==0);
	if(curMemUsage > bgOOCMaxMemSize){
            DEBUGM(1, ("BG > existing memory usage (%.3f) exceeds the limit!\n", curMemUsage));
	}
	if(curMemUsage < (1-LEASTMEMRATE)*bgOOCMaxMemSize) return;
	evictThdCase = 0;
    }else{
	//assert(curThd->isCoreOnDisk==1);
	//curThd->isCoreOnDisk maybe 0 because this "curThd" is the first thread
	//that will be recorded in the list of threads whose head is tblThreadInMemHead
	if(curMemUsage + curThd->memUsed < (1-LEASTMEMRATE)*bgOOCMaxMemSize) {
	    //add curThd to the thd tbl and there's no need to evict other thds
	    //besides bringing this thd into core if its core is on disk
	    threadInMemEntry *newEntry = addNewThdInMemEntry(curThd);
	    newEntry->useFreq = MOSTRECENTACCESSED;
	    if(curThd->isCoreOnDisk){
		curThd->broughtIntoMem();
	    }
	    return;
	}
	evictThdCase = 1;
    }

    int outThdID = -1;
    threadInMemEntry *thdEntry = giveEmptyThdInMemEntry();
    threadInfo *toBeSwapped = thdEntry->thd;
    if(toBeSwapped==curThd){
	//this means there's only one thread in core, we have to reinsert this entry
	//into the list of thds in memory
	thdEntry->thd = curThd;
	thdEntry->useFreq = MOSTRECENTACCESSED;
	thdEntry->nextEntry = tblThreadInMemHead;
	tblThreadInMemHead = thdEntry;
	return;
    }else if(toBeSwapped!=NULL){
	if(toBeSwapped->startOOCChanged){
            //indicate AMPI_Init is called and before it is finished, out-of-core should not 
            //happen for this thread
            //just to track the 0->1 change phase (which means MPI_Init is finished)
            //the 1->0 phase is not tracked because "startOutOfCore" is unset so that
            //the next processing of a msg will not go into this part of code
	    toBeSwapped->startOOCChanged = 0;
	    printf("After finishing MPI_Init, the mem footprint is: %lf\n", curMemUsage);
	}else{
	    assert(toBeSwapped->isCoreOnDisk==0);
	    outThdID = toBeSwapped->globalId;
	    CmiSwitchToPE(outThdID);
	    toBeSwapped->takenOutofMem();
	    CmiSwitchToPE(curThd->globalId);
	    DEBUGM(4, ("Taking thread %d out, bring thread %d in\n", outThdID, curThd->globalId));
	}
    }
    thdEntry->thd = curThd;
    thdEntry->useFreq = MOSTRECENTACCESSED;

    //the isCoreOnDisk may be 0 if evictThdCase is 0
    //or it's the first thread inserted into the thd table
    //after finishing MPI_Init
    if(curThd->isCoreOnDisk){
	assert(evictThdCase);
	curThd->broughtIntoMem();
    } 

    /*//Output Stats
    printf("In thread %d\n", globalId);
    if(outThdID!=-1) printf("Take thread %d out\n", outThdID);
    printTblThdInMem();    
    */
}

/* Set the thread indicated by threadID to the most recently accessed one */
void updateThdInMemTable(/*int threadID*/threadInfo *thd){
    threadInMemEntry *p = tblThreadInMemHead;
    while(p){
        if(p->thd==thd)
            p->useFreq = MOSTRECENTACCESSED;
        else
            (p->useFreq)--;
        p = p->nextEntry;
    } 
}

//=============helping functions===============
void printTblThdInMem(){
    printf("====thread in memory table stats=======\n");
    threadInMemEntry *p = tblThreadInMemHead;
    int cnt=1;
    while(p){
        if(p->thd){
            CmiPrintf("entry %d: thread global id: %d, usefreq: %d\n", cnt++, (p->thd)->globalId, p->useFreq);
        }
        p = p->nextEntry;
    }
    CmiPrintf("\n");
}

int getNumThdTblEntries(){
    threadInMemEntry *p = tblThreadInMemHead;
    int totalCnt=0;
    while(p){
	totalCnt++;
	p = p->nextEntry;
    }
    return totalCnt;
}

//the argument should range from 1 to #table entries.
threadInMemEntry *getNthThdEntry(int nth){
    if(nth<1) return NULL;
    threadInMemEntry *p = tblThreadInMemHead;
    for(int i=1; i<nth && p; i++, p=p->nextEntry);
    return p;
}
#endif



#if BIGSIM_OUT_OF_CORE && BIGSIM_OOC_PREFETCH
//=====functions related with prefetch using AIO=====
void oocPrefetchBufSpace::newPrefetch(workThreadInfo *wthd){

    //printf("Prefetch thread[%d] \n", wthd->globalId);

    //the (bufsize) of memory needed to bring this workthread into memory
    CmiUInt8 bufsize = thdsOOCPreStatus[wthd->preStsIdx].bufsize;
    if(bufsize > PREFETCHSPACESIZE) {
        PREFETCHSPACESIZE = (CmiUInt8)(bufsize*1.5);
        delete [] bufspace;
        bufspace = new char[PREFETCHSPACESIZE];        
    }
    usedBufSize = bufsize;

    occupiedThd = wthd;
    thdsOOCPreStatus[wthd->preStsIdx].isPrefetchFinished = 0;
    
    assert(wthd->isCoreOnDisk==1);

    char *dirname = "/tmp/CORE";
    char filename[128];
    sprintf(filename, "%s/%d.dat", dirname, wthd->globalId);
    
    //struct aiocb prefetchAioCb;    
    int fd = open(filename, O_RDONLY);
    if(fd<0) perror("Failed in opening file");
    
    memset((char *)&prefetchAioCb, 0, sizeof(struct aiocb));
    prefetchAioCb.aio_buf = bufspace;
    prefetchAioCb.aio_fildes = fd;
    prefetchAioCb.aio_nbytes = usedBufSize;
    prefetchAioCb.aio_offset = 0;

#define PTHREAD_CALLBACK 1
#if PTHREAD_CALLBACK
    //use pthread to do callback
    prefetchAioCb.aio_sigevent.sigev_notify = SIGEV_THREAD;
    prefetchAioCb.aio_sigevent.sigev_notify_function = prefetchFinishedHandler;
    prefetchAioCb.aio_sigevent.sigev_notify_attributes = NULL;
    prefetchAioCb.aio_sigevent.sigev_value.sival_ptr = &prefetchAioCb;
#else
    //use signal to notify
    struct sigaction sig_act;
    sigemptyset(&sig_act.sa_mask);
    sig_act.sa_flags = SA_SIGINFO;
    sig_act.sa_sigaction = prefetchFinishedSignalHandler;
    prefetchAioCb.aio_sigevent.sigev_notify = SIGEV_SIGNAL;
    prefetchAioCb.aio_sigevent.sigev_signo = SIGUSR1;
    prefetchAioCb.aio_sigevent.sigev_value.sival_ptr = &prefetchAioCb;
    sigaction(SIGIO, &sig_act, NULL); 
    
#endif

    int ret = aio_read(&prefetchAioCb);
    if(ret<0) perror("Failed in aio_read");
}

void prefetchFinishedHandler(sigval_t sigval){
        struct aiocb *req = (struct aiocb *)sigval.sival_ptr;
        int ret = aio_return(req);
        close(req->aio_fildes);
        if(ret<=0) {
            perror("Failed in aio operation");
        }else{
            int idx = oocPrefetchSpace->occupiedThd->preStsIdx;
            thdsOOCPreStatus[idx].isPrefetchFinished = 1;
        }
}

void prefetchFinishedSignalHandler(int signo, siginfo_t *info, void *context){
    if(info->si_signo == SIGUSR1){
    struct aiocb *req = (struct aiocb *)info->si_value.sival_ptr;
    int ret = aio_return(req);
    close(req->aio_fildes);
    if(ret<=0) {
        perror("Failed in aio operation");
    }else{
        int idx = oocPrefetchSpace->occupiedThd->preStsIdx;
        thdsOOCPreStatus[idx].isPrefetchFinished = 1;
    }
    }
    return;
}
#endif

//=====functions related with system memory information=====
//the unit is MB
double bgGetSysTotalMemSize(){
    FILE *memf = fopen(MEMINFOFILE, "r");
    int totalM = 0;
    fscanf(memf, "MemTotal: %dkB\n", &totalM);
    fclose(memf);
    return totalM/1024.0;
}

double bgGetSysFreeMemSize(){
    FILE *memf = fopen(MEMINFOFILE, "r");
    int freeM = 0;
    fscanf(memf, "MemTotal: %dkB\n", &freeM);
    fscanf(memf, "MemFree: %dkB\n", &freeM);
    fclose(memf);
    return freeM/1024.0;
}

int bgMemPageSize;
char bgMemStsFile[25];

double bgGetProcessMemUsage(){
    FILE *memf = fopen(bgMemStsFile, "r");
    long progsize, memused;
    double retval;

    fscanf(memf, "%ld %ld", &progsize, &memused);
    retval = (double)memused/1024.0*bgMemPageSize/1024.0;
    
    fclose(memf);
    return retval;

}

