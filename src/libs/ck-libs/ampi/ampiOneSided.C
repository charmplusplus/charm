/******************************************************************
 * File: ampiOneSided.C
 *       This file contains one-sided communication functions
 *       win_obj class definition and AMPI_* implementations.
 ******************************************************************/

#include "ampiimpl.h"
extern "C" void applyOp(MPI_Datatype datatype, MPI_Op op, int count, void* a, void* b);
extern ampi *getAmpiInstance(MPI_Comm comm);
extern ampiParent *getAmpiParent(void);

/*************************************************************
 * Local flags used for Win_obj class: 
 *     WIN_ERROR -- the operation fails
 *     WIN_SUCCESS -- the operation succeeds
 *************************************************************/
#define WIN_SUCCESS 	0
#define WIN_ERROR 	(-1)

win_obj::win_obj() {
  winName = NULL;
  winNameLeng = 0;
  baseAddr = NULL;
  comm = MPI_COMM_NULL; 
  initflag = 0;
}


win_obj::win_obj(char *name, void *base, MPI_Aint size, int disp_unit, 
			 MPI_Comm comm) {
  create(name, base, size, disp_unit, comm);
  //ampi *ptr = getAmpiInstance(comm);
  owner = -1;  // the lock is not owned by anyone yet
}

void 
win_obj::setName(const char *src,int len) {
  if(winName==NULL) winName=new char[MPI_MAX_OBJECT_NAME];
  winNameLeng = len;
  memcpy(winName,src,len);
  winName[len] = '\0';
}

void 
win_obj::getName(char *name, int *len) {
  if(winName==NULL){
    name = NULL;
    *len = 0;
    return;
  }
  *len = winNameLeng; 
  memcpy(name, winName, *len+1);
}

win_obj::~win_obj() {
  free();  
}


// Note that this is supposed to be used for migration. 
// We should not hava a remote methos which has to pack the win data --- Inefficient
void
win_obj::pup(PUP::er &p) {
#if 0
  p|winSize;
  p|disp_unit;
  p|comm;
  p|initflag; 
  
  int len = 0;
  if(winName) len = strlen(winName)+1;
  p|len;
  if(p.isUnpacking()) winName = new char[len+1];
  p(winName, len);

  int size = 0;
  if(baseAddr) size = winSize;
  p|size;
  if(p.isUnpacking()) baseAddr = new char[size+1];
  p(baseAddr, size);
#endif
}


int win_obj::create(char *name, void *base, MPI_Aint size, int disp_unit, MPI_Comm comm){
  winName = NULL;
  baseAddr = base;
  winSize = size;
  this->disp_unit = disp_unit;
  this->comm = comm;
  // assume : memory pointed by base has been allocated 
  initflag = 1;
  return WIN_SUCCESS;
}


int win_obj::free(){
  if(winName!=NULL) {delete[] winName; winName = NULL;}
  // Assume : memory will be deallocated by user
  initflag = 0;
  return WIN_SUCCESS;
}

// ???? How to deal with different datatypes and 
//      How to deal with same datatype on different platforms?

// This is a local function. 
// AMPI_Win_put will act as a wrapper: pack the input parameters, copy the 
//   remote data to local, and call this function of the involved WIN object
int win_obj::put(void *orgaddr, int orgcnt, MPI_Datatype orgtype, 
		 MPI_Aint targdisp, int targcnt, MPI_Datatype targtype) {
  if(initflag == 0) {
    CkAbort("Put to non-existing MPI_Win\n");
    return WIN_ERROR;
  }
  
  if((targdisp+targcnt*sizeof(targtype)) > (winSize)){
    CkAbort("Put size exceeds MPI_Win size\n");
    return WIN_ERROR;
  }
  
  memcpy((int*)baseAddr+targdisp, orgaddr,targcnt*sizeof(targtype));
  return WIN_SUCCESS;
}

int win_obj::get(void *orgaddr, int orgcnt, MPI_Datatype orgtype, 
		 MPI_Aint targdisp, int targcnt, MPI_Datatype targtype){
  if(initflag == 0) {
    CkAbort("Get from non-existing MPI_Win\n");
    return WIN_ERROR;
  }

  if((targdisp+targcnt*sizeof(targtype)) > (winSize)){
    CkAbort("Get size exceeds MPI_Win size\n");
    return WIN_ERROR;
  }
  // Call the RMA operation here!!!     
  memcpy(orgaddr, (int*)baseAddr+targdisp, orgcnt*sizeof(orgtype));

  return WIN_SUCCESS;
}

int 
win_obj::accumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype, 
		    MPI_Aint targdisp, int targcnt, 
   	            MPI_Datatype targtype, MPI_Op op){
  applyOp(targtype, op, targcnt, (void*)((int*)baseAddr+targdisp) , (void*)orgaddr);
  return WIN_SUCCESS;
}

int win_obj::fence(){
  return WIN_SUCCESS;
}

int win_obj::lock(int requestRank, int pe_src, int ftHandle, int lock_type){
  owner = requestRank;

  int tmp = 0;
  AmpiMsg *msg = new (&tmp, 0) AmpiMsg(-1, -1, -1, 0, 0, comm);
  CkSendToFuture(ftHandle, (void *)msg, pe_src);
  
  return WIN_SUCCESS;
}


int win_obj::unlock(int requestRank, int pe_src, int ftHandle){
  if (owner != requestRank){
    CkPrintf("    ERROR: Can't unlock a lock which you don't own.\n");
    return WIN_ERROR;
  }  
  owner = -1;

  // dequeue from queue itself
  dequeue();
 
  int tmp = 0;
  AmpiMsg *msg = new (&tmp, 0) AmpiMsg(-1, -1, -1, 0, 0, comm);
  CkSendToFuture(ftHandle, (void *)msg, pe_src); 
  
  return WIN_SUCCESS;
}

void win_obj::dequeue() {
  lockQueueEntry *lq = lockQueue.deq();
  delete lq;
}

void win_obj::enqueue(int requestRank, int pe_src, int ftHandle, int lock_type) {
  lockQueueEntry *lq = new lockQueueEntry(requestRank, pe_src, ftHandle, lock_type);
  lockQueue.enq(lq);
}

bool win_obj::emptyQueue() {
  return (lockQueue.length()==0) ; 
}

void win_obj::lockTopQueue() {
  lockQueueEntry *lq = lockQueue.deq();
  lock(lq->requestRank, lq->pe_src, lq->ftHandle, lq->lock_type);
  lockQueue.insert(0, lq);
}

/* these four functions are yet to implement */
int win_obj::wait(){
  return -1;
}
int win_obj::post(){
  return -1;
}
int win_obj::start(){
  return -1;
}
int win_obj::complete(){
  return -1;
}

/*
 * int AMPI_Win_create(void *base, MPI_Aint size, int disp_unit,
 *	       MPI_Info info, MPI_Comm comm, MPI_Win *newwin)
 *   Creates the window object and returns the pointer for *win
 *
 *   ---Assumption: memory location at *base is pre-allocated 
 *   ---by a MPI_Alloc_mem call
 *
 *   Inputs:
 *     void *base : pointer specifying the memory area to create the window
 *     MPI_Aint size : size of target memory area (in bytes)
 *     int disp_unit : number of bytes for one datatype
 *     MPI_Info info : MPI_Info object, provides hints for optimization
 *     MPI_Comm comm : communicator 
 *     MPI_Win *newwin : stores the handle to the created MPI_Win object on return
 *  
 *   Returns int: MPI_SUCCESS or MPI_ERR_WIN
 */
// A collective call over all processes in the communicator
// MPI_Win object created LOCALLY on all processes when the call returns
CDECL
int AMPI_Win_create(void *base, MPI_Aint size, int disp_unit,
	       MPI_Info info, MPI_Comm comm, MPI_Win *newwin){
  AMPIAPI("AMPI_Win_create");
  ampi *ptr = getAmpiInstance(comm);
  *newwin = ptr->createWinInstance(base, size, disp_unit, info);
  /* need to reduction here: to make sure every processor participates */
  AMPI_Barrier(comm);
  return MPI_SUCCESS;
}

/*
 * int AMPI_Win_free(MPI_Win *win):
 *   Frees the window object and returns a null pointer for *win
 */
// A collective call over all processes in the communicator
// MPI_Win object deleted LOCALLY on all processes when the call returns
CDECL
int AMPI_Win_free(MPI_Win *win){
  AMPIAPI("AMPI_Win_free");
  if(win==NULL) { return MPI_ERR_WIN; }

  WinStruct winStruct = getAmpiParent()->getWinStruct(*win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  ptr->deleteWinInstance(*win);
  /* Need a barrier here: to ensure that every process participates */
  AMPI_Barrier(winStruct.comm);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_delete_attr(MPI_Win win, int key){
  AMPIAPI("AMPI_Win_delete_attr");
  return MPI_SUCCESS;
}

/*
 * ---Note : No atomicity for overlapping Puts. 
 * ---sync calls should be made on this window to ensure the 
 * ---correctness of the operation
 */	
CDECL
int AMPI_Put(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank, 
	MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, MPI_Win win){
  AMPIAPI("AMPI_Put");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  return ptr->winPut(orgaddr, orgcnt, orgtype, rank, targdisp, targcnt, targtype, winStruct);
}

/*
 * ---Note : No atomicity for overlapping Gets. 
 * ---sync calls should be made on this window to ensure the 
 * ---correctness of the operation
 */
CDECL
int AMPI_Get(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank, 
	MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, MPI_Win win){
  AMPIAPI("AMPI_Get");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  // winGet is a local function which will call the remote method on #rank processor 
  return  ptr->winGet(orgaddr, orgcnt, orgtype, rank, targdisp, targcnt, targtype, winStruct);
}

/*
 * int AMPI_Accumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
 *		   MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, 
 *		   MPI_Op op, MPI_Win win)
 *   Accumulates the contents from the origin buffer to the target area using
 *   the predefined op operation.
 *
 * ---Accumulate call is ATOMIC: no sync is needed 
 * ---Many accumulate can be made from many origins to one target
 */
CDECL
int AMPI_Accumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
		   MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, 
		   MPI_Op op, MPI_Win win) {
  AMPIAPI("AMPI_Accumulate");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);  		   
  return  ptr->winAccumulate(orgaddr, orgcnt, orgtype, rank, 
                         targdisp, targcnt, targtype, op, winStruct);   
}


/*
 * int AMPI_Win_fence(int assertion, MPI_Win win)
 *   Synchronizes all one-sided communication calls on this MPI_Win.   
 *   (Synchronized RMA operations on the specified window)
 *
 *   Inputs: 
 *     int assertion : program assertion, used to provide optimization hints
 *   Returns int : MPI_SUCCESS or MPI_ERR_WIN
 */
CDECL
int AMPI_Win_fence(int assertion, MPI_Win win){
  AMPIAPI("AMPI_Win_fence");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  MPI_Comm comm = winStruct.comm;
  ampi *ptr = getAmpiInstance(comm);

  // Wait until everyone reaches the fence
  AMPI_Barrier(comm);

  // Complete all outstanding one-sided comm requests
  // no need to do this for the pseudo-implementation
  return MPI_SUCCESS;
}


/*
 * int AMPI_Win_lock(int lock_type, int rank, int assertion, MPI_Win win)
 *   Locks access to this MPI_Win object.   
 *   Input:
 *     int lock_type : MPI_LOCK_EXCLUSIVE or MPI_LOCK_SHARED
 *     int rank : rank of locked window
 *     int assertion : program assertion, used to provide optimization hints
 *   Returns int : MPI_SUCCESS or MPI_ERR_WIN
 */
CDECL
int AMPI_Win_lock(int lock_type, int rank, int assertion, MPI_Win win){
  AMPIAPI("AMPI_Win_lock");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);

  // process assertion here:   
  // end of assertion
  ptr->winLock(lock_type, rank, winStruct);
  return MPI_SUCCESS;
}



/*
 * int AMPI_Win_unlock(int rank, MPI_Win win)
 *   Unlocks access to this MPI_Win object.   
 *   Input:
 *     int rank : rank of locked window
 *   Returns int : MPI_SUCCESS or MPI_ERR_WIN
 */
// The RMA call is completed both locally and remotely after unlock. 
  // process assertion here: HOW???  
int AMPI_Win_unlock(int rank, MPI_Win win){
  AMPIAPI("AMPI_Win_unlock");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);

  // process assertion here: HOW???  
  // end of assertion
  ptr->winUnlock(rank, winStruct);
  return MPI_SUCCESS;
}

/* the following four functions are yet to implement */
/*
 * int AMPI_Win_post(MPI_Group group, int assertion, MPI_Win win)
 *   Opens a RMA access epoch for local window win.
 *   Only processes in group can access this window with RMA calls.
 *   Each process must issue a matching MPI_Win_start to start the 
 *     access epoch.
 *   Post is non-blocking while start could be blocking.
 *   Input:
 *     MPI_Group group : a group of processes 
 *   Returns int : MPI_SUCCESS or MPI_ERR_WIN
 */
CDECL
int AMPI_Win_post(MPI_Group group, int assertion, MPI_Win win){
  AMPIAPI("AMPI_Win_post");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  //  ptr->winPost(group, winStruct);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_wait(MPI_Win win){
  AMPIAPI("AMPI_Win_wait");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_start(MPI_Group group, int assertion, MPI_Win win){
  AMPIAPI("AMPI_Win_start");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  //  ptr->winStart(group, winStruct);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_complete(MPI_Win win){
  AMPIAPI("AMPI_Win_complete");
  return MPI_SUCCESS;
}


/*
 * int AMPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr) 
 *   A simple wrapper around 'malloc' call. Used to allocate memory
 *   for MPI functions
 *
 *   Inputs: 
 *     MPI_Aint size : size of target memory area
 *     MPI_Info info : 
 *     void *base : stores the base pointer of target memory area on return
 *   Return: 
 *     void* : address of the allocated memory
 */
CDECL
int AMPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr){
  AMPIAPI("AMPI_Alloc_mem");
  *(void **)baseptr = malloc(size);
  return MPI_SUCCESS;
}


/*
 * int AMPI_Free_mem(void *base)
 *   Frees memory that was previous allocated by MPI_Alloc_mem call
 */
CDECL
int AMPI_Free_mem(void *baseptr){
  AMPIAPI("AMPI_Free_mem");
  free(baseptr);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_get_group(MPI_Win win, MPI_Group *group) {
  AMPIAPI("AMPI_Win_get_group");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  ptr->winGetGroup(winStruct, group);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_set_attr(MPI_Win win, int key, void* value) {
  AMPIAPI("AMPI_Win_set_attr");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_set_name(MPI_Win win, char *name) {
  AMPIAPI("AMPI_Win_set_name");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  ptr->winSetName(winStruct, name);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_get_name(MPI_Win win, char *name, int *length) {
  AMPIAPI("AMPI_Win_get_name");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  ptr->winGetName(winStruct, name, length);
  return MPI_SUCCESS;
}

