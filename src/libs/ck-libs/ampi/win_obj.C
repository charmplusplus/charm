/******************************************************************************
 * File: win_obj.C
 *       This file implements the win_obj class. 
 * Author: Yan 
 * Last Revision: 2003/10/30 
 ******************************************************************************/

#include "ampiimpl.h"
extern "C" void applyOp(MPI_Datatype datatype, MPI_Op op, int count, void* a, void* b);


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
  ampi *ptr = getAmpiInstance(comm);
  owner = -1;  // the lock is not owned by anyone yet
  lockQueue = new LockQueue();
}

void 
win_obj::setName(const char *src,int len) {
  winNameLeng = len;
  winName=new char[len+1];
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


int win_obj::create(char *name, void *base, MPI_Aint size, int disp_unit, 
		    MPI_Comm comm){

//  setName(name, strlen(name));
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
  
  delete lockQueue;
  
  initflag = 0;
  return WIN_SUCCESS;

}

// ???? How to deal with different datatypes and 
//      How to deal with same datatype on different platforms?

// This is a local function. 
// MPI_Win_put will act as a wrapper: pack the input parameters, copy the 
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
  ((LockQueue*)lockQueue)->deq();
}

void win_obj::enqueue(int requestRank, int pe_src, int ftHandle, int lock_type) {
  
  lockQueueEntry *lq = new lockQueueEntry(requestRank, pe_src, ftHandle, lock_type);
  ((LockQueue*)lockQueue)->enq(*lq);
  
}

bool win_obj::emptyQueue() {
  return (lockQueue->length()==0) ; 
}

void win_obj::lockTopQueue() {
  lockQueueEntry lq = (lockQueueEntry)lockQueue->deq();
  CkPrintf("          : to [%d]\n", lq.requestRank);
  lock(lq.requestRank, lq.pe_src, lq.ftHandle, lq.lock_type);
  lockQueue->insert(0, lq);
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

