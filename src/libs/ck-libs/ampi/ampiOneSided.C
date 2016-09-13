/*************************************************************
 * File: ampiOneSided.C
 *       This file contains one-sided communication functions
 *       win_obj class definition and AMPI_* implementations.
 *************************************************************/

#include "ampiimpl.h"

/*************************************************************
 * Local flags used for Win_obj class:
 *     WIN_ERROR -- the operation fails
 *     WIN_SUCCESS -- the operation succeeds
 *************************************************************/
#define WIN_SUCCESS 0
#define WIN_ERROR   (-1)

win_obj::win_obj() {
  winNameLen = 0;
  baseAddr = NULL;
  comm = MPI_COMM_NULL;
  initflag = 0;
}

win_obj::win_obj(const char *name, void *base, MPI_Aint size, int disp_unit,
                 MPI_Comm comm) {
  create(name, base, size, disp_unit, comm);
  owner = -1;  // the lock is not owned by anyone yet
}

void win_obj::setName(const char *src) {
  CkDDT_SetName(winName, src, &winNameLen);
}

void win_obj::getName(char *name, int *len) {
  *len = winNameLen;
  memcpy(name, winName, *len+1);
}

win_obj::~win_obj() {
  free();
}

// Note that this is supposed to be used for migration.
// We should not hava a remote methos which has to pack the win data --- Inefficient
void win_obj::pup(PUP::er &p) {
#if 0
  p|winSize;
  p|disp_unit;
  p|comm;
  p|initflag;

  p|winNameLen;
  p(winName,MPI_MAX_OBJECT_NAME);

  int size = 0;
  if(baseAddr) size = winSize;
  p|size;
  if(p.isUnpacking()) baseAddr = new char[size+1];
  p(baseAddr, size);
#endif
}

int win_obj::create(const char *name, void *base, MPI_Aint size, int disp_unit, MPI_Comm comm){
  if (name) setName(name);
  baseAddr = base;
  winSize = size*disp_unit;
  this->disp_unit = disp_unit;
  this->comm = comm;
  // assume : memory pointed by base has been allocated
  initflag = 1;
  return WIN_SUCCESS;
}

int win_obj::free(){
  // Assume : memory will be deallocated by user
  initflag = 0;
  return WIN_SUCCESS;
}

// This is a local function.
// AMPI_Win_put will act as a wrapper: pack the input parameters, copy the
//   remote data to local, and call this function of the involved WIN object
int win_obj::put(void *orgaddr, int orgcnt, int orgunit, MPI_Aint targdisp,
                 int targcnt, int targunit) {
  if(initflag == 0) {
    CkAbort("Put to non-existing MPI_Win\n");
    return WIN_ERROR;
  }
  int totalsize = targdisp+targcnt*targunit;
  if(totalsize > (winSize)){
    CkAbort("Put size exceeds MPI_Win size\n");
    return WIN_ERROR;
  }

  return WIN_SUCCESS;
}

int win_obj::get(void *orgaddr, int orgcnt, int orgunit, MPI_Aint targdisp,
                 int targcnt, int targunit){
  if(initflag == 0) {
    CkAbort("Get from non-existing MPI_Win\n");
    return WIN_ERROR;
  }
  int totalsize = targdisp+targcnt*targunit;
  if(totalsize > (winSize)){
    CkAbort("Get size exceeds MPI_Win size\n");
    return WIN_ERROR;
  }
  // Call the RMA operation here!!!

  return WIN_SUCCESS;
}

int win_obj::iget(int orgcnt, int orgunit, MPI_Aint targdisp,
                  int targcnt, int targunit){
  if(initflag == 0) {
    CkAbort("Get from non-existing MPI_Win\n");
    return WIN_ERROR;
  }

  if((targdisp+targcnt*targunit) > (winSize)){
    CkAbort("Get size exceeds MPI_Win size\n");
    return WIN_ERROR;
  }
  // Call the RMA operation here!!!

  return WIN_SUCCESS;
}

int win_obj::accumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype,
                        MPI_Aint targdisp, int targcnt,
                        MPI_Datatype targtype, MPI_Op op){
  applyOp(targtype, op, targcnt, (void*)((int*)baseAddr+targdisp) , (void*)orgaddr);
  return WIN_SUCCESS;
}

int win_obj::fence(){
  return WIN_SUCCESS;
}

int win_obj::lock(int requestRank, int lock_type){
  owner = requestRank;
  return WIN_SUCCESS;
}

int win_obj::unlock(int requestRank){
  if (owner != requestRank){
    CkPrintf("    ERROR: Can't unlock a lock which you don't own.\n");
    return WIN_ERROR;
  }
  owner = -1;

  // dequeue from queue itself
  dequeue();

  return WIN_SUCCESS;
}

void win_obj::dequeue() {
  lockQueueEntry *lq = lockQueue.deq();
  delete lq;
}

void win_obj::enqueue(int requestRank, int lock_type) {
  lockQueueEntry *lq = new lockQueueEntry(requestRank, lock_type);
  lockQueue.enq(lq);
}

bool win_obj::emptyQueue() {
  return (lockQueue.length()==0);
}

void win_obj::lockTopQueue() {
  lockQueueEntry *lq = lockQueue.deq();
  lock(lq->requestRank, lq->lock_type);
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

int ampiParent::addWinStruct(WinStruct* win) {
  winStructList.push_back(win);
  return winStructList.size()-1;
}

WinStruct ampiParent::getWinStruct(MPI_Win win) const {
  return *(winStructList[(int)win]);
}

void ampiParent::removeWinStruct(WinStruct win) {/*winStructList.remove(win);*/}

int ampi::winPut(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
                 MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, WinStruct win){
  CkDDT_DataType *ddt = getDDT()->getType(orgtype);
  int orgtotalsize = ddt->getSize(orgcnt);

  if (ddt->isContig()) {
    thisProxy[rank].winRemotePut(orgtotalsize, (char*)orgaddr, orgcnt, orgtype, targdisp,
                          targcnt, targtype, win.index);
  }
  else {
    vector<char> sorgaddr(orgtotalsize);
    ddt->serialize((char*)orgaddr, &sorgaddr[0], orgcnt, 1);
    thisProxy[rank].winRemotePut(orgtotalsize, &sorgaddr[0], orgcnt, orgtype, targdisp,
                          targcnt, targtype, win.index);
  }

  return MPI_SUCCESS;
}

void ampi::winRemotePut(int orgtotalsize, char* sorgaddr, int orgcnt, MPI_Datatype orgtype,
                        MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, int winIndex) {
  win_obj *winobj = winObjects[winIndex];
  CkDDT_DataType *tddt = getDDT()->getType(targtype);
  int targunit = tddt->getSize();
  int orgunit = getDDT()->getSize(orgtype);

  winobj->put(sorgaddr, orgcnt, orgunit, targdisp, targcnt, targunit);
  char* targaddr = ((char*)(winobj->baseAddr)) + targunit*targdisp;
  tddt->serialize(targaddr, (char*)sorgaddr, targcnt, (-1));
}

int ampi::winGet(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
                 MPI_Aint targdisp, int targcnt, MPI_Datatype targtype,
                 WinStruct win){
  // Send the request for data to remote side
  AMPI_DEBUG("    Rank[%d:%d] invoke Remote get at [%d]\n", thisIndex, myRank, rank);
  CkDDT_DataType *ddt = getDDT()->getType(orgtype);
  int orgtotalsize = ddt->getSize(orgcnt);
  AmpiMsg *msg = new AmpiMsg();

  msg = thisProxy[rank].winRemoteGet(orgcnt, orgtype, targdisp, targcnt, targtype, win.index);

  // Process the reply message by serializing the data into the desired memory position
  ddt->serialize((char*)orgaddr, msg->data, orgcnt, (-1));
  AMPI_DEBUG("    Rank[%d] got win  [%d] \n", thisIndex, *(int*)msg->data);
  AMPI_DEBUG("    Rank[%d] got win  [%d] , size %d\n", thisIndex, *(int*)orgaddr, orgcnt);

  delete msg;
  return MPI_SUCCESS;
}

AmpiMsg* ampi::winRemoteGet(int orgcnt, MPI_Datatype orgtype, MPI_Aint targdisp, int targcnt,
                            MPI_Datatype targtype, int winIndex) {
  AMPI_DEBUG("    RemoteGet invoked at Rank[%d:%d]\n", thisIndex, myRank);

  CkDDT_DataType *tddt = getDDT()->getType(targtype);
  int targunit = tddt->getSize();
  int targtotalsize = targunit*targcnt;
  int orgunit = getDDT()->getSize(orgtype);
  win_obj *winobj = winObjects[winIndex];
  char* targaddr = (char*)(winobj->baseAddr) + targunit*targdisp;

  winobj->get(targaddr, orgcnt, orgunit, targdisp, targcnt, targunit);

  AMPI_DEBUG("    Rank[%d] get win  [%d] \n", thisIndex, *(int*)(targaddr));
  AmpiMsg *msg = new (targtotalsize, 0) AmpiMsg(-1, -1, -1, thisIndex, targtotalsize, myComm.getComm());
  tddt->serialize(targaddr, msg->data, targcnt, 1);
  return msg;
}

int ampi::winIget(MPI_Aint orgdisp, int orgcnt, MPI_Datatype orgtype, int rank,
                  MPI_Aint targdisp, int targcnt, MPI_Datatype targtype,
                  WinStruct win, MPI_Request *req){
  // Send the request to data and handle of Future to remote side
  AMPI_DEBUG("    Rank[%d:%d] request Remote iget at [%d]\n", thisIndex, myRank, rank);
  *req = thisProxy[rank].winRemoteIget(orgdisp, orgcnt, orgtype, targdisp, targcnt, targtype, win.index);
  return MPI_SUCCESS;
}

AmpiMsg* ampi::winRemoteIget(MPI_Aint orgdisp, int orgcnt, MPI_Datatype orgtype,
                             MPI_Aint targdisp, int targcnt,
                             MPI_Datatype targtype, int winIndex) {
  AMPI_DEBUG("    RemoteIget invoked at Rank[%d:%d]\n", thisIndex, myRank);
  CkDDT_DataType *tddt = getDDT()->getType(targtype);
  int targunit = tddt->getSize();
  int targtotalsize = targunit*targcnt;
  int orgunit = getDDT()->getSize(orgtype);

  win_obj *winobj = winObjects[winIndex];
  winobj->iget(orgcnt, orgunit, targdisp, targcnt, targunit);

  AMPI_DEBUG("    Rank[%d] iget win  [%d] \n", thisIndex, *(int*)(targaddr));

  AmpiMsg *msg = new (targtotalsize, 0) AmpiMsg(-1, -1, -1, thisIndex, targtotalsize,
                                                 myComm.getComm());

  char* targaddr = (char*)(winobj->baseAddr) + targdisp*targunit;
  tddt->serialize(targaddr, msg->data, targcnt, 1);
  AMPI_DEBUG("    Rank[%d] copy win  [%d] \n", thisIndex, *(int*)msg->data);
  return msg;
}

int ampi::winIgetWait(MPI_Request *request, MPI_Status *status) {
  // Wait on the Future object
  AMPI_DEBUG("    [%d] Iget Waiting\n", thisIndex, *request);
  status->msg = (AmpiMsg*)CkWaitReleaseFuture(*request);
  AMPI_DEBUG("    [%d] Iget Waiting [%d] awaken\n", thisIndex, *request);
  return MPI_SUCCESS;
}

int ampi::winIgetFree(MPI_Request *request, MPI_Status *status) {
  AMPI_DEBUG("    [%d] : Iget [%d] frees buffer\n", thisIndex, *request);

  void *data = NULL;
  AMPI_Iget_data(data, *status);
  if(!data) {
    AMPI_DEBUG("    [%d] Iget [%d] attempt to free NULL buffer \n", thisIndex, *request);
    return ampiErrhandler("AMPI_Iget_free", MPI_ERR_BUFFER);
  }
  else {
    delete (status->msg);
    return MPI_SUCCESS;
  }
}

int ampi::winAccumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
                        MPI_Aint targdisp, int targcnt, MPI_Datatype targtype,
                        MPI_Op op, WinStruct win) {
  CkDDT_DataType *ddt = getDDT()->getType(orgtype);
  int orgtotalsize = ddt->getSize(orgcnt);
  AMPI_DEBUG("    Rank[%d:%d] invoke Remote accumulate at [%d]\n", thisIndex, myRank, rank);

  if (ddt->isContig()) {
    thisProxy[rank].winRemoteAccumulate(orgtotalsize, (char*)orgaddr, orgcnt, orgtype,
                                 targdisp, targcnt, targtype,  op, win.index);
  }
  else {
    vector<char> sorgaddr(orgtotalsize);
    ddt->serialize((char*)orgaddr, &sorgaddr[0], orgcnt, 1);
    thisProxy[rank].winRemoteAccumulate(orgtotalsize, &sorgaddr[0], orgcnt, orgtype,
                                 targdisp, targcnt, targtype,  op, win.index);
  }

  return MPI_SUCCESS;
}

void ampi::winRemoteAccumulate(int orgtotalsize, char* sorgaddr, int orgcnt,
                               MPI_Datatype orgtype, MPI_Aint targdisp,
                               int targcnt, MPI_Datatype targtype, MPI_Op op,
                               int winIndex) {
  win_obj *winobj = winObjects[winIndex];
  CkDDT_DataType *ddt = getDDT()->getType(targtype);

  if (ddt->isContig()) {
    winobj->accumulate(sorgaddr, targcnt, targtype, targdisp, targcnt, targtype, op);
  }
  else {
    vector<char> getdata(orgtotalsize);
    ddt->serialize(&getdata[0], sorgaddr, targcnt, (-1));
    winobj->accumulate(&getdata[0], targcnt, targtype, targdisp, targcnt, targtype, op);
  }
}

int ampi::winGetAccumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype,
                           void *resaddr, int rescnt, MPI_Datatype restype, int rank,
                           MPI_Aint targdisp, int targcnt, MPI_Datatype targtype,
                           MPI_Op op, WinStruct win) {
  CkDDT_DataType *ddt = getDDT()->getType(orgtype);
  int orgtotalsize = ddt->getSize(orgcnt);
  AmpiMsg *msg = new AmpiMsg();
  AMPI_DEBUG("    Rank[%d:%d] invoke Remote get at [%d]\n", thisIndex, myRank, rank);

  msg = thisProxy[rank].winRemoteGet(orgcnt, orgtype, targdisp, targcnt, targtype, win.index);

  ddt->serialize((char*)resaddr, msg->data, orgcnt, (-1));
  if (ddt->isContig()) {
    applyOp(orgtype, op, orgcnt, resaddr, orgaddr);
  } else {
    vector<char> sorgaddr(orgtotalsize);
    ddt->serialize((char*)orgaddr, &sorgaddr[0], orgcnt, 1);
    applyOp(orgtype, op, orgcnt, resaddr, &sorgaddr[0]);
    ddt->serialize((char*)orgaddr, &sorgaddr[0], orgcnt, -1);
  }

  delete msg;
  return MPI_SUCCESS;
}

int ampi::winCompareAndSwap(void *orgaddr, void *compaddr, void *resaddr, MPI_Datatype type,
                            int rank, MPI_Aint targdisp, WinStruct win){
  CkDDT_DataType *ddt = getDDT()->getType(type);
  AmpiMsg* msg = new AmpiMsg();

  msg = thisProxy[rank].winRemoteCompareAndSwap(getDDT()->getType(type)->getSize(1), (char*)orgaddr,
                                                (char*)compaddr, type, targdisp, win.index);
  ddt->serialize((char*)resaddr, msg->data, 1, 1);

  delete msg;
  return MPI_SUCCESS;
}

AmpiMsg* ampi::winRemoteCompareAndSwap(int size, char* sorgaddr, char* compaddr, MPI_Datatype type,
                                       MPI_Aint targdisp, int winIndex) {
  win_obj *winobj = winObjects[winIndex];
  winobj->put(sorgaddr, 1, size, targdisp, 1, size);

  CkDDT_DataType *ddt = getDDT()->getType(type);
  char* targaddr = ((char*)(winobj->baseAddr)) + ddt->getSize(targdisp);

  AmpiMsg *msg = new (size, 0) AmpiMsg(-1, -1, -1, thisIndex, size, myComm.getComm());
  ddt->serialize(targaddr, msg->data, 1, 1);

  if (*targaddr == *compaddr) {
    ddt->serialize(targaddr, (char*)sorgaddr, 1, (-1));
  }

  return msg;
}

int ampi::winLock(int lock_type, int rank, WinStruct win) {
  AMPI_DEBUG("    [%d] Lock: invoke Remote lock at [%d]\n", thisIndex, rank);
  thisProxy[rank].winRemoteLock(lock_type, win.index, thisIndex);
  return MPI_SUCCESS;
}

void ampi::winRemoteLock(int lock_type, int winIndex, int requestRank) {
  AMPI_DEBUG("    [%d] RemoteLock: invoked \n", thisIndex);
  win_obj *winobj = winObjects[winIndex];

  // check if any one else waiting in the queue
  if(winobj->owner > -1 && !(winobj->emptyQueue()))  {
  // queue it if queue non-empty
    winobj->enqueue(requestRank, lock_type);
    AMPI_DEBUG("    [%d] RemoteLock: queue lock from [%d] \n", thisIndex, requestRank);
  }
  // if queue empty, get semaphore and queue it
  else {
    winobj->lock(requestRank, lock_type);
    winobj->enqueue(requestRank, lock_type);
    AMPI_DEBUG("    [%d] RemoteLock: give lock to [%d] \n", thisIndex, requestRank);
  }
}

int ampi::winUnlock(int rank, WinStruct win) {
  AMPI_DEBUG("    [%d] Unlock: invoke Remote lock at [%d]\n", thisIndex, rank);
  thisProxy[rank].winRemoteUnlock(win.index, thisIndex);
  return MPI_SUCCESS;
}

void ampi::winRemoteUnlock(int winIndex, int requestRank) {
  AMPI_DEBUG("    [%d] RemoteUnlock: invoked \n", thisIndex);
  win_obj *winobj = winObjects[winIndex];
  winobj->unlock(requestRank);
  AMPI_DEBUG("    [%d] RemoteUnlock: [%d] release lock\n", thisIndex, requestRank);

  // if queue non-empty, get lock for the first waiting one and reply
  if(!(winobj->emptyQueue())) {
    AMPI_DEBUG("    [%d] RemoteUnlock: queue non-empty, give lock to \n", thisIndex);
    winobj->lockTopQueue();
  }
}

MPI_Win ampi::createWinInstance(void *base, MPI_Aint size, int disp_unit, MPI_Info info) {
  AMPI_DEBUG("     Creating win obj {%d, %p}\n ", myComm.getComm(), base);
  win_obj *newobj = new win_obj((char*)(NULL), base, size, disp_unit, myComm.getComm());
  winObjects.push_back(newobj);
  WinStruct *newwin = new WinStruct(myComm.getComm(),winObjects.size()-1);
  AMPI_DEBUG("     Creating MPI_WIN at (%p) with {%d, %d}\n", &newwin, myComm.getComm(), winObjects.size()-1);
  return (parent->addWinStruct(newwin));
}

int ampi::deleteWinInstance(MPI_Win win) {
  WinStruct winStruct = parent->getWinStruct(win);
  win_obj *winobj = winObjects[winStruct.index];
  parent->removeWinStruct(winStruct); // really it does nothing at all
  winobj->free();
  return MPI_SUCCESS;
}

int ampi::winGetGroup(WinStruct win, MPI_Group *group) const {
   *group = parent->comm2group(win.comm);
   return MPI_SUCCESS;
}

void ampi::winSetName(WinStruct win, const char *name) {
  win_obj *winobj = winObjects[win.index];
  winobj->setName(name);
}

void ampi::winGetName(WinStruct win, char *name, int *length) const {
  win_obj *winobj = winObjects[win.index];
  winobj->getName(name, length);
}

win_obj* ampi::getWinObjInstance(WinStruct win) const {
  return winObjects[win.index];
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
  /* set the builtin attributes on the window */
  AMPI_Win_set_attr(*newwin, MPI_WIN_BASE, &base);
  AMPI_Win_set_attr(*newwin, MPI_WIN_SIZE, &size);
  AMPI_Win_set_attr(*newwin, MPI_WIN_DISP_UNIT, &disp_unit);
  /* need to reduction here: to make sure every processor participates */
  ptr->barrier();
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
  if(win==NULL) { return ampiErrhandler("AMPI_Win_free", MPI_ERR_WIN); }

  WinStruct winStruct = getAmpiParent()->getWinStruct(*win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  ptr->deleteWinInstance(*win);
  /* Need a barrier here: to ensure that every process participates */
  ptr->barrier();
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
  handle_MPI_BOTTOM(orgaddr, orgtype);
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
  handle_MPI_BOTTOM(orgaddr, orgtype);
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
  handle_MPI_BOTTOM(orgaddr, orgtype);
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  return ptr->winAccumulate(orgaddr, orgcnt, orgtype, rank,
                            targdisp, targcnt, targtype, op, winStruct);
}

/*
 * int AMPI_Get_accumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype,
 *         void *resaddr, int rescnt, MPI_Datatype restype,
 *         int rank, MPI_Aint targdisp, int targcnt,
 *         MPI_Datatype targtype, MPI_Op op, MPI_Win win)
 *   Perform an atomic, one-sided read-and-accumulate operation.
 */
CDECL
int AMPI_Get_accumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype,
                        void *resaddr, int rescnt, MPI_Datatype restype,
                        int rank, MPI_Aint targdisp, int targcnt,
                        MPI_Datatype targtype, MPI_Op op, MPI_Win win) {
  AMPIAPI("AMPI_Get_accumulate");
  handle_MPI_BOTTOM(orgaddr, orgtype);
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  return ptr->winGetAccumulate(orgaddr, orgcnt, orgtype, resaddr, rescnt, restype,
                               rank, targdisp, targcnt, targtype, op, winStruct);
}

/*
 * int AMPI_Fetch_and_op(void *orgaddr, void *resaddr, MPI_Datatype type,
 *         int rank, MPI_Aint targdisp, MPI_Op op, MPI_Win win)
 *   Perform one-sided read-modify-write.
 */
CDECL
int AMPI_Fetch_and_op(void *orgaddr, void *resaddr, MPI_Datatype type,
                      int rank, MPI_Aint targdisp, MPI_Op op, MPI_Win win) {
  AMPIAPI("AMPI_Fetch_and_op");
  handle_MPI_BOTTOM(orgaddr, type);
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  // HACK: use GetAccumulate for FetchAndOp
  return ptr->winGetAccumulate(orgaddr, 1, type, resaddr, 1, type,
                               rank, targdisp, 1, type, op, winStruct);
}

/*
 * int AMPI_Compare_and_swap(void *orgaddr, void *compaddr, void *resaddr,
 *         MPI_Datatype type, int rank, MPI_Aint targdisp, MPI_Win win)
 *   Perform one-sided atomic compare-and-swap.
 */
CDECL
int AMPI_Compare_and_swap(void *orgaddr, void *compaddr, void *resaddr, MPI_Datatype type,
                          int rank, MPI_Aint targdisp, MPI_Win win) {
  AMPIAPI("AMPI_Compare_and_swap");
  handle_MPI_BOTTOM(orgaddr, type);
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  return ptr->winCompareAndSwap(orgaddr, compaddr, resaddr, type, rank, targdisp, winStruct);
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
  ptr->barrier();

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
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_complete(MPI_Win win){
  AMPIAPI("AMPI_Win_complete");
  return MPI_SUCCESS;
}

// FIX PLACE II
CDECL
int AMPI_Iget(MPI_Aint orgdisp, int orgcnt, MPI_Datatype orgtype, int rank,
              MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, MPI_Win win,
              MPI_Request *request) {
  AMPIAPI("AMPI_Iget");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  // winGet is a local function which will call the remote method on #rank processor
  return  ptr->winIget(orgdisp, orgcnt, orgtype, rank, targdisp, targcnt, targtype, winStruct,
		       request);
}

CDECL
int AMPI_Iget_wait(MPI_Request *request, MPI_Status *status, MPI_Win win) {
  AMPIAPI("AMPI_Iget_wait");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  // winGet is a local function which will call the remote method on #rank processor
  return  ptr->winIgetWait(request,status);
}

CDECL
int AMPI_Iget_free(MPI_Request *request, MPI_Status *status, MPI_Win win) {
  AMPIAPI("AMPI_Iget_free");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  // winGet is a local function which will call the remote method on #rank processor
  return  ptr->winIgetFree(request, status);
}

CDECL
int AMPI_Iget_data(void *data, MPI_Status status) {
  data = (void*)((AmpiMsg*)status.msg)->data;
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
  TCharmAPIRoutine apiRoutineSentry("AMPI_Alloc_mem", "ampi", false);
  *(void **)baseptr = malloc(size);
  return MPI_SUCCESS;
}

/*
 * int AMPI_Free_mem(void *base)
 *   Frees memory that was previous allocated by MPI_Alloc_mem call
 */
CDECL
int AMPI_Free_mem(void *baseptr){
  TCharmAPIRoutine apiRoutineSentry("AMPI_Free_mem", "ampi", false);
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
int AMPI_Win_delete_attr(MPI_Win win, int key){
  AMPIAPI("AMPI_Win_delete_attr");
  return getAmpiParent()->deleteWinAttr(win,key);
}

CDECL
int AMPI_Win_get_attr(MPI_Win win, int key, void* value, int* flag) {
  AMPIAPI("AMPI_Win_get_attr");
  return getAmpiParent()->getWinAttr(win,key,value,flag);
}

CDECL
int AMPI_Win_set_attr(MPI_Win win, int key, void* value) {
  AMPIAPI("AMPI_Win_set_attr");
  return getAmpiParent()->setWinAttr(win,key,value);
}

CDECL
int AMPI_Win_set_name(MPI_Win win, const char *name) {
  AMPIAPI("AMPI_Win_set_name");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  ptr->winSetName(winStruct, name);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_set_info(MPI_Win win, MPI_Info info){
  AMPIAPI("AMPI_Comm_set_info");
  /* FIXME: no-op implementation */
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_get_info(MPI_Win win, MPI_Info *info){
  AMPIAPI("AMPI_Comm_get_info");
  /* FIXME: no-op implementation */
  *info = MPI_INFO_NULL;
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_create_errhandler(MPI_Win_errhandler_function *win_errhandler_fn,
                               MPI_Errhandler *errhandler) {
  AMPIAPI("AMPI_Win_create_errhandler");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_get_errhandler(MPI_Win win, MPI_Errhandler *errhandler) {
  AMPIAPI("AMPI_Win_get_errhandler");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler) {
  AMPIAPI("AMPI_Win_set_errhandler");
  return MPI_SUCCESS;
}

int MPI_win_null_copy_fn(MPI_Win win, int keyval, void *extra_state,
                         void *attr_in, void *attr_out, int *flag){
  (*flag) = 0;
  return MPI_SUCCESS;
}

int MPI_win_dup_fn(MPI_Win win, int keyval, void *extra_state,
                   void *attr_in, void *attr_out, int *flag){
  (*(void **)attr_out) = attr_in;
  (*flag) = 1;
  return MPI_SUCCESS;
}

int MPI_win_null_delete_fn(MPI_Win win, int keyval, void *attr, void *extra_state){
  return MPI_SUCCESS;
}

CDECL
int AMPI_Win_create_keyval(MPI_Win_copy_attr_function *copy_fn,
                           MPI_Win_delete_attr_function *delete_fn,
                           int *keyval, void *extra_state) {
  AMPIAPI("AMPI_Win_create_keyval");
  return getAmpiParent()->createKeyval(copy_fn,delete_fn,keyval,extra_state);
}

CDECL
int AMPI_Win_free_keyval(int *keyval) {
  AMPIAPI("AMPI_Win_free_keyval");
  return getAmpiParent()->freeKeyval(keyval);
}

CDECL
int AMPI_Win_get_name(MPI_Win win, char *name, int *length) {
  AMPIAPI("AMPI_Win_get_name");
  WinStruct winStruct = getAmpiParent()->getWinStruct(win);
  ampi *ptr = getAmpiInstance(winStruct.comm);
  ptr->winGetName(winStruct, name, length);
  return MPI_SUCCESS;
}
