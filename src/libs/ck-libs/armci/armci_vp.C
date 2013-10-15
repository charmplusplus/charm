#include <vector>
#include "armci_impl.h"

using namespace std;

int **_armciRednLookupTable;

// FIXME: might be memory leakage in put
// This is the way to adapt a library's preferred start interface with the
// one provided by TCharm (eg. argc,argv vs void).
extern "C" void armciLibStart(void) {
  int argc=CkGetArgc();
  char **argv=CkGetArgv();
  ARMCI_Main_cpp(argc, argv);
}

_ARMCI_GENERATE_POLYMORPHIC_REDUCTION(sum,ret[i]+=value[i];)
_ARMCI_GENERATE_POLYMORPHIC_REDUCTION(product,ret[i]*=value[i];)
_ARMCI_GENERATE_POLYMORPHIC_REDUCTION(max,if (ret[i]<value[i]) ret[i]=value[i];)
_ARMCI_GENERATE_POLYMORPHIC_REDUCTION(min,if (ret[i]>value[i]) ret[i]=value[i];)
_ARMCI_GENERATE_ABS_REDUCTION()

static int armciLibStart_idx = -1;

void armciNodeInit(void) {
  CmiAssert(armciLibStart_idx == -1);
  armciLibStart_idx = TCHARM_Register_thread_function((TCHARM_Thread_data_start_fn)armciLibStart);

  // initialize the reduction table
  _armciRednLookupTable = new int*[_ARMCI_NUM_REDN_OPS];
  for (int ops=0; ops<_ARMCI_NUM_REDN_OPS; ops++) {
    _armciRednLookupTable[ops] = new int[ARMCI_NUM_DATATYPES];
  }

  // Add the new reducers for ARMCI.
  _ARMCI_REGISTER_POLYMORPHIC_REDUCTION(sum,_ARMCI_REDN_OP_SUM);
  _ARMCI_REGISTER_POLYMORPHIC_REDUCTION(product,_ARMCI_REDN_OP_SUM);
  _ARMCI_REGISTER_POLYMORPHIC_REDUCTION(max,_ARMCI_REDN_OP_MAX);
  _ARMCI_REGISTER_POLYMORPHIC_REDUCTION(min,_ARMCI_REDN_OP_MIN);
  _ARMCI_REGISTER_POLYMORPHIC_REDUCTION(absmax,_ARMCI_REDN_OP_ABSMAX);
  _ARMCI_REGISTER_POLYMORPHIC_REDUCTION(absmin,_ARMCI_REDN_OP_ABSMIN);
}

// Default startup routine (can be overridden by user's own)
// This will be registered with TCharm's startup routine
// in the Node initialization function.
static void ArmciDefaultSetup(void) {
  // Create the base threads on TCharm using user-defined start routine.
  TCHARM_Create(TCHARM_Get_num_chunks(), armciLibStart_idx);
}

CtvDeclare(ArmciVirtualProcessor *, _armci_ptr);

// Node initialization (made by initproc of the module armci)
void armciProcInit(void) {
  CtvInitialize(ArmciVirtualProcessor, _armci_ptr);
  CtvAccess(_armci_ptr) = NULL;

  // Register the library's default startup routine to TCharm
  TCHARM_Set_fallback_setup(ArmciDefaultSetup);
}

ArmciVirtualProcessor::ArmciVirtualProcessor(const CProxy_TCharm &_thr_proxy)
  : TCharmClient1D(_thr_proxy) {
  thisProxy = this;
  tcharmClientInit();
  thread->semaPut(ARMCI_TCHARM_SEMAID,this);
  memBlock = CmiIsomallocBlockListNew(thread->getThread());
  thisProxy = CProxy_ArmciVirtualProcessor(thisArrayID);
  addressReply = NULL;
  // Save ourselves for the waiting ARMCI_Init
}

ArmciVirtualProcessor::ArmciVirtualProcessor(CkMigrateMessage *m) 
  : TCharmClient1D(m) 
{
//  memBlock = NULL; //Paranoia-- makes sure we initialize this in pup
  thread = NULL;
  addressReply = NULL;
}

ArmciVirtualProcessor::~ArmciVirtualProcessor()
{
#if !CMK_USE_MEMPOOL_ISOMALLOC
  CmiIsomallocBlockListDelete(memBlock);
#endif
  if (addressReply) {delete addressReply;}
}

void ArmciVirtualProcessor::setupThreadPrivate(CthThread forThread) {
  CtvAccessOther(forThread, _armci_ptr) = this;
  armci_nproc = thread->getNumElements();
}

void ArmciVirtualProcessor::getAddresses(AddressMsg *msg) {
  addressReply = msg;
  thread->resume();
}

// implemented as a blocking put for now
void ArmciVirtualProcessor::put(pointer src, pointer dst,
			       int nbytes, int dst_proc) {
/*  if(dst_proc == thisIndex){
    memcpy(dst,src,nbytes);
    return;
  }*/
  int hdl = hdlList.size();
  Armci_Hdl* entry = new Armci_Hdl(ARMCI_BPUT, dst_proc, nbytes, src, dst);
  hdlList.push_back(entry);

  ArmciMsg *msg = new (nbytes, 0) ArmciMsg(dst,nbytes,thisIndex,hdl);
  memcpy(msg->data, src, nbytes);
  thisProxy[dst_proc].putData(msg);
//  thisProxy[dst_proc].putData(dst,nbytes,msg->data,thisIndex,hdl);
}

int ArmciVirtualProcessor::nbput(pointer src, pointer dst,
			       int nbytes, int dst_proc) {
/*  if(dst_proc == thisIndex){
    memcpy(dst,src,nbytes);
    return -1;
  }*/
  int hdl = hdlList.size();
  Armci_Hdl* entry = new Armci_Hdl(ARMCI_PUT, dst_proc, nbytes, src, dst);
  hdlList.push_back(entry);

  ArmciMsg *msg = new (nbytes, 0) ArmciMsg(dst,nbytes,thisIndex,hdl);
  memcpy(msg->data, src, nbytes);
  thisProxy[dst_proc].putData(msg);
  
  return hdl;
}

void ArmciVirtualProcessor::nbput_implicit(pointer src, pointer dst,
					  int nbytes, int dst_proc) {
  int hdl = hdlList.size();
  Armci_Hdl* entry = new Armci_Hdl(ARMCI_IPUT, dst_proc, nbytes, src, dst);
  hdlList.push_back(entry);

  ArmciMsg *msg = new (nbytes, 0) ArmciMsg(dst,nbytes,thisIndex,hdl);
  memcpy(msg->data, src, nbytes);
  thisProxy[dst_proc].putData(msg);
}

void ArmciVirtualProcessor::putData(pointer dst, int nbytes, char *data,
				    int src_proc, int hdl) {
  memcpy(dst, data, nbytes);
  thisProxy[src_proc].putAck(hdl);
}

void ArmciVirtualProcessor::putData(ArmciMsg *m) {
  memcpy(m->dst, m->data, m->nbytes);
  thisProxy[m->src_proc].putAck(m->hdl);
  delete m;
}

void ArmciVirtualProcessor::putAck(int hdl) {
  if(hdl != -1) { // non-blocking 
    hdlList[hdl]->acked = 1;  
    if (hdlList[hdl]->wait == 1) {
      hdlList[hdl]->wait = 0;
      thread->resume();
    }
  }
  thread->resume();
}

void ArmciVirtualProcessor::get(pointer src, pointer dst,
			       int nbytes, int src_proc) {
/*  if(src_proc == thisIndex){
    memcpy(dst,src,nbytes);
    return;
  }*/
  thisProxy[src_proc].requestFromGet(src, dst, nbytes, thisIndex, -1);
  // wait for reply
  thread->suspend();
}

int ArmciVirtualProcessor::nbget(pointer src, pointer dst,
			       int nbytes, int src_proc) {
/*  if(src_proc == thisIndex){
    memcpy(dst,src,nbytes);
    return -1;
  }*/

  int hdl = hdlList.size();
  Armci_Hdl* entry = new Armci_Hdl(ARMCI_GET, src_proc, nbytes, src, dst);
  hdlList.push_back(entry);
  
  thisProxy[src_proc].requestFromGet(src, dst, nbytes, thisIndex, hdl);

  return hdl;
}

void ArmciVirtualProcessor::nbget_implicit(pointer src, pointer dst,
					   int nbytes, int src_proc) {
  int hdl = hdlList.size();
  Armci_Hdl* entry = new Armci_Hdl(ARMCI_IGET, src_proc, nbytes, src, dst);
  hdlList.push_back(entry);
  
  thisProxy[src_proc].requestFromGet(src, dst, nbytes, thisIndex, hdl);
}

void ArmciVirtualProcessor::wait(int hdl){
  if(hdl == -1) return;
  while (1) {
    if(hdlList[hdl]->acked != 0)
      break;
    else
      thread->suspend();
  }
}

// CWL NOTE: This works only because in wait(), the while (1) loop
//   insists on matching the first unackowledged non-blocking call 
//   waitmulti is
//   waiting on. Out-of-order acknowledgements will wake the thread
//   but cause it to suspend itself again until the call is acknowledged.
//   Subsequent calls to wait from waitmulti will then succeed because
//   out-of-order acks would have set the acked flag.
void ArmciVirtualProcessor::waitmulti(vector<int> procs){
  for(int i=0;i<procs.size();i++){
    wait(procs[i]);
  }
}

void ArmciVirtualProcessor::waitproc(int proc){
  vector<int> procs;
  for(int i=0;i<hdlList.size();i++){
    if((hdlList[i]->acked == 0) && 
       (hdlList[i]->proc == proc) && 
       ((hdlList[i]->op & IMPLICIT_MASK) != 0)) {
      hdlList[i]->wait = 1;
      procs.push_back(i);
    }
  }
  waitmulti(procs);
}

void ArmciVirtualProcessor::waitall(){
  vector<int> procs;
  for(int i=0;i<hdlList.size();i++){
    if((hdlList[i]->acked == 0) && 
       ((hdlList[i]->op & IMPLICIT_MASK) != 0)) {
      hdlList[i]->wait = 1;
      procs.push_back(i);
    }
  }
  waitmulti(procs);
}

void ArmciVirtualProcessor::fence(int proc){
  vector<int> procs;
  for(int i=0;i<hdlList.size();i++){
    if((hdlList[i]->acked == 0) && 
       ((hdlList[i]->op & BLOCKING_MASK) != 0) && 
       (hdlList[i]->proc == proc))
      procs.push_back(i);
  }
  waitmulti(procs);
}
void ArmciVirtualProcessor::allfence(){
  vector<int> procs;
  for(int i=0;i<hdlList.size();i++){
    if((hdlList[i]->acked == 0) && 
       ((hdlList[i]->op & BLOCKING_MASK) != 0))
      procs.push_back(i);
  }
  waitmulti(procs);
}
void ArmciVirtualProcessor::barrier(){
  allfence();
  CkCallback cb(CkIndex_ArmciVirtualProcessor::resumeThread(),thisProxy);
  contribute(0,NULL,CkReduction::sum_int,cb);
  thread->suspend();
}

void ArmciVirtualProcessor::resumeThread(void){
  thread->resume();
}

int ArmciVirtualProcessor::test(int hdl){
  if(hdl == -1) return 1;
  return hdlList[hdl]->acked;
}

void ArmciVirtualProcessor::requestFromGet(pointer src, pointer dst, int nbytes,
				       int dst_proc, int hdl) {
  ArmciMsg *msg = new (nbytes, 0) ArmciMsg(dst,nbytes,-1,hdl);
  memcpy(msg->data, src, nbytes);
  thisProxy[dst_proc].putDataFromGet(msg);
}

// this is essentially the same as putData except that no acknowledgement
// is required and the thread suspended while waiting for the data is
// awoken.
void ArmciVirtualProcessor::putDataFromGet(pointer dst, int nbytes, char *data, int hdl) {
  memcpy(dst, data, nbytes);
  if(hdl != -1) { // non-blocking 
    hdlList[hdl]->acked = 1;  
    if (hdlList[hdl]->wait == 1) {
      hdlList[hdl]->wait = 0;
      thread->resume();
    }
  }
  thread->resume();
}

void ArmciVirtualProcessor::putDataFromGet(ArmciMsg *m) {
  memcpy(m->dst, m->data, m->nbytes);
  if(m->hdl != -1) { // non-blocking 
    hdlList[m->hdl]->acked = 1;  
    if (hdlList[m->hdl]->wait == 1) {
      hdlList[m->hdl]->wait = 0;
      thread->resume();
    }
  }
  delete m;
  thread->resume();
}

void ArmciVirtualProcessor::puts(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[],
	   int count[], int stride_levels, int dst_proc){
  int nbytes = 1;
  for(int i=0;i<stride_levels+1;i++) 
    nbytes *= count[i];
  
/*  if(dst_proc == thisIndex){
    buffer = new char[nbytes];
    stridedCopy(src_ptr, buffer, src_stride_ar, count, stride_levels, 1);
    stridedCopy(dst_ptr, buffer, dst_stride_ar, count, stride_levels, 0);
    return;
  }
*/
  int hdl = hdlList.size();
  Armci_Hdl* entry = new Armci_Hdl(ARMCI_BPUT, dst_proc, nbytes, src_ptr, dst_ptr);
  hdlList.push_back(entry);
  
  ArmciStridedMsg *m = new (stride_levels,stride_levels+1,nbytes, 0) ArmciStridedMsg(dst_ptr,stride_levels,nbytes,thisIndex,hdl);

  memcpy(m->dst_stride_ar,dst_stride_ar,sizeof(int)*stride_levels);
  memcpy(m->count,count,sizeof(int)*(stride_levels+1));
  stridedCopy(src_ptr, m->data, src_stride_ar, count, stride_levels, 1);
  thisProxy[dst_proc].putsData(m);
}

int ArmciVirtualProcessor::nbputs(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[],
	   int count[], int stride_levels, int dst_proc){
  int nbytes = 1;
  for(int i=0;i<stride_levels+1;i++) 
    nbytes *= count[i];
  
/*  if(dst_proc == thisIndex){
    buffer = new char[nbytes];
    stridedCopy(src_ptr, buffer, src_stride_ar, count, stride_levels, 1);
    stridedCopy(dst_ptr, buffer, dst_stride_ar, count, stride_levels, 0);
    return -1;
  }
  */
  int hdl = hdlList.size();
  Armci_Hdl* entry = new Armci_Hdl(ARMCI_PUT, dst_proc, nbytes, src_ptr, dst_ptr);
  hdlList.push_back(entry);
 
  ArmciStridedMsg *m = new (stride_levels,stride_levels+1,nbytes, 0) ArmciStridedMsg(dst_ptr,stride_levels,nbytes,thisIndex,hdl);

  memcpy(m->dst_stride_ar,dst_stride_ar,sizeof(int)*stride_levels);
  memcpy(m->count,count,sizeof(int)*(stride_levels+1));
  stridedCopy(src_ptr, m->data, src_stride_ar, count, stride_levels, 1);
  thisProxy[dst_proc].putsData(m);
  return hdl;
}

void ArmciVirtualProcessor::nbputs_implicit(pointer src_ptr, 
					    int src_stride_ar[], 
					    pointer dst_ptr, 
					    int dst_stride_ar[],
					    int count[], int stride_levels, 
					    int dst_proc){
  int nbytes = 1;
  for(int i=0;i<stride_levels+1;i++) 
    nbytes *= count[i];
  int hdl = hdlList.size();
  Armci_Hdl* entry = new Armci_Hdl(ARMCI_IPUT, dst_proc, nbytes, 
				   src_ptr, dst_ptr);
  hdlList.push_back(entry);
 
  ArmciStridedMsg *m = new (stride_levels,stride_levels+1,nbytes, 0) ArmciStridedMsg(dst_ptr,stride_levels,nbytes,thisIndex,hdl);

  memcpy(m->dst_stride_ar,dst_stride_ar,sizeof(int)*stride_levels);
  memcpy(m->count,count,sizeof(int)*(stride_levels+1));
  stridedCopy(src_ptr, m->data, src_stride_ar, count, stride_levels, 1);
  thisProxy[dst_proc].putsData(m);
}

void ArmciVirtualProcessor::putsData(pointer dst_ptr, int dst_stride_ar[], 
  		int count[], int stride_levels,
		int nbytes, char *data, int src_proc, int hdl){
  stridedCopy(dst_ptr, data, dst_stride_ar, count, stride_levels, 0);
  thisProxy[src_proc].putAck(hdl);
}

void ArmciVirtualProcessor::putsData(ArmciStridedMsg *m){
  stridedCopy(m->dst, m->data, m->dst_stride_ar, m->count, m->stride_levels, 0);
  thisProxy[m->src_proc].putAck(m->hdl);
  delete m;
}

void ArmciVirtualProcessor::gets(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[],
	   int count[], int stride_levels, int src_proc){
/*  if(src_proc == thisIndex){
    char *buffer;
    int nbytes = 1;
    for(int i=0;i<stride_levels+1;i++) 
      nbytes *= count[i];
    buffer = new char[nbytes];
    stridedCopy(src_ptr, buffer, src_stride_ar, count, stride_levels, 1);
    stridedCopy(dst_ptr, buffer, dst_stride_ar, count, stride_levels, 0);
    delete buffer;
    return;
  }*/
  thisProxy[src_proc].requestFromGets(src_ptr, src_stride_ar, dst_ptr, dst_stride_ar, 
  					count, stride_levels, thisIndex, -1);
  // wait for reply
  thread->suspend();
}

int ArmciVirtualProcessor::nbgets(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[],
	   int count[], int stride_levels, int src_proc){
  int hdl = hdlList.size();
  int nbytes = 1;
  for(int i=0;i<stride_levels+1;i++) 
    nbytes *= count[i];
/*  if(src_proc == thisIndex){
    char *buffer;
    buffer = new char[nbytes];
    stridedCopy(src_ptr, buffer, src_stride_ar, count, stride_levels, 1);
    stridedCopy(dst_ptr, buffer, dst_stride_ar, count, stride_levels, 0);
    delete buffer;
    return -1;
  }*/
  
  Armci_Hdl* entry = new Armci_Hdl(ARMCI_GET, src_proc, nbytes, src_ptr, dst_ptr);
  hdlList.push_back(entry);

  thisProxy[src_proc].requestFromGets(src_ptr, src_stride_ar, dst_ptr, dst_stride_ar, 
  					count, stride_levels, thisIndex, hdl);

  return hdl;
}

void ArmciVirtualProcessor::nbgets_implicit(pointer src_ptr, 
					    int src_stride_ar[], 
					    pointer dst_ptr, 
					    int dst_stride_ar[],
					    int count[], int stride_levels, 
					    int src_proc) {
  int hdl = hdlList.size();
  int nbytes = 1;
  for(int i=0;i<stride_levels+1;i++) 
    nbytes *= count[i];

  Armci_Hdl* entry = new Armci_Hdl(ARMCI_IGET, src_proc, nbytes, src_ptr, dst_ptr);
  hdlList.push_back(entry);

  thisProxy[src_proc].requestFromGets(src_ptr, src_stride_ar, dst_ptr, dst_stride_ar, 
  					count, stride_levels, thisIndex, hdl);
}

void ArmciVirtualProcessor::requestFromGets(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[], int count[], int stride_levels, int dst_proc, int hdl){
  int nbytes = 1;
  for(int i=0;i<stride_levels+1;i++) 
    nbytes *= count[i];
  
  ArmciStridedMsg *m = new (stride_levels,stride_levels+1,nbytes, 0) ArmciStridedMsg(dst_ptr,stride_levels,nbytes,thisIndex,hdl);

  memcpy(m->dst_stride_ar,dst_stride_ar,sizeof(int)*stride_levels);
  memcpy(m->count,count,sizeof(int)*(stride_levels+1));
  stridedCopy(src_ptr, m->data, src_stride_ar, count, stride_levels, 1);
  thisProxy[dst_proc].putDataFromGets(m);
}
void ArmciVirtualProcessor::putDataFromGets(pointer dst_ptr, int dst_stride_ar[], 
		int count[], int stride_levels, int nbytes, char *data, int hdl){
  stridedCopy(dst_ptr, data, dst_stride_ar, count, stride_levels, 0);
  if(hdl != -1) { // non-blocking 
    hdlList[hdl]->acked = 1;  
    if (hdlList[hdl]->wait == 1) {
      hdlList[hdl]->wait = 0;
      thread->resume();
    }
  }
  thread->resume();
}

void ArmciVirtualProcessor::putDataFromGets(ArmciStridedMsg *m){
  stridedCopy(m->dst, m->data, m->dst_stride_ar, m->count, m->stride_levels, 0);
  if(m->hdl != -1) { // non-blocking 
    hdlList[m->hdl]->acked = 1;  
    if (hdlList[m->hdl]->wait == 1) {
      hdlList[m->hdl]->wait = 0;
      thread->resume();
    }
  }
  delete m;
  thread->resume();
}

void ArmciVirtualProcessor::notify(int proc){
  thisProxy[proc].sendNote(thisIndex);
}
void ArmciVirtualProcessor::sendNote(int proc){
  // check if note exists
  // if so, decrement it and see if resume thread is appropriate
  // if not, create a new note
  int hasNote = -1;
  for(int i=0;i<noteList.size();i++){
    if(noteList[i]->proc == proc){
      hasNote = i;
      break;
    }
  }
  if(hasNote!=-1){
    (noteList[hasNote]->notified)++;
  } else {
    Armci_Note* newNote = new Armci_Note(proc, 0, 1);
    noteList.push_back(newNote);
    hasNote = noteList.size() - 1;
  }
  if(noteList[hasNote]->notified >= noteList[hasNote]->waited){
/*
    noteList[hasNote]->notified -= noteList[hasNote]->waited;
    noteList[hasNote]->waited = 0;
*/
    thread->resume();
  }
}
void ArmciVirtualProcessor::notify_wait(int proc){
  // check if note exists
  // if so, check if suspend is necessary
  // if not, create a waited note and suspend
  int hasNote = -1;
  for(int i=0;i<noteList.size();i++){
    if(noteList[i]->proc == proc){
      hasNote = i;
      break;
    }
  }
  if(hasNote!=-1){
    (noteList[hasNote]->waited)++;
  } else {
    Armci_Note* newNote = new Armci_Note(proc, 1, 0);
    noteList.push_back(newNote);
    hasNote = noteList.size() - 1;
  }
  if(noteList[hasNote]->notified < noteList[hasNote]->waited){
    thread->suspend();
  }
}

void ArmciVirtualProcessor::pup(PUP::er &p) {
  TCharmClient1D::pup(p);
  //Copying only address, the mempool will be pupped as part of the thread
#if CMK_USE_MEMPOOL_ISOMALLOC
  pup_bytes(&p, &memBlock, sizeof(CmiIsomallocBlockList*));
#else
  CmiIsomallocBlockListPup(&p, &memBlock, NULL);
#endif
  p|thisProxy;
  p|hdlList;
  p|noteList;
  CkPupMessage(p, (void **)&addressReply, 1);
}

// NOT an entry method. This is an object-interface to the API interface.
void ArmciVirtualProcessor::requestAddresses(pointer ptr, pointer ptr_arr[], int bytes) {
  int thisPE = armci_me;
  int numPE = armci_nproc;
  // reset the reply field
  addressReply = NULL;
  addressPair *pair = new addressPair;
  pair->pe = thisPE;
  pair->ptr = ptr;
  // do a reduction to get everyone else's data.
  CkCallback cb(CkIndex_ArmciVirtualProcessor::mallocClient(NULL),CkArrayIndex1D(0),thisProxy);
  contribute(sizeof(addressPair), pair, CkReduction::concat, cb);
  // wait for the reply to arrive.
  while(addressReply==NULL) thread->suspend();

  // copy the acquired data to the user-allocated array.
  for (int i=0; i<numPE; i++) {
    ptr_arr[i] = addressReply->addresses[i];
  }
  delete addressReply;
  addressReply = NULL;
}

void ArmciVirtualProcessor::stridedCopy(void *base, void *buffer_ptr,
		  int *stride, int *count, int stride_levels, bool flatten) {
  if (stride_levels == 0) {
    if (flatten) {
      memcpy(buffer_ptr, base, count[stride_levels]);
    } else {
      memcpy(base, buffer_ptr, count[stride_levels]);
    }
  } else {
    int mystride = 1;
    for(int i=0;i<stride_levels;i++)
      mystride *= count[i];
    for (int i=0; i<count[stride_levels]; i++) {
      stridedCopy((void *)((char *)base + stride[stride_levels-1]*i), 
		(void *)((char *)buffer_ptr + mystride*i), stride, count, stride_levels-1, flatten);
    }
  }
}

// malloc reduction client
void ArmciVirtualProcessor::mallocClient(CkReductionMsg *msg) {
  int numBlocks = msg->getSize()/sizeof(addressPair);
  addressPair *dataBlocks = (addressPair *)msg->getData();
  AddressMsg *addrmsg = new(numBlocks, 0) AddressMsg;
  // constructing the ordered set of shared pointers
  for (int i=0; i<numBlocks; i++) {
    addrmsg->addresses[dataBlocks[i].pe] = dataBlocks[i].ptr;
  }
  // broadcast the results to everyone.
  thisProxy.getAddresses(addrmsg);
  delete msg;
}

// **** CAF collective operations **** 

// **CWL**
//   Assumptions:
//   1. this operation blocks until data is ready.
//   2. buffer pointers can be different for each Virtual Processor.
//   3. len represents length of buffer in bytes.
void ArmciVirtualProcessor::msgBcast(void *buffer, int len, int root) {
  int me;
  ARMCI_Myid(&me);
  if (me == root) {
    thisProxy.recvMsgBcast(len, (char *)buffer, root);
  } else {
    // copy the buffer pointer to thread object
    collectiveTmpBufferPtr = buffer;
    thread->suspend();
  }
}

// **CWL** For now, we have to live with a double-copy
void ArmciVirtualProcessor::recvMsgBcast(int len, char *buffer, int root) {
  int me;
  ARMCI_Myid(&me);
  if (me != root) {
    // Copy broadcast buffer into the area of memory pointed to by
    //   buffer specified in the original broadcast collective and then
    //   setting the temporary thread object pointer back to NULL.
    collectiveTmpBufferPtr = memcpy(collectiveTmpBufferPtr, buffer, len);
    collectiveTmpBufferPtr = NULL;
    thread->resume();
  }
}

// **CWL**
//   Assumptions (seems true from ARMCI 1.4 implementation):
//   1. the root is always 0.
void ArmciVirtualProcessor::msgGop(void *x, int n, char *op, int type) {
  CkReduction::reducerType reducer;
  if (strcmp(op,"+") == 0) {
  } else if (strcmp(op,"*") == 0) {
  } else if (strcmp(op,"min") == 0) {
  } else if (strcmp(op,"max") == 0) {
  } else if (strcmp(op,"absmin") == 0) {
  } else if (strcmp(op,"absmax") == 0) {
  } else {
    CkPrintf("Operator %s not supported\n",op);
    CmiAbort("ARMCI ERROR: msgGop - Unknown operator\n");
  }
  switch (type) {
  case ARMCI_INT:
    
    break;
  case ARMCI_LONG:
    break;
  case ARMCI_LONG_LONG:
    break;
  case ARMCI_FLOAT:
    break;
  case ARMCI_DOUBLE:
    break;
  default:
    CkPrintf("ARMCI Type %d not supported\n", type);
    CmiAbort("ARMCI ERROR: msgGop - Unknown type\n");
  }
}

// reduction client data - preparation for checkpointing
class ckptClientStruct {
public:
  const char *dname;
  ArmciVirtualProcessor *vp;
  ckptClientStruct(const char *s, ArmciVirtualProcessor *p): dname(s), vp(p) {}
};

static void checkpointClient(void *param,void *msg)
{       
  ckptClientStruct *client = (ckptClientStruct*)param;
  const char *dname = client->dname;
  ArmciVirtualProcessor *vp = client->vp;
  vp->checkpoint(strlen(dname), dname);
  delete client;
}               
                
void ArmciVirtualProcessor::startCheckpoint(const char* dname){
  if (thisIndex==0) {
    ckptClientStruct *clientData = new ckptClientStruct(dname, this);
    CkCallback cb(checkpointClient, clientData);
    contribute(0, NULL, CkReduction::sum_int, cb);
  } else {
    contribute(0, NULL, CkReduction::sum_int);
  }
  thread->suspend();
}
void ArmciVirtualProcessor::checkpoint(int len, const char* dname){
  if (len == 0) { // memory checkpoint
    CkCallback cb(CkIndex_ArmciVirtualProcessor::resumeThread(),thisProxy);
    CkStartMemCheckpoint(cb);
  } else {
    char dirname[256];
    strncpy(dirname,dname,len);
    dirname[len]='\0';
    CkCallback cb(CkIndex_ArmciVirtualProcessor::resumeThread(),thisProxy);
    CkStartCheckpoint(dirname,cb);
  }
}

#include "armci.def.h"

