#include <vector>
#include "armci_impl.h"

using namespace std;

// FIXME: might be memory leakage in put
// This is the way to adapt a library's preferred start interface with the
// one provided by TCharm (eg. argc,argv vs void).
extern "C" void armciLibStart(void) {
  int argc=CkGetArgc();
  char **argv=CkGetArgv();
  ARMCI_Main_cpp(argc, argv);
}

// Default startup routine (can be overridden by user's own)
// This will be registered with TCharm's startup routine
// in the Node initialization function.
static void ArmciDefaultSetup(void) {
  // Create the base threads on TCharm using user-defined start routine.
  TCHARM_Create(TCHARM_Get_num_chunks(), armciLibStart);
}

CtvDeclare(ArmciVirtualProcessor *, _armci_ptr);

// Node initialization (made by initcall of the module armci)
void armciProcInit(void) {
  CtvInitialize(ArmciVirtualProcessor, _armci_ptr);
  CtvAccess(_armci_ptr) = NULL;

  // Register the library's default startup routine to TCharm
  TCHARM_Set_fallback_setup(ArmciDefaultSetup);
};

ArmciVirtualProcessor::ArmciVirtualProcessor(const CProxy_TCharm &_thr_proxy)
  : TCharmClient1D(_thr_proxy) {
  thisProxy = this;
  tcharmClientInit();
  thread->semaPut(ARMCI_TCHARM_SEMAID,this);
  memBlock = CmiIsomallocBlockListNew();
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
  CmiIsomallocBlockListDelete(memBlock);
  if (addressReply) {delete addressReply;}
}

void ArmciVirtualProcessor::setupThreadPrivate(CthThread forThread) {
  CtvAccessOther(forThread, _armci_ptr) = this;
  armci_nproc = thread->getNumElements();
}

void ArmciVirtualProcessor::getAddresses(AddressMessage *msg) {
  addressReply = msg;
  thread->resume();
}

// implemented as a blocking put for now
void ArmciVirtualProcessor::put(pointer src, pointer dst,
			       int nbytes, int dst_proc) {
  char *buffer;
  buffer = new char[nbytes];
  buffer = (char *)memcpy(buffer, src, nbytes);
  thisProxy[dst_proc].putData(dst, nbytes, buffer, thisIndex, -1);
  // blocking call. Wait for acknowledgement from target
  thread->suspend();
}

int ArmciVirtualProcessor::nbput(pointer src, pointer dst,
			       int nbytes, int dst_proc) {
  int hdl;
  char *buffer;

  Armci_Hdl* entry = new Armci_Hdl(ARMCI_PUT, dst_proc, nbytes, src, dst);
  hdlList.push_back(entry);
  hdl = hdlList.size() - 1;
  
  buffer = new char[nbytes];
  buffer = (char *)memcpy(buffer, src, nbytes);
  thisProxy[dst_proc].putData(dst, nbytes, buffer, thisIndex, hdl);

  return hdl;
}

void ArmciVirtualProcessor::putData(pointer dst, int nbytes, char *data,
				    int src_proc, int hdl) {
  memcpy(dst, data, nbytes);
  thisProxy[src_proc].putAck(hdl);
}

void ArmciVirtualProcessor::putAck(int hdl) {
  if(hdl != -1) { // non-blocking 
    hdlList[hdl]->acked = 1;  
  }
  thread->resume();
}

void ArmciVirtualProcessor::get(pointer src, pointer dst,
			       int nbytes, int src_proc) {
  thisProxy[src_proc].requestFromGet(src, dst, nbytes, thisIndex, -1);
  // wait for reply
  thread->suspend();
}

int ArmciVirtualProcessor::nbget(pointer src, pointer dst,
			       int nbytes, int src_proc) {
  int hdl;  
  Armci_Hdl* entry = new Armci_Hdl(ARMCI_GET, src_proc, nbytes, src, dst);
  hdlList.push_back(entry);
  hdl = hdlList.size() - 1;
  
  thisProxy[src_proc].requestFromGet(src, dst, nbytes, thisIndex, hdl);

  return hdl;
}

void ArmciVirtualProcessor::wait(int hdl){
  while (1) {
    if(hdlList[hdl]->acked != 0)
      break;
    else
      thread->suspend();
  }
}

void ArmciVirtualProcessor::waitmulti(vector<int> procs){
  for(int i=0;i<procs.size();i++){
    wait(procs[i]);
  }
}

void ArmciVirtualProcessor::waitproc(int proc){
  vector<int> procs;
  for(int i=0;i<hdlList.size();i++){
    if(hdlList[i]->proc == proc)
      procs.push_back(i);
  }
  waitmulti(procs);
}

void ArmciVirtualProcessor::waitall(){
  vector<int> procs;
  for(int i=0;i<hdlList.size();i++){
    procs.push_back(i);
  }
  waitmulti(procs);
}

void ArmciVirtualProcessor::fence(int proc){
  vector<int> procs;
  for(int i=0;i<hdlList.size();i++){
    if((hdlList[i]->op == ARMCI_PUT || hdlList[i]->op == ARMCI_ACC) && hdlList[i]->proc == proc)
      procs.push_back(i);
  }
  waitmulti(procs);
}
void ArmciVirtualProcessor::allfence(){
  vector<int> procs;
  for(int i=0;i<hdlList.size();i++){
    if(hdlList[i]->op == ARMCI_PUT || hdlList[i]->op == ARMCI_ACC)
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
  return hdlList[hdl]->acked;
}

void ArmciVirtualProcessor::requestFromGet(pointer src, pointer dst, int nbytes,
				       int dst_proc, int hdl) {
  char *buffer;
  buffer = new char[nbytes];
  buffer = (char *)memcpy(buffer, src, nbytes);
  thisProxy[dst_proc].putDataFromGet(dst, nbytes, buffer, hdl);
}

// this is essentially the same as putData except that no acknowledgement
// is required and the thread suspended while waiting for the data is
// awoken.
void ArmciVirtualProcessor::putDataFromGet(pointer dst, int nbytes, char *data, int hdl) {
  memcpy(dst, data, nbytes);
  if(hdl != -1) { // non-blocking 
    hdlList[hdl]->acked = 1;  
  }
  thread->resume();
}

void ArmciVirtualProcessor::puts(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[],
	   int count[], int stride_levels, int dst_proc){
  char *buffer;
  int nbytes = 1;
  for(int i=0;i<stride_levels+1;i++) 
    nbytes *= count[i];
  buffer = new char[nbytes];
  buffer = (char *)stridedCopy(src_ptr, buffer, src_stride_ar, count, stride_levels, 1);
  thisProxy[dst_proc].putsData(dst_ptr, dst_stride_ar, count, stride_levels, nbytes, buffer, thisIndex, -1);
  // blocking call. Wait for acknowledgement from dst_proc
  thread->suspend();
}
int ArmciVirtualProcessor::nbputs(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[],
	   int count[], int stride_levels, int dst_proc){
  int hdl;
  char *buffer;
  int nbytes = 1;
  for(int i=0;i<stride_levels+1;i++) 
    nbytes *= count[i];
  
  Armci_Hdl* entry = new Armci_Hdl(ARMCI_PUT, dst_proc, nbytes, src_ptr, dst_ptr);
  hdlList.push_back(entry);
  hdl = hdlList.size() - 1;
  
  buffer = new char[nbytes];
  buffer = (char *)stridedCopy(src_ptr, buffer, src_stride_ar, count, stride_levels, 1);
  thisProxy[dst_proc].putsData(dst_ptr, dst_stride_ar, count, stride_levels, nbytes, buffer, thisIndex, hdl);
  
  return hdl;
}
void ArmciVirtualProcessor::putsData(pointer dst_ptr, int dst_stride_ar[], 
  		int count[], int stride_levels,
		int nbytes, char *data, int src_proc, int hdl){
  stridedCopy(dst_ptr, data, dst_stride_ar, count, stride_levels, 0);
  thisProxy[src_proc].putAck(hdl);
}

void ArmciVirtualProcessor::gets(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[],
	   int count[], int stride_levels, int src_proc){
  thisProxy[src_proc].requestFromGets(src_ptr, src_stride_ar, dst_ptr, dst_stride_ar, 
  					count, stride_levels, thisIndex, -1);
  // wait for reply
  thread->suspend();
}
int ArmciVirtualProcessor::nbgets(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[],
	   int count[], int stride_levels, int src_proc){
  int hdl;  
  int nbytes = 1;
  for(int i=0;i<stride_levels+1;i++) 
    nbytes *= count[i];
  Armci_Hdl* entry = new Armci_Hdl(ARMCI_GET, src_proc, nbytes, src_ptr, dst_ptr);
  hdlList.push_back(entry);
  hdl = hdlList.size() - 1;

  thisProxy[src_proc].requestFromGets(src_ptr, src_stride_ar, dst_ptr, dst_stride_ar, 
  					count, stride_levels, thisIndex, hdl);

  return hdl;
}
void ArmciVirtualProcessor::requestFromGets(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[], int count[], int stride_levels, int dst_proc, int hdl){
  char *buffer;
  int nbytes = 1;
  for(int i=0;i<stride_levels+1;i++) 
    nbytes *= count[i];
  buffer = new char[nbytes];
  buffer = (char *)stridedCopy(src_ptr, buffer, src_stride_ar, count, stride_levels, 1);
  thisProxy[dst_proc].putDataFromGets(dst_ptr, dst_stride_ar, count, stride_levels, nbytes, buffer, hdl);
}
void ArmciVirtualProcessor::putDataFromGets(pointer dst_ptr, int dst_stride_ar[], 
		int count[], int stride_levels, int nbytes, char *data, int hdl){
  stridedCopy(dst_ptr, data, dst_stride_ar, count, stride_levels, 0);
  if(hdl != -1) { // non-blocking 
    hdlList[hdl]->acked = 1;  
  }
  thread->resume();
}

void ArmciVirtualProcessor::notify(int proc){
  thisProxy[proc].sendNote(thisIndex);
}
void ArmciVirtualProcessor::sendNote(int proc){
  // check if unacked note exists
  // if so, resume thread and remove the note
  // if not, create an acked note
  int hasNote = -1;
  for(int i=0;i<noteList.size();i++){
    if(noteList[i]->proc == proc){
      CkAssert(noteList[i]->acked == 0);
      hasNote = i;
      break;
    }
  }
  if(hasNote!=-1){
    delete noteList[hasNote];
    noteList.remove(hasNote);
    thread->resume();
  } else {
    Armci_Note* newNote = new Armci_Note(proc, 1);
    noteList.push_back(newNote);
  }
}
void ArmciVirtualProcessor::notify_wait(int proc){
  // check if notify already arrived
  // if so, remove it and continue
  // if not, create unacked note and suspend
  int hasNote = -1;
  for(int i=0;i<noteList.size();i++){
    if(noteList[i]->proc == proc){
      CkAssert(noteList[i]->acked == 1);
      hasNote = i;
      break;
    }
  }
  if(hasNote!=-1){
    delete noteList[hasNote];
    noteList.remove(hasNote);
  } else {
    Armci_Note* newNote = new Armci_Note(proc, 0);
    noteList.push_back(newNote);
    thread->suspend();
  }
}

void ArmciVirtualProcessor::pup(PUP::er &p) {
  TCharmClient1D::pup(p);
  CmiIsomallocBlockListPup(&p, &memBlock);
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
ckout << "[" << thisIndex << "]malloced " << bytes << " bytes starting " << ptr << endl;
  pair->pe = thisPE;
  pair->ptr = ptr;
  // do a reduction to get everyone else's data.
  CkCallback cb(CkIndex_ArmciVirtualProcessor::mallocClient(NULL),CkArrayIndex1D(0),thisProxy);
  contribute(sizeof(addressPair), pair, CkReduction::concat, cb);
  // wait for the reply to arrive.
  thread->suspend();

  // copy the acquired data to the user-allocated array.
  for (int i=0; i<numPE; i++) {
    ptr_arr[i] = addressReply->addresses[i];
  }
  delete addressReply;
  addressReply = NULL;
}

void* ArmciVirtualProcessor::stridedCopy(void *base, void *buffer_ptr,
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
      stridedCopy((void *)((char *)base + stride[stride_levels-1]*i), (void *)((char *)buffer_ptr + mystride*i), stride, 
			       count, stride_levels-1, flatten);
    }
  }
  return buffer_ptr;
}

// malloc reduction client
void ArmciVirtualProcessor::mallocClient(CkReductionMsg *msg) {
  int numBlocks = msg->getSize()/sizeof(addressPair);
  addressPair *dataBlocks = (addressPair *)msg->getData();
  AddressMessage *addrmsg = new(numBlocks, 0) AddressMessage;
  // constructing the ordered set of shared pointers
  for (int i=0; i<numBlocks; i++) {
    addrmsg->addresses[dataBlocks[i].pe] = dataBlocks[i].ptr;
  }
  // broadcast the results to everyone.
  thisProxy.getAddresses(addrmsg);
  delete msg;
}

#include "armci.def.h"

