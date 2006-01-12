#ifndef _ARMCI_IMPL_H
#define _ARMCI_IMPL_H

#include <vector>
using std::vector;

#include "tcharmc.h"
#include "tcharm.h"

//Types needed for remote method parameters:
typedef void* pointer;
PUPbytes(pointer); //Pointers get sent as raw bytes

#include "armci.decl.h"
#include "armci.h"

/* Operations for Armci_Hdl */
#define ARMCI_INVALID	0
#define ARMCI_GET 	1
#define ARMCI_PUT 	2
#define ARMCI_ACC 	3

class Armci_Hdl {
public:
   int op;
   int proc;
   int nbytes;
   int acked;
   pointer src;
   pointer dst;
   
   Armci_Hdl() : op(ARMCI_INVALID), proc(-1), nbytes(0), acked(0), src(NULL), dst(NULL) 
   	{ }
   Armci_Hdl(int o, int p, int n, pointer s, pointer d):
   	op(o), proc(p), nbytes(n), acked(0), src(s), dst(d) { }
   void pup(PUP::er &p){
     p|op; p|proc; p|nbytes; p|acked; p|src; p|dst;	
   }
};

class Armci_Note{
public:
  int proc;
  int acked;
  Armci_Note() : proc(-1), acked(0) { }
  Armci_Note(int p, int a) : proc(p), acked(a) { }
  void pup(PUP::er &p){ p|proc; p|acked; }
};

// structure definitions and forward declarations (for reductions)
typedef struct peAddr {
  int pe;
  pointer ptr;
} addressPair;

extern CkArrayID armciVPAid;

#define ARMCI_TCHARM_SEMAID 0x00A53C10 /* __ARMCI_ */

// virtual processor class declaration
// ARMCI is supposed to be platform neutral, so calling this a thread did
// not seem like a proper abstraction.
class ArmciVirtualProcessor : public TCharmClient1D {
  CmiIsomallocBlockList *memBlock;
  CProxy_ArmciVirtualProcessor thisProxy;
  AddressMessage *addressReply;
  CkPupPtrVec<Armci_Hdl> hdlList;
  CkPupPtrVec<Armci_Note> noteList;
 protected:
  virtual void setupThreadPrivate(CthThread forThread);
 public:
  ArmciVirtualProcessor(const CProxy_TCharm &_thr_proxy);
  ArmciVirtualProcessor(CkMigrateMessage *m);
  ~ArmciVirtualProcessor();
  
  pointer BlockMalloc(int bytes) { return (void *)CmiIsomallocBlockListMalloc(memBlock, bytes); }
  void getAddresses(AddressMessage *msg);

  void put(pointer src, pointer dst, int bytes, int dst_proc);
  void putData(pointer dst, int nbytes, char *data, int src_proc, int hdl);
  void putAck(int hdl);
  int nbput(pointer src, pointer dst, int bytes, int dst_proc);
  void wait(int hdl);
  int test(int hdl);
  void waitmulti(vector<int> procs);
  void waitproc(int proc);
  void waitall();
  void fence(int proc);
  void allfence();
  void barrier();
  
  void get(pointer src, pointer dst, int bytes, int src_proc);
  int nbget(pointer src, pointer dst, int bytes, int dst_proc);
  void requestFromGet(pointer src, pointer dst, int nbytes, int dst_proc, int hdl);
  void putDataFromGet(pointer dst, int nbytes, char *data, int hdl);

  void puts(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[],
	   int count[], int stride_levels, int dst_proc);
  int nbputs(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[],
	   int count[], int stride_levels, int dst_proc);
  void putsData(pointer dst_ptr, int dst_stride_ar[], 
  		int count[], int stride_levels,
		int nbytes, char *data, int src_proc, int hdl);
  
  void gets(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[],
	   int count[], int stride_levels, int src_proc);
  int nbgets(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[],
	   int count[], int stride_levels, int src_proc);
  void requestFromGets(pointer src_ptr, int src_stride_ar[], 
	   pointer dst_ptr, int dst_stride_ar[],
	   int count[], int stride_levels, int dst_proc, int hdl);
  void putDataFromGets(pointer dst_ptr, int dst_stride_ar[], 
  		int count[], int stride_levels,
		int nbytes, char *data, int hdl);

  void notify(int proc);
  void sendNote(int proc);
  void notify_wait(int proc);

  // non-entry methods. Mainly interfaces to API interface methods.
  void requestAddresses(pointer  ptr, pointer ptr_arr[], int bytes);
  void stridedCopy(void *base, void *buffer_ptr,
		  int *stride, int *count, 
		  int dim_id, bool flatten);
  virtual void pup(PUP::er &p);
  
  void mallocClient(CkReductionMsg *msg);
  void resumeThread(void);
};

class AddressMessage : public CMessage_AddressMessage {
 public:
  pointer *addresses;
  friend class CMessage_AddressMessage;
};

// pointer to the current tcshmem thread. Needed to regain context after
// getting called by user.
CtvExtern(ArmciVirtualProcessor *, _armci_ptr);

#endif
