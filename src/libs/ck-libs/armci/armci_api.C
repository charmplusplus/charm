// APIs exposed in armci.h. These should be called from within the driver
// code "armciStart". Applications running the armci library MUST use
// the runtime option -memory isomalloc.

#include "armci_impl.h"

// API implementations

int armci_nproc;

// initialization api

// Bind the virtual processors in the armci library to TCharm's.
// This is called by the user's thread when it starts up.
CDECL int ARMCI_Init(void) {
  if (TCHARM_Element()==0) {
    CkArrayID _tc_aid;
    CkArrayOptions opt = TCHARM_Attach_start(&_tc_aid, NULL);
    CkArrayID aid = CProxy_ArmciVirtualProcessor::ckNew(_tc_aid, opt);
    CProxy_ArmciVirtualProcessor vpProxy = CProxy_ArmciVirtualProcessor(aid);
    // FIXME: should do reductions to element 0 of the array, not some bizarre malloc'd thing.
    CkArrayID *clientAid = new CkArrayID;
    *clientAid = aid;
    vpProxy.setReductionClient(mallocClient, (void *)clientAid);
  }
  
  ArmciVirtualProcessor *vp=(ArmciVirtualProcessor *)
  	TCharm::get()->semaGet(ARMCI_TCHARM_SEMAID);
  return 0;
}

CDECL int ARMCI_Finalize(void) {
  return 0;
}

CDECL void ARMCI_Cleanup(void) {
}

CDECL void ARMCI_Error(char *message, int code) {
  ckerr << "armci error: " << message << " | code = " << code << endl;
}

CDECL int ARMCI_GetV(
	        armci_giov_t darr[], /* descriptor array */
	        int len,              /* length of descriptor array */
	        int proc              /* remote process(or) ID */
	        ){
  return 0;
}

CDECL int ARMCI_NbGetV(
		armci_giov_t *dsrc_arr,
		int arr_len,
		int proc,
		armci_hdl_t* handle
		){
  return 0;
}

CDECL int ARMCI_PutV(
	        armci_giov_t darr[], /* descriptor array */
	        int len,              /* length of descriptor array */
	        int proc              /* remote process(or) ID */
	        ){
  return 0;
}

CDECL int ARMCI_NbPutV(
		armci_giov_t *dsrc_arr,
		int arr_len,
		int proc,
		armci_hdl_t* handle
	         ){
  return 0;
}

CDECL int ARMCI_AccV(
	        int op,                /* operation code */
	        void *scale,          /* scaling factor for accumulate */
	        armci_giov_t darr[], /* descriptor array */
	        int len,              /* length of descriptor array */
	        int proc              /* remote process(or) ID */
	        ){
  return 0;
}
        
CDECL int ARMCI_NbAccV(
		int datatype, 
		void *scale, 
		armci_giov_t *dsrc_arr, 
		int arr_len, 
		int proc, 
		armci_hdl_t* handle
		){
  return 0;
}

// src is local memory, dst is remote address
CDECL int ARMCI_Put(void *src, void *dst, int bytes, int proc) {
  TCHARM_API_TRACE("ARMCI_Put", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  return vp->put(src, dst, bytes, proc);
}

CDECL int ARMCI_NbPut(void *src, void* dst, int bytes, int proc, armci_hdl_t *handle){
  return 0;
}

// src is remote memory addr, dst is local address
CDECL int ARMCI_Get(void *src, void *dst, int bytes, int proc) {
  TCHARM_API_TRACE("ARMCI_Get", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  return vp->get(src, dst, bytes, proc);
}

CDECL int ARMCI_NbGet(void *src, void* dst, int bytes, int proc, armci_hdl_t *handle){
  return 0;
}

CDECL int ARMCI_Acc(int datatype, void *scale, void* src, void* dst, int bytes, int proc){
  return 0;
}

CDECL int ARMCI_NbAcc(int datatype, void *scale, void* src, void* dst, int bytes, int proc, 
                  armci_hdl_t* ) {
  return 0;
}

// strided copy operations
CDECL int ARMCI_PutS(void *src_ptr, int src_stride_ar[], 
	        void *dst_ptr, int dst_stride_ar[],
	        int count[], int stride_levels, int proc) {
  char *buffer;
  
  return 0;
}

CDECL int ARMCI_NbPutS(
		 void *src_ptr,         /* ptr to 1st segment at source */
		 int src_stride_arr[], /* array of strides at source  */
		 void* dst_ptr,         /* ptr to 1st segment at dest */
		 int dst_stride_arr[], /* array of strides at destination */
		 int count[],           /* number of units at each stride  */
		                          /* level count[0]=bytes  */
		 int stride_levels,    /* number of stride levels */
		 int proc,              /* remote process(or) ID  */
		 armci_hdl_t *handle   /* pointer to descriptor associated */
		                          /* with a particular non-blocking */
                                            /* transfer. Passing NULL value */
		                          /* makes this function do an */
                                            /* implicit handle non-blocking */
                                            /* transfer */
		 ){
  return 0;
}

CDECL int ARMCI_GetS(
	        void *src_ptr,         /* pointer to 1st segment at source */
	        int src_stride_arr[], /* array of strides at source */
	        void* dst_ptr,         /* ptr to 1st segment at destination */
	        int dst_stride_arr[], /* array of strides at destination */
	        int count[],           /* number of units at each stride  */
	                                 /* level count[0]=bytes */
	        int stride_levels,    /* number of stride levels */
	        int proc                /* remote process(or) ID */
	        ){
  return 0;
}

CDECL int ARMCI_NbGetS(
	        void *src_ptr,         /* pointer to 1st segment at source */
	        int src_stride_arr[], /* array of strides at source */
	        void* dst_ptr,         /* ptr to 1st segment at destination */
	        int dst_stride_arr[], /* array of strides at destination */
	        int count[],           /* number of units at each stride  */
	                                 /* level count[0]=bytes */
	        int stride_levels,    /* number of stride levels */
	        int proc,              /* remote process(or) ID  */
		armci_hdl_t *handle   /* pointer to descriptor associated */
		                          /* with a particular non-blocking */
                                            /* transfer. Passing NULL value */
		                          /* makes this function do an */
                                            /* implicit handle non-blocking */
                                            /* transfer */
	        ){
  return 0;
}

CDECL int ARMCI_AccS(
	        int  optype,           /* operation */
	        void *scale,           /* scale factor x += scale*y */
	        void *src_ptr,         /* pointer to 1st segment at source */
	        int src_stride_arr[], /* array of strides at source */
	        void* dst_ptr,         /* ptr to 1st segment at destination */
	        int dst_stride_arr[], /* array of strides at destination */
	        int count[],           /* number of units at each stride  */
	                                 /* level count[0]=bytes */
	        int stride_levels,    /* number of stride levels */
	        int proc                /* remote process(or) ID */
	        ){
  return 0;
}

CDECL int ARMCI_NbAccS(
	        int  optype,           /* operation */
	        void *scale,           /* scale factor x += scale*y */
	        void *src_ptr,         /* pointer to 1st segment at source */
	        int src_stride_arr[], /* array of strides at source */
	        void* dst_ptr,         /* ptr to 1st segment at destination */
	        int dst_stride_arr[], /* array of strides at destination */
	        int count[],           /* number of units at each stride  */
	                                 /* level count[0]=bytes */
	        int stride_levels,    /* number of stride levels */
		int proc,             /* remote process(or) ID  */
		armci_hdl_t *handle   /* pointer to descriptor associated */
		                       /* with a particular non-blocking */
                                       /* transfer. Passing NULL value */
		                       /* makes this function do an */
                                       /* implicit handle non-blocking */
                                       /* transfer */
	        ){
  return 0;
}

CDECL int ARMCI_PutValueLong(long src, void* dst, int proc) { return 0; }
CDECL int ARMCI_PutValueInt(int src, void* dst, int proc) { return 0; }
CDECL int ARMCI_PutValueFloat(float src, void* dst, int proc) { return 0; }
CDECL int ARMCI_PutValueDouble(double src, void* dst, int proc) { return 0; }
CDECL int ARMCI_NbPutValueLong(long src, void* dst, int proc, armci_hdl_t* handle) { return 0; }
CDECL int ARMCI_NbPutValueInt(int src, void* dst, int proc, armci_hdl_t* handle) { return 0; }
CDECL int ARMCI_NbPutValueFloat(float src, void* dst, int proc, armci_hdl_t* handle) { return 0; }
CDECL int ARMCI_NbPutValueDouble(double src, void* dst, int proc, armci_hdl_t* handle) { return 0; }
CDECL long ARMCI_GetValueLong(void *src, int proc) { return 0; }
CDECL int ARMCI_GetValueInt(void *src, int proc) { return 0; }
CDECL float ARMCI_GetValueFloat(void *src, int proc) { return 0.0; }
CDECL double ARMCI_GetValueDouble(void *src, int proc) { return 0.0; }

// global completion operations
CDECL int ARMCI_Wait(armci_hdl_t *handle){
  return 0;
}

CDECL int ARMCI_WaitProc(int proc){
  return 0;
}

CDECL int ARMCI_WaitAll(){
  return 0;
}

CDECL int ARMCI_Test(armci_hdl_t *handle){
  return 0;
}

CDECL int ARMCI_Barrier(){
  return 0;
}

// these are no-ops because Put is blocking
CDECL int ARMCI_Fence(int proc) {
  return 0;
}

CDECL int ARMCI_AllFence(void) {
  return 0;
}

// memory operations

// malloc is a collective operation. The user is expected to allocate
// and manage ptr_arr.
CDECL int ARMCI_Malloc(void *ptr_arr[], int bytes) {
  TCHARM_API_TRACE("ARMCI_Malloc", "armci");
  // shift work off to entry method for split-phase communication.
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  // requestAddresses is called to bridge the split-phase gap at the
  // virtual processor object.
  return vp->requestAddresses(ptr_arr, bytes);
}

// CmiIsomalloc does not return a value and no indication is given about
// the success nor failure of the operation. Hence, it is assumed always
// that free works.
CDECL int ARMCI_Free(void *address) {
  TCHARM_API_TRACE("ARMCI_Free", "armci");
  CmiIsomallocBlockListFree(address);
  return 0;
}
CDECL void *ARMCI_Malloc_local(int bytes){
  return NULL;
}

CDECL int ARMCI_Free_local(void *ptr){
  return 0;
}

CDECL void ARMCI_SET_AGGREGATE_HANDLE (armci_hdl_t* handle) { }
CDECL void ARMCI_UNSET_AGGREGATE_HANDLE (armci_hdl_t* handle) { }

CDECL int ARMCI_Rmw(int op, int *ploc, int *prem, int extra, int proc){
  return 0;
}

CDECL int ARMCI_Create_mutexes(int num){
  return 0;
}
CDECL int ARMCI_Destroy_mutexes(void){
  return 0;
}
CDECL void ARMCI_Lock(int mutex, int proc){
}
CDECL void ARMCI_Unlock(int mutex, int proc){
}

CDECL int armci_notify(int proc){
  return 0;
}

CDECL int armci_notify_wait(int proc, int *pval){
  return 0;
}


