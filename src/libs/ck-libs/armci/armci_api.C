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
  TCHARM_API_TRACE("ARMCI_Init", "armci");
  if (TCHARM_Element()==0) {
    CkArrayID threadsAID;
    int nChunks;
    CkArrayOptions opts = TCHARM_Attach_start(&threadsAID, &nChunks);
    CkArrayID aid = CProxy_ArmciVirtualProcessor::ckNew(threadsAID, opts);
    CProxy_ArmciVirtualProcessor vpProxy = CProxy_ArmciVirtualProcessor(aid);
  }
  
  ArmciVirtualProcessor *vp=(ArmciVirtualProcessor *)
  	TCharm::get()->semaGet(ARMCI_TCHARM_SEMAID);
  return 0;
}

CDECL int ARMCI_Finalize(void) {
  TCHARM_API_TRACE("ARMCI_Finalize", "armci");
  TCHARM_Done();
  return 0;
}

CDECL void ARMCI_Cleanup(void) {
  TCHARM_API_TRACE("ARMCI_Cleanup", "armci");
}

CDECL void ARMCI_Error(char *message, int code) {
  TCHARM_API_TRACE("ARMCI_Error", "armci");
  ckerr << "armci error: " << message << " | code = " << code << endl;
}


CDECL int ARMCI_Procs(int *procs){
  TCHARM_API_TRACE("ARMCI_Procs", "armci");
  *procs = TCHARM_Num_elements();
  return 0;
}
CDECL int ARMCI_Myid(int *myid){
  TCHARM_API_TRACE("ARMCI_Myid", "armci");
  *myid = TCHARM_Element();
  return 0;
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
  vp->put(src, dst, bytes, proc);
  return 0;
}

CDECL int ARMCI_NbPut(void *src, void* dst, int bytes, int proc, armci_hdl_t *handle){
  TCHARM_API_TRACE("ARMCI_NbPut", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  if (handle != NULL) {
    *handle = vp->nbput(src, dst, bytes, proc);
  } else {
    vp->nbput_implicit(src, dst, bytes, proc);
  }
  return 0;
}

// src is remote memory addr, dst is local address
CDECL int ARMCI_Get(void *src, void *dst, int bytes, int proc) {
  TCHARM_API_TRACE("ARMCI_Get", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->get(src, dst, bytes, proc);
  return 0;
}

CDECL int ARMCI_NbGet(void *src, void* dst, int bytes, int proc, armci_hdl_t *handle){
  TCHARM_API_TRACE("ARMCI_NbGet", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  if (handle != NULL) {
    *handle = vp->nbget(src, dst, bytes, proc);
  } else {
    vp->nbget_implicit(src, dst, bytes, proc);
  }
  return 0;
}

CDECL int ARMCI_Acc(int datatype, void *scale, void* src, void* dst, int bytes, int proc){
  return 0;
}

CDECL int ARMCI_NbAcc(int datatype, void *scale, void* src, void* dst, int bytes, int proc, armci_hdl_t* handle) {
  return 0;
}

// strided copy operations
CDECL int ARMCI_PutS(void *src_ptr, int src_stride_ar[], 
	        void *dst_ptr, int dst_stride_ar[],
	        int count[], int stride_levels, int proc) {
  TCHARM_API_TRACE("ARMCI_PutS", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->puts(src_ptr, src_stride_ar, dst_ptr, dst_stride_ar, count, stride_levels, proc);
  return 0;
}

CDECL int ARMCI_NbPutS(
		 void *src_ptr,         /* ptr to 1st segment at source */
		 int src_stride_ar[], /* array of strides at source  */
		 void* dst_ptr,         /* ptr to 1st segment at dest */
		 int dst_stride_ar[], /* array of strides at destination */
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
  TCHARM_API_TRACE("ARMCI_NbPutS", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  if (handle != NULL) {
    *handle = vp->nbputs(src_ptr, src_stride_ar, dst_ptr, dst_stride_ar, 
			 count, stride_levels, proc);
  } else {
    vp->nbputs_implicit(src_ptr, src_stride_ar, dst_ptr, dst_stride_ar,
			count, stride_levels, proc);
  }
  return 0;
}

CDECL int ARMCI_GetS(
	        void *src_ptr,         /* pointer to 1st segment at source */
	        int src_stride_ar[], /* array of strides at source */
	        void* dst_ptr,         /* ptr to 1st segment at destination */
	        int dst_stride_ar[], /* array of strides at destination */
	        int count[],           /* number of units at each stride  */
	                                 /* level count[0]=bytes */
	        int stride_levels,    /* number of stride levels */
	        int proc                /* remote process(or) ID */
	        ){
  TCHARM_API_TRACE("ARMCI_GetS", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->gets(src_ptr, src_stride_ar, dst_ptr, dst_stride_ar, count, stride_levels, proc);
  return 0;
}

CDECL int ARMCI_NbGetS(
	        void *src_ptr,         /* pointer to 1st segment at source */
	        int src_stride_ar[], /* array of strides at source */
	        void* dst_ptr,         /* ptr to 1st segment at destination */
	        int dst_stride_ar[], /* array of strides at destination */
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
  TCHARM_API_TRACE("ARMCI_NbGetS", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  if (handle != NULL) {
    *handle = vp->nbgets(src_ptr, src_stride_ar, dst_ptr, dst_stride_ar, 
			 count, stride_levels, proc);
  } else {
    vp->nbgets_implicit(src_ptr, src_stride_ar, dst_ptr, dst_stride_ar,
			count, stride_levels, proc);
  }
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
CDECL long ARMCI_NbGetValueLong(void *src, int proc, armci_hdl_t* handle) { return 0; }
CDECL int ARMCI_NbGetValueInt(void *src, int proc, armci_hdl_t* handle) { return 0; }
CDECL float ARMCI_NbGetValueFloat(void *src, int proc, armci_hdl_t* handle) { return 0.0; }
CDECL double ARMCI_NbGetValueDouble(void *src, int proc, armci_hdl_t* handle) { return 0.0; }

// global completion operations
CDECL int ARMCI_Wait(armci_hdl_t *handle){
  TCHARM_API_TRACE("ARMCI_Wait", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  if (handle != NULL) {
    vp->wait(*handle);
  } else {
    CmiAbort("ARMCI ERROR: Cannot pass NULL to ARMCI_Wait\n");
  }
  return 0;
}

CDECL int ARMCI_WaitProc(int proc){
  TCHARM_API_TRACE("ARMCI_WaitProc", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->waitproc(proc);
  return 0;
}

CDECL int ARMCI_WaitAll(){
  TCHARM_API_TRACE("ARMCI_WaitAll", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->waitall();
  return 0;
}

CDECL int ARMCI_Test(armci_hdl_t *handle){
  TCHARM_API_TRACE("ARMCI_Test", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  if(vp->test(*handle))
    return 0;
  else
    return 1;
}

CDECL int ARMCI_Barrier(){
  TCHARM_API_TRACE("ARMCI_Barrier", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->barrier();
  return 0;
}

// these are no-ops because Put is blocking
CDECL int ARMCI_Fence(int proc) {
  TCHARM_API_TRACE("ARMCI_Fence", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->fence(proc);
  return 0;
}

CDECL int ARMCI_AllFence(void) {
  TCHARM_API_TRACE("ARMCI_AllFence", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->allfence();
  return 0;
}

// memory operations

// malloc is a collective operation. The user is expected to allocate
// and manage ptr_arr.
CDECL int ARMCI_Malloc(void *ptr_arr[], armci_size_t bytes) {
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
//  pointer ptr = malloc(bytes);
  pointer ptr = vp->BlockMalloc(bytes);
  // TCHARM_API_TRACE disables isomalloc, so malloc first.
  TCHARM_API_TRACE("ARMCI_Malloc", "armci");
  vp->requestAddresses(ptr, ptr_arr, bytes);
  return 0;  
}

// CmiIsomalloc does not return a value and no indication is given about
// the success nor failure of the operation. Hence, it is assumed always
// that free works.
CDECL int ARMCI_Free(void *address) {
  CmiIsomallocBlockListFree(address);
//  free(address);
  TCHARM_API_TRACE("ARMCI_Free", "armci");
  return 0;
}
CDECL void *ARMCI_Malloc_local(armci_size_t bytes){
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  pointer ptr = vp->BlockMalloc(bytes);
  TCHARM_API_TRACE("ARMCI_Malloc_local", "armci");
  //return malloc(bytes);
  return ptr;
}

CDECL int ARMCI_Free_local(void *ptr){
  CmiIsomallocBlockListFree(ptr);
  TCHARM_API_TRACE("ARMCI_Free_local", "armci");
  //free(ptr);
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
  TCHARM_API_TRACE("armci_notify", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->notify(proc);
  return 0;
}

CDECL int armci_notify_wait(int proc, int *pval){
  TCHARM_API_TRACE("armci_notify_wait", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->notify_wait(proc);
  return 0;
}

/* ********************************* */
/* Collective Operations             */
CDECL void armci_msg_brdcst(void *buffer, int len, int root) {
  armci_msg_bcast(buffer, len, root);
}

CDECL void armci_msg_bcast(void *buffer, int len, int root) {
  TCHARM_API_TRACE("armci_msg_bcast", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->msgBcast(buffer, len, root);
}

// This does not look like an API actually used in ARMCI
CDECL void armci_msg_gop2(void *x, int n, int type, char *op) {
}

CDECL void armci_msg_igop(int *x, int n, char *op) {
  TCHARM_API_TRACE("armci_msg_dgop", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->msgGop(x, n, op, ARMCI_INT);
}

CDECL void armci_msg_lgop(CmiInt8 *x, int n, char *op) {
  TCHARM_API_TRACE("armci_msg_lgop", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->msgGop(x, n, op, ARMCI_LONG);
}

/*
CDECL void armci_msg_llgop(long long *x, int n, char *op) {
  TCHARM_API_TRACE("armci_msg_llgop", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->msgGop(x, n, op, ARMCI_LONG_LONG);
}
*/

CDECL void armci_msg_fgop(float *x, int n, char *op) {
  TCHARM_API_TRACE("armci_msg_fgop", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->msgGop(x, n, op, ARMCI_FLOAT);
}

CDECL void armci_msg_dgop(double *x, int n, char *op) {
  TCHARM_API_TRACE("armci_msg_dgop", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->msgGop(x, n, op, ARMCI_DOUBLE);
}

CDECL void armci_msg_barrier(void) {
}

CDECL void armci_msg_reduce(void *x, int n, char *op, int type) {
}

/* ******************************* */
/* System Configuration            */
CDECL int armci_domain_nprocs(armci_domain_t domain, int id) {
  return -1;
}

CDECL int armci_domain_count(armci_domain_t domain) {
  return -1;
}

CDECL int armci_domain_id(armci_domain_t domain, int glob_proc_id) {
  return -1;
}

CDECL int armci_domain_glob_proc_id(armci_domain_t domain, int id, 
				    int loc_proc_id) {
  return -1;
}

CDECL int armci_domain_my_id(armci_domain_t domain) {
  return -1;
}

/* ********************************** */
/* Charm++ Runtime Support Extensions */

CDECL void ARMCI_Migrate(void){
  TCHARM_API_TRACE("ARMCI_Migrate", "armci");
  TCHARM_Migrate();
}
CDECL void ARMCI_Async_Migrate(void){
  TCHARM_API_TRACE("ARMCI_Async_Migrate", "armci");
  TCHARM_Async_Migrate();
}
CDECL void ARMCI_Checkpoint(char* dname){
  TCHARM_API_TRACE("ARMCI_Checkpoint", "armci");
  ARMCI_Barrier();
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->startCheckpoint(dname);
}
CDECL void ARMCI_MemCheckpoint(void){
  TCHARM_API_TRACE("ARMCI_MemCheckpoint", "armci");
  ARMCI_Barrier();
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  vp->startCheckpoint("");
}

