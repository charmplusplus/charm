/*
 This is the public interface file to the armci library
 The interface is derived from the header file of the same name by the
 original developer(?) Jarek Nieplocha of ARMCI at Pacific Northwest 
 National Laboratory.
 
 This version by Chee Wai Lee, UIUC, 2002
*/

#ifndef _ARMCI_H
#define _ARMCI_H

#include "tcharmc.h" /* for TCHARM_Element */

#ifdef __cplusplus
extern "C" {
#endif

/* user's thread start routine */
void armciStart(int argc, char **argv);

/* redefine global variables used by armci */
#define armci_me TCHARM_Element()
#define armci_master 0
/* #define armci_nproc CpvAccess(_armci_nproc) */
/* #define armci_nproc TCharmNumElements()  /* not good? it crashes sometimes! */
extern int armci_nproc;

/* structures */
typedef struct {
  void **src_ptr_array;
  void **dst_ptr_array;
  int  ptr_array_len;
  int bytes;
} armci_giov_t;

/* virtual processor aggregated shared memory API */

/* basic copy operations */
void ARMCI_Copy(void *src, void *dst, int n);
int ARMCI_Put(void *src, void* dst, int bytes, int proc);
int ARMCI_Put_flag(void *src, void* dst,int bytes,int *f,int v,int proc);
int ARMCI_Get(void *src, void* dst, int bytes, int proc);

/* strided copy operations */
int ARMCI_PutS(void *src_ptr,        /* ptr to 1st segment at source */
		     int src_stride_arr[], /* array of strides at source  */
		     void* dst_ptr,        /* ptr to 1st segment at dest */
		     int dst_stride_arr[], /* array of strides at destination */
		     int count[],          /* number of units at each stride  */
		                           /* level count[0]=bytes  */
		     int stride_levels,    /* number of stride levels */
		     int proc              /* remote process(or) ID  */
		     );
int ARMCI_PutS_flag(void *src_ptr,        /* ptr to 1st segment at source */
			  int src_stride_arr[], /* array of strides at source */
			  void* dst_ptr,        /* ptr to 1st segment at dest */
			  int dst_stride_arr[], /* array of strides at dest */
			  int count[],          /* num segments at each stride */
			                        /* levels: count[0]=bytes */
			  int stride_levels,    /* number of stride levels */
			  int *flag,            /* pointer to remote flag */
			  int val,              /* value to set flag upon  */
			                        /* completion of data transfer */
			  int proc              /* remote process(or) ID */
			  );
int ARMCI_AccS(
		     int  optype,          /* operation */
		     void *scale,          /* scale factor x += scale*y */
		     void *src_ptr,        /* pointer to 1st segment at source */
		     int src_stride_arr[], /* array of strides at source */
		     void* dst_ptr,        /* ptr to 1st segment at destination */
		     int dst_stride_arr[], /* array of strides at destination */
		     int count[],          /* number of units at each stride  */
		                           /* level count[0]=bytes */
		     int stride_levels,    /* number of stride levels */
		     int proc              /* remote process(or) ID */
		     );
int ARMCI_GetS(void *src_ptr,        /* pointer to 1st segment at source */
		     int src_stride_arr[], /* array of strides at source */
		     void* dst_ptr,        /* ptr to 1st segment at destination */
		     int dst_stride_arr[], /* array of strides at destination */
		     int count[],          /* number of units at each stride  */
		                           /* level count[0]=bytes */
		     int stride_levels,    /* number of stride levels */
		     int proc              /* remote process(or) ID */
		     );

/* vector IO-type copy operations */
int ARMCI_GetV(armci_giov_t darr[], /* descriptor array */
		     int len,             /* length of descriptor array */
		     int proc             /* remote process(or) ID */
		     );
int ARMCI_PutV(armci_giov_t darr[], /* descriptor array */
		     int len,             /* length of descriptor array */
		     int proc             /* remote process(or) ID */
		     );
int ARMCI_AccV(int op,              /* operation code */
		     void *scale,         /* scaling factor for accumulate */
		     armci_giov_t darr[], /* descriptor array */
		     int len,             /* length of descriptor array */
		     int proc             /* remote process(or) ID */
		     );

/* fence operations (for synchronizing with put operations) */
int ARMCI_Fence(int proc);
int ARMCI_AllFence(void);

/* memory operations */
int ARMCI_Malloc(void* ptr_arr[], int bytes);
int ARMCI_Free(void *ptr);
 
/* misc operations */
int  ARMCI_Rmw(int op, int *ploc, int *prem, int extra, int proc);

/* system operations */
int ARMCI_Init(void);
int ARMCI_Finalize(void);
void ARMCI_Error(char *msg, int code);
void ARMCI_Cleanup(void);

/* mutex operations */
int ARMCI_Create_mutexes(int num);
int ARMCI_Destroy_mutexes(void);
void ARMCI_Lock(int mutex, int proc);
void ARMCI_Unlock(int mutex, int proc);

/* this is highly platform specific and I don't think it needs to be
   included. */
/* extern void ARMCI_Set_shm_limit(unsigned long shmemlimit); */
/* extern int ARMCI_Uses_shm(); */

#ifdef __cplusplus
};
#endif

#endif
