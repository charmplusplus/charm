/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef CONVERSE_H
#define CONVERSE_H

#ifndef _conv_mach_h
#include "conv-mach.h"
#include "conv-autoconfig.h"
#endif

/**** DEAL WITH DIFFERENCES: KERNIGHAN-RITCHIE-C, ANSI-C, AND C++ ****/

#define CMK_CONCAT(x,y) x##y

#include "pup_c.h"

/* the following flags denote properties of the C compiler,  */
/* not the C++ compiler.  If this is C++, ignore them.       */
#ifdef __cplusplus

/* Only C++ needs this backup bool defined.  We'll assume that C doesn't
   use it */

#if CMK_BOOL_UNDEFINED
enum CmiBool {CmiFalse=0, CmiTrue=1};
#else
typedef bool CmiBool;
#define CmiFalse false
#define CmiTrue true
#endif

extern "C" {
#endif

/******************************************************************************
 *
 * Deal with Shared Memory
 *
 * Shared memory strongly affects how CPV, CSV, and CmiMyPe are defined,
 * and how memory locking is performed. Therefore, we control all these
 * functions with a single flag.
 *
 *****************************************************************************/

#if CMK_SHARED_VARS_UNAVAILABLE /* Non-SMP version of shared vars. */

extern int Cmi_mype;
extern int Cmi_numpes;

#define CmiMyPe()           Cmi_mype
#define CmiMyRank()         0
#define CmiNumPes()         Cmi_numpes
#define CmiMyNodeSize()     1
#define CmiMyNode()         Cmi_mype
#define CmiNumNodes()       Cmi_numpes
#define CmiNodeFirst(node)  (node)
#define CmiNodeSize(node)   1
#define CmiNodeOf(pe)       (pe)
#define CmiRankOf(pe)       0

#define CpvDeclare(t,v) t CMK_CONCAT(Cpv_Var_,v)
#define CpvExtern(t,v)  extern t CMK_CONCAT(Cpv_Var_,v)
#define CpvStaticDeclare(t,v) static t CMK_CONCAT(Cpv_Var_,v)
#define CpvInitialize(t,v) do {} while(0)
#define CpvInitialized(v) 1
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)
#define CpvAccessOther(v, r) CMK_CONCAT(Cpv_Var_,v)

extern void CmiMemLock();
extern void CmiMemUnlock();
#define CmiNodeBarrier() /*empty*/
#define CmiSvAlloc CmiAlloc

typedef void *CmiNodeLock;
#define CmiCreateLock() ((void *)0)
#define CmiLock(lock) /*empty*/
#define CmiUnlock(lock) /*empty*/
#define CmiTryLock(lock) /*empty*/
#define CmiDestroyLock(lock) /*empty*/

#endif

#if CMK_SHARED_VARS_POSIX_THREADS_SMP /*Used by the net-*-smp versions*/

#include <pthread.h>
#include <sched.h>
#ifdef CMK_FAKE_SCHED_YIELD
#include <unistd.h>
#define sched_yield() sleep(0)
#endif

extern int Cmi_numpes;
extern int Cmi_mynodesize;
extern int Cmi_mynode;
extern int Cmi_numnodes;

extern int CmiMyPe();
extern int CmiMyRank();
#define CmiNumPes()         Cmi_numpes
#define CmiMyNodeSize()     Cmi_mynodesize
#define CmiMyNode()         Cmi_mynode
#define CmiNumNodes()       Cmi_numnodes
extern int CmiNodeFirst(int node);
extern int CmiNodeSize(int node);
extern int CmiNodeOf(int pe);
extern int CmiRankOf(int pe);

#define CMK_CPV_IS_SMP sched_yield();

extern void CmiNodeBarrier(void);
#define CmiSvAlloc CmiAlloc

typedef pthread_mutex_t *CmiNodeLock;
extern CmiNodeLock CmiCreateLock();
#define CmiLock(lock) (pthread_mutex_lock(lock))
#define CmiUnlock(lock) (pthread_mutex_unlock(lock))
#define CmiTryLock(lock) (pthread_mutex_trylock(lock))
extern void CmiDestroyLock(CmiNodeLock lock);

extern CmiNodeLock CmiMemLock_lock;
#define CmiMemLock() do{if (CmiMemLock_lock) CmiLock(CmiMemLock_lock);} while (0)
#define CmiMemUnlock() do{if (CmiMemLock_lock) CmiUnlock(CmiMemLock_lock);} while (0)

#endif


#if CMK_SHARED_VARS_EXEMPLAR /* Used only by HP Exemplar version */

#include <spp_prog_model.h>
#include <cps.h>

extern int Cmi_numpes;
extern int Cmi_mynodesize;

#define CmiMyPe()           (my_thread())
#define CmiMyRank()         (my_thread())
#define CmiNumPes()         Cmi_numpes
#define CmiMyNodeSize()     Cmi_numpes
#define CmiMyNode()         0
#define CmiNumNodes()       1
#define CmiNodeFirst(node)  0
#define CmiNodeSize(node)   Cmi_numpes
#define CmiNodeOf(pe)       0
#define CmiRankOf(pe)       (pe)

#define CMK_CPV_IS_SMP {} 

extern void CmiMemLock();
extern void CmiMemUnlock();
extern void CmiNodeBarrier(void);
extern void *CmiSvAlloc(int);

typedef cps_mutex_t *CmiNodeLock;
extern CmiNodeLock CmiCreateLock(void);
#define CmiLock(lock) (cps_mutex_lock(lock))
#define CmiUnlock(lock) (cps_mutex_unlock(lock))
#define CmiTryLock(lock) (cps_mutex_trylock(lock))
#define CmiDestroyLock(lock) (cps_mutex_free(lock))

#endif

#if CMK_SHARED_VARS_UNIPROCESSOR /*Used only by uth- and sim- versions*/

extern int Cmi_mype;
extern int Cmi_numpes;

#define CmiMyPe()              Cmi_mype
#define CmiMyRank()            Cmi_mype
#define CmiNumPes()            Cmi_numpes
#define CmiMyNodeSize()        Cmi_numpes
#define CmiMyNode()            0
#define CmiNumNodes()          1
#define CmiNodeFirst(node)     0
#define CmiNodeSize(node)      Cmi_numpes
#define CmiNodeOf(pe)          0
#define CmiRankOf(pe)          (pe)

#define CpvDeclare(t,v) t* CMK_CONCAT(Cpv_Var_,v)
#define CpvExtern(t,v)  extern t* CMK_CONCAT(Cpv_Var_,v)
#define CpvStaticDeclare(t,v) static t* CMK_CONCAT(Cpv_Var_,v)
#define CpvInitialize(t,v)\
  do  { if (CMK_CONCAT(Cpv_Var_,v)==0)\
        { CMK_CONCAT(Cpv_Var_,v) = (t *)CmiAlloc(CmiNumPes()*sizeof(t)); }}\
  while(0)
#define CpvInitialized(v) (0!=CMK_CONCAT(Cpv_Var_,v))
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)[CmiMyPe()]
#define CpvAccessOther(v, r) CMK_CONCAT(Cpv_Var_,v)[r]

#define CmiMemLock() 0
#define CmiMemUnlock() 0
extern void CmiNodeBarrier();
#define CmiSvAlloc CmiAlloc

typedef int *CmiNodeLock;
extern CmiNodeLock  CmiCreateLock(void);
extern void         CmiLock(CmiNodeLock lock);
extern void         CmiUnlock(CmiNodeLock lock);
extern int          CmiTryLock(CmiNodeLock lock);
extern void         CmiDestroyLock(CmiNodeLock lock);

#endif

#if CMK_SHARED_VARS_PTHREADS /*Used only by origin-pthreads*/

#include <pthread.h>
#include <sched.h>

extern int CmiMyPe();
extern int Cmi_numpes;

#define CmiNumPes()            Cmi_numpes
#define CmiMyRank()            CmiMyPe()
#define CmiMyNodeSize()        Cmi_numpes
#define CmiMyNode()            0
#define CmiNumNodes()          1
#define CmiNodeFirst(node)     0
#define CmiNodeSize(node)      Cmi_numpes
#define CmiNodeOf(pe)          0
#define CmiRankOf(pe)          (pe)

#define CMK_CPV_IS_SMP sched_yield();

extern void CmiMemLock();
extern void CmiMemUnlock();
extern void CmiNodeBarrier();
#define CmiSvAlloc CmiAlloc

typedef pthread_mutex_t *CmiNodeLock;
extern CmiNodeLock  CmiCreateLock(void);
extern void         CmiLock(CmiNodeLock lock);
extern void         CmiUnlock(CmiNodeLock lock);
extern int          CmiTryLock(CmiNodeLock lock);
extern void         CmiDestroyLock(CmiNodeLock lock);

#endif

#if CMK_SHARED_VARS_NT_THREADS /*Used only by win32 versions*/

#include <windows.h>

extern int Cmi_numpes;
extern int Cmi_mynodesize;
extern int Cmi_mynode;
extern int Cmi_numnodes;

extern int CmiMyPe();
extern int CmiMyRank();
#define CmiNumPes()         Cmi_numpes
#define CmiMyNodeSize()     Cmi_mynodesize
#define CmiMyNode()         Cmi_mynode
#define CmiNumNodes()       Cmi_numnodes
extern int CmiNodeFirst(int node);
extern int CmiNodeSize(int node);
extern int CmiNodeOf(int pe);
extern int CmiRankOf(int pe);

#define CMK_CPV_IS_SMP Sleep(0);

extern void CmiNodeBarrier(void);
#define CmiSvAlloc CmiAlloc

typedef HANDLE CmiNodeLock;
extern  CmiNodeLock CmiCreateLock(void);
#define CmiLock(lock) (WaitForSingleObject(lock, INFINITE))
#define CmiUnlock(lock) (ReleaseMutex(lock))
#define CmiTryLock(lock) (WaitForSingleObject(lock, 0))
extern  void CmiDestroyLock(CmiNodeLock lock);

extern CmiNodeLock CmiMemLock_lock;
#define CmiMemLock() do{if (CmiMemLock_lock) CmiLock(CmiMemLock_lock);} while (0)
#define CmiMemUnlock() do{if (CmiMemLock_lock) CmiUnlock(CmiMemLock_lock);} while (0)

#endif

/* This is the default Cpv implmentation for SMP-style systems:
A Cpv variable is actually a pointer to an array of values, one
for each processor in the node.
*/
#ifdef CMK_CPV_IS_SMP

#define CpvDeclare(t,v) t* CMK_CONCAT(Cpv_Var_,v)
#define CpvExtern(t,v)  extern t* CMK_CONCAT(Cpv_Var_,v)
#define CpvStaticDeclare(t,v) static t* CMK_CONCAT(Cpv_Var_,v)
#define CpvInitialize(t,v)\
    do { \
       if (CmiMyRank()) { \
		while (!CpvInitialized(v)) CMK_CPV_IS_SMP \
       } else { \
	       CMK_CONCAT(Cpv_Var_,v)=(t*)calloc((1+CmiMyNodeSize()),sizeof(t));\
       } \
    } while(0)
#define CpvInitialized(v) (0!=CMK_CONCAT(Cpv_Var_,v))
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)[CmiMyRank()]
#define CpvAccessOther(v, r) CMK_CONCAT(Cpv_Var_,v)[r]

#endif

/*Csv are the same almost everywhere:*/
#ifndef CsvDeclare
#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvExtern(t,v) extern t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v) do{}while(0)
#define CsvAccess(v) CMK_CONCAT(Csv_Var_,v)
#endif



/******** CMI: TYPE DEFINITIONS ********/

typedef CMK_TYPEDEF_INT2      CmiInt2;
typedef CMK_TYPEDEF_INT4      CmiInt4;
typedef CMK_TYPEDEF_INT8      CmiInt8;
typedef CMK_TYPEDEF_UINT2     CmiUInt2;
typedef CMK_TYPEDEF_UINT4     CmiUInt4;
typedef CMK_TYPEDEF_UINT8     CmiUInt8;
typedef CMK_TYPEDEF_FLOAT4    CmiFloat4;
typedef CMK_TYPEDEF_FLOAT8    CmiFloat8;

typedef void  *CmiCommHandle;
typedef void (*CmiHandler)();

typedef struct CMK_MSG_HEADER_BASIC CmiMsgHeaderBasic;
typedef struct CMK_MSG_HEADER_EXT   CmiMsgHeaderExt;

#define CmiMsgHeaderSizeBytes (sizeof(CmiMsgHeaderBasic))
#define CmiExtHeaderSizeBytes (sizeof(CmiMsgHeaderExt))

/******** CMI, CSD: MANY LOW-LEVEL OPERATIONS ********/

CpvExtern(CmiHandler*, CmiHandlerTable);
CpvExtern(int,         CmiHandlerMax);
CpvExtern(void*,       CsdSchedQueue);
#if CMK_NODE_QUEUE_AVAILABLE
CsvExtern(void*,       CsdNodeQueue);
CsvExtern(CmiNodeLock, CsdNodeQueueLock);
#endif
CpvExtern(int,         CsdStopFlag);

extern int CmiRegisterHandler(CmiHandler);
extern int CmiRegisterHandlerLocal(CmiHandler);
extern int CmiRegisterHandlerGlobal(CmiHandler);
extern void CmiNumberHandler(int, CmiHandler);

#define CmiGetHandler(m)  (((CmiMsgHeaderExt*)m)->hdl)
#define CmiGetXHandler(m) (((CmiMsgHeaderExt*)m)->xhdl)
#define CmiGetInfo(m)     (((CmiMsgHeaderExt*)m)->info)

#define CmiSetHandler(m,v)  do {((((CmiMsgHeaderExt*)m)->hdl)=(v));} while(0)
#define CmiSetXHandler(m,v) do {((((CmiMsgHeaderExt*)m)->xhdl)=(v));} while(0)
#define CmiSetInfo(m,v)     do {((((CmiMsgHeaderExt*)m)->info)=(v));} while(0)

#define CmiHandlerToFunction(n) (CpvAccess(CmiHandlerTable)[n])
#define CmiGetHandlerFunction(env) (CmiHandlerToFunction(CmiGetHandler(env)))

void    *CmiAlloc(int size);
int      CmiSize(void *);
void     CmiFree(void *);

double   CmiCpuTimer(void);

#if CMK_TIMER_USE_RDTSC

extern double cpu_speed_factor;

static __inline__ unsigned long long int rdtsc(void)
{
        unsigned long long int x;
#ifdef CMK_IA64
	__asm__ __volatile__("mov %0=ar.itc" : "=r"(x) :: "memory");
#else
        __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
#endif
        return x;
}

#define CmiWallTimer() ((double)rdtsc()*(cpu_speed_factor))
#define CmiTimer CmiCpuTimer

#else
double   CmiTimer(void);
double   CmiWallTimer(void);
#endif

#if CMK_NODE_QUEUE_AVAILABLE

#define CsdNodeEnqueueGeneral(x,s,i,p) do { \
          CmiLock(CsvAccess(CsdNodeQueueLock));\
          CqsEnqueueGeneral(CsvAccess(CsdNodeQueue),(x),(s),(i),(p)); \
          CmiUnlock(CsvAccess(CsdNodeQueueLock)); \
        } while(0)
#define CsdNodeEnqueueFifo(x)     do { \
          CmiLock(CsvAccess(CsdNodeQueueLock));\
          CqsEnqueueFifo(CsvAccess(CsdNodeQueue),(x)); \
          CmiUnlock(CsvAccess(CsdNodeQueueLock)); \
        } while(0)
#define CsdNodeEnqueueLifo(x)     do { \
          CmiLock(CsvAccess(CsdNodeQueueLock));\
          CqsEnqueueLifo(CsvAccess(CsdNodeQueue),(x))); \
          CmiUnlock(CsvAccess(CsdNodeQueueLock)); \
        } while(0)
#define CsdNodeEnqueue(x)     do { \
          CmiLock(CsvAccess(CsdNodeQueueLock));\
          CqsEnqueueFifo(CsvAccess(CsdNodeQueue),(x));\
          CmiUnlock(CsvAccess(CsdNodeQueueLock)); \
        } while(0)

#define CsdNodeEmpty()            (CqsEmpty(CpvAccess(CsdNodeQueue)))
#define CsdNodeLength()           (CqsLength(CpvAccess(CsdNodeQueue)))

#else

#define CsdNodeEnqueueGeneral(x,s,i,p) (CsdEnqueueGeneral(x,s,i,p))
#define CsdNodeEnqueueFifo(x) (CqsEnqueueFifo(CpvAccess(CsdSchedQueue),(x)))
#define CsdNodeEnqueueLifo(x) (CqsEnqueueLifo(CpvAccess(CsdSchedQueue),(x)))
#define CsdNodeEnqueue(x)     (CsdEnqueue(x))
#define CsdNodeEmpty()        (CqsEmpty(CpvAccess(CsdSchedQueue)))
#define CsdNodeLength()       (CqsLength(CpvAccess(CsdSchedQueue)))

#endif

#define CsdEnqueueGeneral(x,s,i,p)\
    (CqsEnqueueGeneral(CpvAccess(CsdSchedQueue),(x),(s),(i),(p)))
#define CsdEnqueueFifo(x)     (CqsEnqueueFifo(CpvAccess(CsdSchedQueue),(x)))
#define CsdEnqueueLifo(x)     (CqsEnqueueLifo(CpvAccess(CsdSchedQueue),(x)))
#define CsdEnqueue(x)         (CqsEnqueueFifo(CpvAccess(CsdSchedQueue),(x)))
#define CsdEmpty()            (CqsEmpty(CpvAccess(CsdSchedQueue)))
#define CsdLength()           (CqsLength(CpvAccess(CsdSchedQueue)))

#if CMK_CMIPRINTF_IS_A_BUILTIN
void  CmiPrintf(const char *, ...);
void  CmiError(const char *, ...);
int   CmiScanf(const char *, ...);
#endif

#if CMK_CMIPRINTF_IS_JUST_PRINTF
#include <stdio.h>

/*
 * I made vprintf functions for CmiPrintf and CmiError, but on the
 * O2K, there is no equivalent vscanf!

 #define CmiPrintf printf
 #define CmiError  printf
*/
#include <stdarg.h>

void  CmiPrintf(const char *format, ...);
void  CmiError(const char *format, ...);
#define CmiScanf  scanf

#endif

#if CMK_OPTIMIZE
#define CmiAssert(expr) ((void) 0)
#else
#if defined(__STDC__) || defined(__cplusplus)
#define __CMK_STRING(x) #x
#else
#define __CMK_STRING(x) "x"
#endif
extern void __cmi_assert(const char *, const char *, int);
#define CmiAssert(expr) \
  ((void) ((expr) ? 0 :                   \
     (__cmi_assert (__CMK_STRING(expr), __FILE__, __LINE__), 0)))
#endif

typedef void (*CmiStartFn)(int argc, char **argv);

/********* CSD - THE SCHEDULER ********/

CpvExtern(int, _ccd_numchecks);
extern void  CcdCallBacks();
#define CsdPeriodic() do{ if (CpvAccess(_ccd_numchecks)-- <= 0) CcdCallBacks(); } while(0)

extern void  CsdEndIdle(void);
extern void  CsdStillIdle(void);
extern void  CsdBeginIdle(void);

typedef struct {
  void *localQ;
  void *nodeQ;
  void *schedQ;
  CmiNodeLock nodeLock;
} CsdSchedulerState_t;
extern void CsdSchedulerState_new(CsdSchedulerState_t *state);
extern void *CsdNextMessage(CsdSchedulerState_t *state);

extern void  *CmiGetNonLocal(void);
extern void   CmiNotifyIdle(void);

/*Different kinds of schedulers: generic, eternal, counting, polling*/
extern  int CsdScheduler(int maxmsgs);
extern void CsdScheduleForever(void);
extern  int CsdScheduleCount(int maxmsgs);
extern void CsdSchedulePoll(void);

#define CsdExitScheduler()  (CpvAccess(CsdStopFlag)++)

#if CMK_SPANTREE_USE_COMMON_CODE

#define SPANTREE_W  (CMK_SPANTREE_MAXSPAN)
#define NN (CmiNumNodes())
#define CmiNodeSpanTreeParent(n) ((n)?(((n)-1)/SPANTREE_W):(-1))
#define CmiNodeSpanTreeChildren(n,c) do {\
          int _i; \
          for(_i=0; _i<SPANTREE_W; _i++) { \
            int _x = (n)*SPANTREE_W+_i+1; \
            if(_x<NN) (c)[_i]=_x; \
          }\
        } while(0)
#define CmiNumNodeSpanTreeChildren(n) ((((n)+1)*SPANTREE_W<NN)? SPANTREE_W : \
          ((((n)*SPANTREE_W+1)>=NN)?0:((NN-1)-(n)*SPANTREE_W)))
#define R(p) (CmiRankOf(p))
#define NF(n) (CmiNodeFirst(n))
#define SP(n) (CmiNodeSpanTreeParent(n))
#define ND(p) (CmiNodeOf(p))
#define NS(p) (CmiNodeSize(ND(p)))
#define CmiSpanTreeParent(p) ((p)?(R(p)?(NF(ND(p))+R(p)/SPANTREE_W):NF(SP(ND(p)))):(-1))
#define C(p) (((R(p)+1)*SPANTREE_W<NS(p))?SPANTREE_W:(((R(p)*SPANTREE_W+1)>=NS(p))?0:((NS(p)-1)-R(p)*SPANTREE_W)))
#define SC(p) (CmiNumNodeSpanTreeChildren(ND(p)))
#define CmiNumSpanTreeChildren(p) (R(p)?C(p):(SC(p)+C(p)))
#define CmiSpanTreeChildren(p,c) do {\
          int _i,_c=0; \
          if(R(p)==0) { \
            for(_i=0;_i<SPANTREE_W;_i++) { \
              int _x = ND(p)*SPANTREE_W+_i+1; \
              if(_x<NN) (c)[_c++]=NF(_x); \
            }\
          } \
          for(_i=0;_i<SPANTREE_W;_i++) { \
            int _x = R(p)*SPANTREE_W+_i+1; \
            if(_x<NS(p)) (c)[_c++]=NF(ND(p))+_x; \
          }\
        } while(0)
#endif

#if CMK_SPANTREE_USE_SPECIAL_CODE
int      CmiSpanTreeNumChildren(int) ;
int      CmiSpanTreeParent(int) ;
void     CmiSpanTreeChildren(int node, int *children);
int      CmiNodeSpanTreeNumChildren(int);
int      CmiNodeSpanTreeParent(int) ;
void     CmiNodeSpanTreeChildren(int node, int *children) ;
#endif

/****** MULTICAST GROUPS ******/

typedef CMK_MULTICAST_GROUP_TYPE CmiGroup;

CmiGroup CmiEstablishGroup(int npes, int *pes);
void     CmiLookupGroup(CmiGroup grp, int *npes, int **pes);

/****** CMI MESSAGE TRANSMISSION ******/

void          CmiSyncSendFn(int, int, char *);
CmiCommHandle CmiAsyncSendFn(int, int, char *);
void          CmiFreeSendFn(int, int, char *);

void          CmiSyncBroadcastFn(int, char *);
CmiCommHandle CmiAsyncBroadcastFn(int, char *);
void          CmiFreeBroadcastFn(int, char *);

void          CmiSyncBroadcastAllFn(int, char *);
CmiCommHandle CmiAsyncBroadcastAllFn(int, char *);
void          CmiFreeBroadcastAllFn(int, char *);

void          CmiSyncListSendFn(int, int *, int, char*);
CmiCommHandle CmiAsyncListSendFn(int, int *, int, char*);
void          CmiFreeListSendFn(int, int *, int, char*);

void          CmiSyncMulticastFn(CmiGroup, int, char*);
CmiCommHandle CmiAsyncMulticastFn(CmiGroup, int, char*);
void          CmiFreeMulticastFn(CmiGroup, int, char*);

void          CmiSyncVectorSend(int, int, int *, char **);
CmiCommHandle CmiAsyncVectorSend(int, int, int *, char **);
void          CmiSyncVectorSendAndFree(int, int, int *, char **);
void	      CmiMultipleSend(unsigned int, int, int *, char **);

#define CmiSyncSend(p,s,m)              (CmiSyncSendFn((p),(s),(char *)(m)))
#define CmiAsyncSend(p,s,m)             (CmiAsyncSendFn((p),(s),(char *)(m)))
#define CmiSyncSendAndFree(p,s,m)       (CmiFreeSendFn((p),(s),(char *)(m)))

#define CmiSyncBroadcast(s,m)           (CmiSyncBroadcastFn((s),(char *)(m)))
#define CmiAsyncBroadcast(s,m)          (CmiAsyncBroadcastFn((s),(char *)(m)))
#define CmiSyncBroadcastAndFree(s,m)    (CmiFreeBroadcastFn((s),(char *)(m)))

#define CmiSyncBroadcastAll(s,m)        (CmiSyncBroadcastAllFn((s),(char *)(m)))
#define CmiAsyncBroadcastAll(s,m)       (CmiAsyncBroadcastAllFn((s),(char *)(m)))
#define CmiSyncBroadcastAllAndFree(s,m) (CmiFreeBroadcastAllFn((s),(char *)(m)))

#define CmiSyncListSend(n,l,s,m)        (CmiSyncListSendFn((n),(l),(s),(char *)(m)))
#define CmiAsyncListSend(n,l,s,m)       (CmiAsyncListSendFn((n),(l),(s),(char *)(m)))
#define CmiSyncListSendAndFree(n,l,s,m) (CmiFreeListSendFn((n),(l),(s),(char *)(m)))

#define CmiSyncMulticast(g,s,m)         (CmiSyncMulticastFn((g),(s),(char*)(m)))
#define CmiAsyncMulticast(g,s,m)        (CmiAsyncMulticastFn((g),(s),(char*)(m)))
#define CmiSyncMulticastAndFree(g,s,m)  (CmiFreeMulticastFn((g),(s),(char*)(m)))

#if CMK_NODE_QUEUE_AVAILABLE
void          CmiSyncNodeSendFn(int, int, char *);
CmiCommHandle CmiAsyncNodeSendFn(int, int, char *);
void          CmiFreeNodeSendFn(int, int, char *);

void          CmiSyncNodeBroadcastFn(int, char *);
CmiCommHandle CmiAsyncNodeBroadcastFn(int, char *);
void          CmiFreeNodeBroadcastFn(int, char *);

void          CmiSyncNodeBroadcastAllFn(int, char *);
CmiCommHandle CmiAsyncNodeBroadcastAllFn(int, char *);
void          CmiFreeNodeBroadcastAllFn(int, char *);
#endif

#if CMK_NODE_QUEUE_AVAILABLE
#define CmiSyncNodeSend(p,s,m)          (CmiSyncNodeSendFn((p),(s),(char *)(m)))
#define CmiAsyncNodeSend(p,s,m)             (CmiAsyncNodeSendFn((p),(s),(char *)(m)))
#define CmiSyncNodeSendAndFree(p,s,m)       (CmiFreeNodeSendFn((p),(s),(char *)(m)))
#define CmiSyncNodeBroadcast(s,m)           (CmiSyncNodeBroadcastFn((s),(char *)(m)))
#define CmiAsyncNodeBroadcast(s,m)          (CmiAsyncNodeBroadcastFn((s),(char *)(m)))
#define CmiSyncNodeBroadcastAndFree(s,m)    (CmiFreeNodeBroadcastFn((s),(char *)(m)))
#define CmiSyncNodeBroadcastAll(s,m)        (CmiSyncNodeBroadcastAllFn((s),(char *)(m)))
#define CmiAsyncNodeBroadcastAll(s,m)       (CmiAsyncNodeBroadcastAllFn((s),(char *)(m)))
#define CmiSyncNodeBroadcastAllAndFree(s,m) (CmiFreeNodeBroadcastAllFn((s),(char *)(m)))
#else
#define CmiSyncNodeSend(n,s,m)        CmiSyncSend(CmiNodeFirst(n),s,m)
#define CmiAsyncNodeSend(n,s,m)       CmiAsyncSend(CmiNodeFirst(n),s,m)
#define CmiSyncNodeSendAndFree(n,s,m) CmiSyncSendAndFree(CmiNodeFirst(n),s,m)
#define CmiSyncNodeBroadcast(s,m)           do { \
          int _i; \
          for(_i=0; _i<CmiNumNodes(); _i++) \
            if(_i != CmiMyNode()) \
              CmiSyncSend(CmiNodeFirst(_i),s,m); \
        } while(0)
#define CmiAsyncNodeBroadcast(s,m)          CmiSyncNodeBroadcast(s,m)
#define CmiSyncNodeBroadcastAndFree(s,m)    do { \
          CmiSyncNodeBroadcast(s,m); \
          CmiFree(m); \
        } while(0)
#define CmiSyncNodeBroadcastAll(s,m)           do { \
          int _i; \
          for(_i=0; _i<CmiNumNodes(); _i++) \
            CmiSyncSend(CmiNodeFirst(_i),s,m); \
        } while(0)
#define CmiAsyncNodeBroadcastAll(s,m)       CmiSyncNodeBroadcastAll(s,m)
#define CmiSyncNodeBroadcastAllAndFree(s,m) do { \
          CmiSyncNodeBroadcastAll(s,m); \
          CmiFree(m); \
        } while(0)
#endif

/******** CMI MESSAGE RECEPTION ********/

int    CmiDeliverMsgs(int maxmsgs);
void   CmiDeliverSpecificMsg(int handler);
void   CmiHandleMessage(void *msg);

/******** CQS: THE QUEUEING SYSTEM ********/

#define CQS_QUEUEING_FIFO 2
#define CQS_QUEUEING_LIFO 3
#define CQS_QUEUEING_IFIFO 4
#define CQS_QUEUEING_ILIFO 5
#define CQS_QUEUEING_BFIFO 6
#define CQS_QUEUEING_BLIFO 7

/****** Isomalloc Memory Allocation ********/
struct CmiIsomallocBlock {
	int slot; /*First mapped slot*/
	int nslots; /*Number of mapped slots*/
};
typedef struct CmiIsomallocBlock CmiIsomallocBlock;

void *CmiIsomalloc(int size,CmiIsomallocBlock *b);
void *CmiIsomallocPup(pup_er p,CmiIsomallocBlock *b);
void  CmiIsomallocFree(CmiIsomallocBlock *b);
int   CmiIsomallocInRange(void *addr);

/****** CTH: THE LOW-LEVEL THREADS PACKAGE ******/

typedef struct CthThreadStruct *CthThread;

typedef void        (*CthVoidFn)();
typedef void        (*CthAwkFn)(CthThread,int,
				int prioBits,unsigned int *prioptr);
typedef CthThread   (*CthThFn)();

int        CthImplemented(void);

CthThread  CthPup(pup_er, CthThread);

CthThread  CthSelf(void);
CthThread  CthCreate(CthVoidFn, void *, int);
CthThread  CthCreateMigratable(CthVoidFn, void *, int);
void       CthResume(CthThread);
void       CthFree(CthThread);

void       CthSetSuspendable(CthThread, int);
int        CthIsSuspendable(CthThread);

void       CthSuspend(void);
void       CthAwaken(CthThread);
void       CthAwakenPrio(CthThread, int, int, unsigned int *);
void       CthSetStrategy(CthThread, CthAwkFn, CthThFn);
void       CthSetStrategyDefault(CthThread);
void       CthYield(void);
void       CthYieldPrio(int,int,unsigned int*);

void       CthSetNext(CthThread t, CthThread next);
CthThread  CthGetNext(CthThread t);

void       CthAutoYield(CthThread t, int flag);
double     CthAutoYieldFreq(CthThread t);
void       CthAutoYieldBlock(void);
void       CthAutoYieldUnblock(void);

void       CthSwitchThread(CthThread t);

/****** CTH: THREAD-PRIVATE VARIABLES ******/

#if CMK_THREADS_REQUIRE_NO_CPV

#define CthCpvDeclare(t,v)    t v
#define CthCpvExtern(t,v)     extern t v
#define CthCpvStatic(t,v)     static t v
#define CthCpvInitialize(t,v) do {} while(0)
#define CthCpvAccess(x)       x

#else

#define CthCpvDeclare(t,v)    CpvDeclare(t,v)
#define CthCpvExtern(t,v)     CpvExtern(t,v)
#define CthCpvStatic(t,v)     CpvStaticDeclare(t,v)
#define CthCpvInitialize(t,v) CpvInitialize(t,v)
#define CthCpvAccess(x)       CpvAccess(x)

#endif

CthCpvExtern(char *,CthData);
extern int CthRegister(int);
extern char *CthGetData(CthThread t);

#define CtvDeclare(t,v)         typedef t CtvType##v; CsvDeclare(int,CtvOffs##v)=(-1)
#define CtvStaticDeclare(t,v)   typedef t CtvType##v; CsvStaticDeclare(int,CtvOffs##v)=(-1)
#define CtvExtern(t,v)          typedef t CtvType##v; CsvExtern(int,CtvOffs##v)
#define CtvAccess(v)            (*((CtvType##v *)(CthCpvAccess(CthData)+CsvAccess(CtvOffs##v))))
#define CtvAccessOther(t,v)            (*((CtvType##v *)(CthGetData(t)+CsvAccess(CtvOffs##v))))
#define CtvInitialize(t,v)      do { \
	if(CsvAccess(CtvOffs##v)==(-1)) \
		CsvAccess(CtvOffs##v)=CthRegister(sizeof(CtvType##v));\
	else CthRegister(sizeof(CtvType##v));\
} while(0)

/************************************************************************
 *
 * CpmDestination
 *
 * A CpmDestination structure enables the user of the Cpm module to tell
 * the parameter-marshalling system what kind of envelope to put int the
 * message, and what to do with it after it has been filled.
 *
 ***********************************************************************/

typedef struct CpmDestinationStruct *CpmDestination;

typedef void *(*CpmSender)(CpmDestination, int, void *);

struct CpmDestinationStruct
{
  CpmSender sendfn;
  int envsize;
};

#define CpmPE(n) n
#define CpmALL (-1)
#define CpmOTHERS (-2)

CpmDestination CpmSend(int pe);
CpmDestination CpmMakeThread(int pe);
CpmDestination CpmMakeThreadSize(int pe, int size);
CpmDestination CpmEnqueueFIFO(int pe);
CpmDestination CpmEnqueueLIFO(int pe);
CpmDestination CpmEnqueueIFIFO(int pe, int prio);
CpmDestination CpmEnqueueILIFO(int pe, int prio);
CpmDestination CpmEnqueueBFIFO(int pe, int priobits, unsigned int *prioptr);
CpmDestination CpmEnqueueBLIFO(int pe, int priobits, unsigned int *prioptr);
CpmDestination CpmEnqueue(int pe,int qs,int priobits,unsigned int *prioptr);

/***********************************************************************
 *
 * CPM macros
 *
 *      CpmInvokable
 *      CpmDeclareSimple(x)
 *      CpmDeclarePointer(x)
 *
 * These macros expand into CPM ``declarations''.  The CPM ``declarations''
 * are actually C code that has no effect, but when the CPM scanner sees
 * them, it recognizes them and understands them as declarations.
 *
 **********************************************************************/

typedef void CpmInvokable;
typedef int CpmDeclareSimple1;
typedef int CpmDeclarePointer1;
#define CpmDeclareSimple(c) typedef CpmDeclareSimple1 CpmType_##c
#define CpmDeclarePointer(c) typedef CpmDeclarePointer1 CpmType_##c

/***********************************************************************
 *
 * Accessing a CPM message:
 *
 ***********************************************************************/

struct CpmHeader
{
  char convcore[CmiMsgHeaderSizeBytes];
  int envpos;
};
#define CpmEnv(msg) (((char *)msg)+(((struct CpmHeader *)msg)->envpos))
#define CpmAlign(val, type) ((val+sizeof(type)-1)&(~(sizeof(type)-1)))

/***********************************************************************
 *
 * Built-in CPM types
 *
 **********************************************************************/

CpmDeclareSimple(char);
#define CpmPack_char(v) do{}while(0)
#define CpmUnpack_char(v) do{}while(0)

CpmDeclareSimple(short);
#define CpmPack_short(v) do{}while(0)
#define CpmUnpack_short(v) do{}while(0)

CpmDeclareSimple(int);
#define CpmPack_int(v) do{}while(0)
#define CpmUnpack_int(v) do{}while(0)

CpmDeclareSimple(long);
#define CpmPack_long(v) do{}while(0)
#define CpmUnpack_long(v) do{}while(0)

CpmDeclareSimple(float);
#define CpmPack_float(v) do{}while(0)
#define CpmUnpack_float(v) do{}while(0)

CpmDeclareSimple(double);
#define CpmPack_double(v) do{}while(0)
#define CpmUnpack_double(v) do{}while(0)

typedef int CpmDim;
CpmDeclareSimple(CpmDim);
#define CpmPack_CpmDim(v) do{}while(0)
#define CpmUnpack_CpmDim(v) do{}while(0)

CpmDeclareSimple(Cfuture);
#define CpmPack_Cfuture(v) do{}while(0)
#define CpmUnpack_Cfuture(v) do{}while(0)

typedef char *CpmStr;
CpmDeclarePointer(CpmStr);
#define CpmPtrSize_CpmStr(v) (strlen(v)+1)
#define CpmPtrPack_CpmStr(p, v) (strcpy(p, v))
#define CpmPtrUnpack_CpmStr(v) do{}while(0)
#define CpmPtrFree_CpmStr(v) do{}while(0)

/****** CFUTURE: CONVERSE FUTURES ******/

typedef struct Cfuture_s
{
  int pe;
  struct Cfuture_data_s *data;
}
Cfuture;

#define CfutureValueData(v) ((void*)((v)->rest))

Cfuture       CfutureCreate(void);
void          CfutureSet(Cfuture f, void *val, int len);
void         *CfutureWait(Cfuture f);
void          CfutureDestroy(Cfuture f);

void         *CfutureCreateBuffer(int bytes);
void          CfutureDestroyBuffer(void *val);
void          CfutureStoreBuffer(Cfuture f, void *value);

#define       CfuturePE(f) ((f).pe)

void CfutureInit();

/****** CLD: THE LOAD BALANCER ******/

#define CLD_ANYWHERE (-1)
#define CLD_BROADCAST (-2)
#define CLD_BROADCAST_ALL (-3)

typedef void (*CldPackFn)(void *msg);

typedef void (*CldInfoFn)(void *msg, 
                          CldPackFn *packer,
                          int *len,
                          int *queueing,
                          int *priobits, 
                          unsigned int **prioptr);

typedef int (*CldEstimator)(void);

int CldRegisterInfoFn(CldInfoFn fn);
int CldRegisterPackFn(CldPackFn fn);
void CldRegisterEstimator(CldEstimator fn);
int CldEstimate(void);
char *CldGetStrategy(void);

void CldEnqueue(int pe, void *msg, int infofn);
void CldEnqueueMulti(int npes, int *pes, void *msg, int infofn);
void CldNodeEnqueue(int node, void *msg, int infofn);

/****** CMM: THE MESSAGE MANAGER ******/

typedef struct CmmTableStruct *CmmTable;

#define CmmWildCard (-1)

CmmTable CmmPup(pup_er, CmmTable);

CmmTable   CmmNew();
void       CmmFree(CmmTable t);
void       CmmPut(CmmTable t, int ntags, int *tags, void *msg);
void      *CmmFind(CmmTable t, int ntags, int *tags, int *returntags, int del);
int        CmmEntries(CmmTable t);
#define    CmmGet(t,nt,tg,rt)   (CmmFind((t),(nt),(tg),(rt),1))
#define    CmmProbe(t,nt,tg,rt) (CmmFind((t),(nt),(tg),(rt),0))

/******** ConverseInit and ConverseExit ********/

void ConverseInit(int, char**, CmiStartFn, int, int);
void ConverseExit(void);

void CmiAbort(const char *);

#if CMK_MEMCHECK_OFF
#define _MEMCHECK(p) do{}while(0)
#else
#define _MEMCHECK(p) do { \
                         if ((p)==0) CmiAbort("Memory Allocation Failure.\n");\
                     } while(0)
#endif

/*********** CPATH ***********/

typedef struct
{
  int    seqno;
  short  creator;
  short  startfn;
  short  mapfn;
  short  nsizes;
  int    sizes[13];
}
CPath;

#define CPathArrayDimensions(a) ((a)->nsizes)
#define CPathArrayDimension(a,n) ((a)->sizes[n])

#define CPATH_WILD (-1)

typedef unsigned int (*CPathMapFn)(CPath *path, int *indices);
typedef void (*CPathReduceFn)(int nelts,void *updateme,void *inputme);

#define CPathRegisterMapper(x)   CmiRegisterHandler((CmiHandler)(x))
#define CPathRegisterThreadFn(x) CmiRegisterHandler((CmiHandler)(x))
#define CPathRegisterReducer(x)  CmiRegisterHandler((CmiHandler)(x))

void CPathMakeArray(CPath *path, int startfn, int mapfn, ...);
void CPathMakeThread(CPath *path, int startfn, int pe);

void  CPathSend(int key, ...);
void *CPathRecv(int key, ...);
void  CPathReduce(int key, ...);

void CPathMsgDecodeBytes(void *msg, int *len, void *bytes);
void CPathMsgDecodeReduction(void *msg,int *vecsize,int *eltsize,void *bytes);
void CPathMsgFree(void *msg);

#define CPATH_ALL    (-1)
#define CPATH_END      0
#define CPATH_DEST     1
#define CPATH_DESTELT  2
#define CPATH_TAG      3
#define CPATH_TAGS     4
#define CPATH_TAGVEC   5
#define CPATH_BYTES    6
#define CPATH_OVER     7
#define CPATH_REDUCER  8
#define CPATH_REDBYTES 9

/******** CONVCONDS ********/

typedef void (*CcdVoidFn)(void *);

/*CPU conditions*/
#define CcdPROCESSOR_BEGIN_BUSY 0
#define CcdPROCESSOR_END_IDLE 0 /*Synonym*/
#define CcdPROCESSOR_BEGIN_IDLE 1
#define CcdPROCESSOR_END_BUSY 1 /*Synonym*/
#define CcdPROCESSOR_STILL_IDLE 2

/*Periodic calls*/
#define CcdPERIODIC       16 /*every few ms*/
#define CcdPERIODIC_10ms  17 /*every 10ms (100Hz)*/
#define CcdPERIODIC_100ms 18 /*every 100ms (10Hz)*/
#define CcdPERIODIC_1second  19 /*every second*/
#define CcdPERIODIC_1s       19 /*every second*/
#define CcdPERIODIC_10second 20 /*every 10 seconds*/
#define CcdPERIODIC_10seconds 20 /*every 10 seconds*/
#define CcdPERIODIC_10s      20 /*every 10 seconds*/
#define CcdPERIODIC_1minute  21 /*every minute*/
#define CcdPERIODIC_10minute 22 /*every 10 minutes*/
#define CcdPERIODIC_1hour    23 /*every hour*/
#define CcdPERIODIC_12hour   24 /*every 12 hours*/
#define CcdPERIODIC_1day     25 /*every day*/

/*Other conditions*/
#define CcdQUIESCENCE 30
#define CcdSIGUSR1 32+1
#define CcdSIGUSR2 32+2

/*User-defined conditions start here*/
#define CcdUSER    48

#define CcdIGNOREPE   -2
#if CMK_CONDS_USE_SPECIAL_CODE
int CmiSwitchToPE(int pe);
#else
#define CmiSwitchToPE(pe)  pe
#endif
void CcdCallFnAfter(CcdVoidFn fnp, void *arg, unsigned int msecs);
int CcdCallOnCondition(int condnum, CcdVoidFn fnp, void *arg);
int CcdCallOnConditionKeep(int condnum, CcdVoidFn fnp, void *arg);
void CcdCallFnAfterOnPE(CcdVoidFn fnp, void *arg, unsigned int msecs, int pe);
int CcdCallOnConditionOnPE(int condnum, CcdVoidFn fnp, void *arg, int pe);
int CcdCallOnConditionKeepOnPE(int condnum, CcdVoidFn fnp, void *arg, int pe);
void CcdCancelCallOnCondition(int condnum, int idx);
void CcdCancelCallOnConditionKeep(int condnum, int idx);
void CcdRaiseCondition(int condnum);

/* Command-Line-Argument handling */
int CmiGetArgString(char **argv,const char *arg,char **optDest);
int CmiGetArgInt(char **argv,const char *arg,int *optDest);
int CmiGetArgFlag(char **argv,const char *arg);
void CmiDeleteArgs(char **argv,int k);
int CmiGetArgc(char **argv);
char **CmiCopyArgs(char **argv);

#if CMK_CMIDELIVERS_USE_COMMON_CODE
CpvExtern(void*, CmiLocalQueue);
#endif

/*****************************************************************************
 *
 *    Converse Quiescence Detection
 *
 *****************************************************************************/

struct ConvQdMsg;
struct ConvQdState;
typedef struct ConvQdMsg    *CQdMsg;
typedef struct ConvQdState  *CQdState;
typedef CcdVoidFn CQdVoidFn; 

CpvExtern(CQdState, cQdState);

void CQdInit(void);
void CQdCreate(CQdState, int);
void CQdProcess(CQdState, int);
int  CQdGetCreated(CQdState);
int  CQdGetProcessed(CQdState);
void CQdRegisterCallback(CQdVoidFn, void *);
void CmiStartQD(CQdVoidFn, void *);

#include "conv-random.h"
#include "conv-lists.h"
#include "conv-trace.h"

/**** DEAL WITH DIFFERENCES: KERNIGHAN-RITCHIE-C, ANSI-C, AND C++ ****/

#if defined(__cplusplus)
}
#endif

#include "debug-conv.h"


#endif /* CONVERSE_H */
