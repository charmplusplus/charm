#ifndef CONVERSE_H
#define CONVERSE_H

#ifndef _conv_mach_h
#include "conv-mach.h"
#endif

/**** DEAL WITH DIFFERENCES: KERNIGHAN-RITCHIE-C, ANSI-C, AND C++ ****/

#define CMK_CONCAT(x,y) x##y

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

#if CMK_SHARED_VARS_UNAVAILABLE

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

#define SHARED_DECL
#define CpvDeclare(t,v) t CMK_CONCAT(Cpv_Var_,v)
#define CpvExtern(t,v)  extern t CMK_CONCAT(Cpv_Var_,v)
#define CpvStaticDeclare(t,v) static t CMK_CONCAT(Cpv_Var_,v)
#define CpvInitialize(t,v) do {} while(0)
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)

#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v) do{}while(0)
#define CsvExtern(t,v) extern t CMK_CONCAT(Csv_Var_,v)
#define CsvAccess(v) CMK_CONCAT(Csv_Var_,v)

extern void CmiMemLock();
extern void CmiMemUnlock();
#define CmiNodeBarrier() 0
#define CmiSvAlloc CmiAlloc

typedef void *CmiNodeLock;
#define CmiCreateLock() ((void *)0)
#define CmiLock(lock) 0
#define CmiUnlock(lock) 0
#define CmiTryLock(lock) 0
#define CmiDestroyLock(lock) do{}while(0)

#endif

#if CMK_SHARED_VARS_EXEMPLAR

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

#define SHARED_DECL
#define CpvDeclare(t,v) t* CMK_CONCAT(Cpv_Var_,v)
#define CpvExtern(t,v)  extern t* CMK_CONCAT(Cpv_Var_,v)
#define CpvStaticDeclare(t,v) static t* CMK_CONCAT(Cpv_Var_,v)
#define CpvInitialize(t,v)\
    do { if (CmiMyRank()) CmiNodeBarrier();\
    else { CMK_CONCAT(Cpv_Var_,v)=(t*)malloc(sizeof(t)*CmiMyNodeSize());\
           CmiNodeBarrier();}} while(0)
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)[CmiMyRank()]

#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvExtern(t,v) extern t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v) do{}while(0)
#define CsvAccess(v) CMK_CONCAT(Csv_Var_,v)


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

#if CMK_SHARED_VARS_SUN_THREADS

#include <thread.h>
#include <synch.h>
#include <stdlib.h>

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

#define SHARED_DECL

#define CpvDeclare(t,v) t* CMK_CONCAT(Cpv_Var_,v)
#define CpvExtern(t,v)  extern t* CMK_CONCAT(Cpv_Var_,v)
#define CpvStaticDeclare(t,v) static t* CMK_CONCAT(Cpv_Var_,v)
#define CpvInitialize(t,v)\
  do { if (CmiMyRank()) while (CMK_CONCAT(Cpv_Var_,v)==0) thr_yield();\
       else { CMK_CONCAT(Cpv_Var_,v)=(t*)malloc(sizeof(t)*CmiMyNodeSize()); }}\
  while(0)
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)[CmiMyRank()]

#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvExtern(t,v) extern t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v) do{}while(0)
#define CsvAccess(v) CMK_CONCAT(Csv_Var_,v)

extern void CmiMemLock();
extern void CmiMemUnlock();
extern void CmiNodeBarrier(void);
#define CmiSvAlloc CmiAlloc


typedef mutex_t *CmiNodeLock;
extern CmiNodeLock CmiCreateLock();
#define CmiLock(lock) (mutex_lock(lock))
#define CmiUnlock(lock) (mutex_unlock(lock))
#define CmiTryLock(lock) (mutex_trylock(lock))
extern void CmiDestroyLock(CmiNodeLock lock);

#endif

#if CMK_SHARED_VARS_POSIX_THREADS_SMP

#include <pthread.h>
#include <sched.h>

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

#define SHARED_DECL

#define CpvDeclare(t,v) t* CMK_CONCAT(Cpv_Var_,v)
#define CpvExtern(t,v)  extern t* CMK_CONCAT(Cpv_Var_,v)
#define CpvStaticDeclare(t,v) static t* CMK_CONCAT(Cpv_Var_,v)
#define CpvInitialize(t,v)\
  do { if (CmiMyRank()) while (CMK_CONCAT(Cpv_Var_,v)==0) sched_yield();\
       else { CMK_CONCAT(Cpv_Var_,v)=(t*)malloc(sizeof(t)*CmiMyNodeSize()); }}\
  while(0)
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)[CmiMyRank()]

#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvExtern(t,v) extern t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v) do{}while(0)
#define CsvAccess(v) CMK_CONCAT(Csv_Var_,v)

extern void CmiMemLock();
extern void CmiMemUnlock();
extern void CmiNodeBarrier(void);
#define CmiSvAlloc CmiAlloc


typedef pthread_mutex_t *CmiNodeLock;
extern CmiNodeLock CmiCreateLock();
#define CmiLock(lock) (pthread_mutex_lock(lock))
#define CmiUnlock(lock) (pthread_mutex_unlock(lock))
#define CmiTryLock(lock) (pthread_mutex_trylock(lock))
extern void CmiDestroyLock(CmiNodeLock lock);

#endif

#if CMK_SHARED_VARS_UNIPROCESSOR

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

#define SHARED_DECL

#define CpvDeclare(t,v) t* CMK_CONCAT(Cpv_Var_,v)
#define CpvExtern(t,v)  extern t* CMK_CONCAT(Cpv_Var_,v)
#define CpvStaticDeclare(t,v) static t* CMK_CONCAT(Cpv_Var_,v)
#define CpvInitialize(t,v)\
  do  { if (CMK_CONCAT(Cpv_Var_,v)==0)\
        { CMK_CONCAT(Cpv_Var_,v) = (t *)CmiAlloc(CmiNumPes()*sizeof(t)); }}\
  while(0)
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)[CmiMyPe()]

#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvExtern(t,v) extern t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v) do{}while(0)
#define CsvAccess(v) CMK_CONCAT(Csv_Var_,v)

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

#if CMK_SHARED_VARS_PTHREADS

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

#define SHARED_DECL

#define CpvDeclare(t,v) t* CMK_CONCAT(Cpv_Var_,v)
#define CpvExtern(t,v)  extern t* CMK_CONCAT(Cpv_Var_,v)
#define CpvStaticDeclare(t,v) static t* CMK_CONCAT(Cpv_Var_,v)
#define CpvInitialize(t,v)\
  do { if (CmiMyRank()) while (CMK_CONCAT(Cpv_Var_,v)==0) sched_yield();\
       else {CMK_CONCAT(Cpv_Var_,v)=(t*)CmiAlloc(sizeof(t)*CmiMyNodeSize());}}\
  while(0)
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)[CmiMyRank()]

#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvExtern(t,v) extern t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v) do{}while(0)
#define CsvAccess(v) CMK_CONCAT(Csv_Var_,v)

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

#if CMK_SHARED_VARS_NT_THREADS

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

#define SHARED_DECL

#define CpvDeclare(t,v) t* CMK_CONCAT(Cpv_Var_,v)
#define CpvExtern(t,v)  extern t* CMK_CONCAT(Cpv_Var_,v)
#define CpvStaticDeclare(t,v) static t* CMK_CONCAT(Cpv_Var_,v)
#define CpvInitialize(t,v)\
 do { if (CmiMyRank()) while (CMK_CONCAT(Cpv_Var_,v)==0) Sleep(0);\
    else { CMK_CONCAT(Cpv_Var_,v)=(t*)malloc(sizeof(t)*CmiMyNodeSize()); }} \
 while(0)
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)[CmiMyRank()]

#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvExtern(t,v) extern t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v) do{}while(0)
#define CsvAccess(v) CMK_CONCAT(Csv_Var_,v)

extern void CmiMemLock();
extern void CmiMemUnlock();
extern void CmiNodeBarrier(void);
#define CmiSvAlloc CmiAlloc


typedef HANDLE CmiNodeLock;
extern  CmiNodeLock CmiCreateLock();
#define CmiLock(lock) (WaitForSingleObject(lock, INFINITE))
#define CmiUnlock(lock) (ReleaseMutex(lock))
#define CmiTryLock(lock) (WaitForSingleObject(lock 0))
extern  void CmiDestroyLock(CmiNodeLock lock);

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

#if CMK_COMMHANDLE_IS_A_POINTER
typedef void  *CmiCommHandle;
#endif

#if CMK_COMMHANDLE_IS_AN_INTEGER
typedef int    CmiCommHandle;
#endif

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
CpvExtern(CmiHandler,  CsdNotifyIdle);
CpvExtern(CmiHandler,  CsdNotifyBusy);
CpvExtern(int,         CsdStopNotifyFlag);

extern int CmiRegisterHandler(CmiHandler);
extern int CmiRegisterHandlerLocal(CmiHandler);
extern int CmiRegisterHandlerGlobal(CmiHandler);
extern void CmiNumberHandler(int, CmiHandler);

/*
 * I'm planning on doing the byte-order conversion slightly differently
 * now.  The repair of the header will be done in the machine layer,
 * just after receiving a message.  This will be cheaper than doing it
 * here.  Since the CMI can only repair the header and not the contents
 * of the message, we provide these functions that the user can use to
 * repair the contents of the message.
 *
 * CmiConvertInt2(msg, p)
 * CmiConvertInt4(msg, p)
 * CmiConvertInt8(msg, p)
 * CmiConvertFloat4(msg, p)
 * CmiConvertFloat8(msg, p)
 * CmiConvertFloat16(msg, p)
 *
 *   Given a message and a pointer to a number in that message,
 *   converts the number in-place.  This accounts for the byte-order
 *   and other format peculiarities of the sender.
 *
 * CmiConversionNeeded(msg)
 *
 *   When speed is of the essence, this function may make it possible
 *   to skip some conversions.  It returns a combination of the following
 *   flags:
 *
 *   CMI_CONVERT_INTS_BACKWARD   - ints are in backward byte-order.
 *   CMI_CONVERT_INTS_FOREIGN    - ints are in a wildly different format.
 *   CMI_CONVERT_FLOATS_BACKWARD - floats are in backward byte-order.
 *   CMI_CONVERT_FLOATS_FOREIGN  - floats are in a wildly different format.
 *
 * If neither bit is set, the numbers are in local format, and no
 * conversion is needed whatsoever.  Thus, a value of 0 indicates that
 * the message is entirely in local format.  If the values are in wildly
 * different format, one has no choice but to use the CmiConvert functions.
 * If they're just in backward-byte-order, you can swap the bytes yourself,
 * possibly faster than we can.
 *
 */

#define CmiConversionNeeded(m) 0
#define CmiConvertInt2(m,p) 0
#define CmiConvertInt4(m,p) 0
#define CmiConvertInt8(m,p) 0
#define CmiConvertFloat4(m,p) 0
#define CmiConvertFloat8(m,p) 0
#define CmiConvertFloat16(m,p) 0

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

double   CmiTimer();
double   CmiWallTimer();
double   CmiCpuTimer();

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

extern int CqsEmpty(void *);

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

typedef void (*CmiStartFn)(int argc, char **argv);

/********* CSD - THE SCHEDULER ********/

extern  int CsdScheduler(int);
#define CsdSetNotifyIdle(f1,f2) do {CpvAccess(CsdNotifyIdle)=(f1);\
                                 CpvAccess(CsdNotifyBusy)=(f2);} while(0)
#define CsdStartNotifyIdle() (CpvAccess(CsdStopNotifyFlag)=0)
#define CsdStopNotifyIdle() (CpvAccess(CsdStopNotifyFlag)=1)

#if CMK_CSDEXITSCHEDULER_IS_A_FUNCTION
extern void CsdExitScheduler(void);
#endif 

#if CMK_CSDEXITSCHEDULER_SET_CSDSTOPFLAG
#define CsdExitScheduler()  (CpvAccess(CsdStopFlag)++)
#endif

void     CmiGrabBuffer(void **ppbuf);
void     CmiReleaseBuffer(void *pbuf);

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

/******** CQS: THE QUEUEING SYSTEM ********/

#define CQS_QUEUEING_FIFO 2
#define CQS_QUEUEING_LIFO 3
#define CQS_QUEUEING_IFIFO 4
#define CQS_QUEUEING_ILIFO 5
#define CQS_QUEUEING_BFIFO 6
#define CQS_QUEUEING_BLIFO 7

/****** CTH: THE LOW-LEVEL THREADS PACKAGE ******/

typedef struct CthThreadStruct *CthThread;

typedef void        (*CthVoidFn)();
typedef CthThread   (*CthThFn)();

int        CthImplemented(void);

CthThread  CthSelf(void);
CthThread  CthCreate(CthVoidFn, void *, int);
void       CthResume(CthThread);
void       CthFree(CthThread);

void       CthSetSuspendable(CthThread, int);
int        CthIsSuspendable(CthThread);

void       CthSuspend(void);
void       CthAwaken(CthThread);
void       CthSetStrategy(CthThread, CthVoidFn, CthThFn);
void       CthSetStrategyDefault(CthThread);
void       CthYield(void);

void       CthSetNext(CthThread t, CthThread next);
CthThread  CthGetNext(CthThread t);

void       CthAutoYield(CthThread t, int flag);
double     CthAutoYieldFreq(CthThread t);
void       CthAutoYieldBlock(void);
void       CthAutoYieldUnblock(void);

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

extern int CthPackBufSize(CthThread);
extern void CthPackThread(CthThread, void *);
extern CthThread CthUnpackThread(void *);

CthCpvExtern(char *,CthData);
extern int CthRegister(int);
#define CtvDeclare(t,v)         typedef t CtvType##v; CsvDeclare(int,CtvOffs##v);
#define CtvStaticDeclare(t,v)   typedef t CtvType##v; CsvStaticDeclare(int,CtvOffs##v);
#define CtvExtern(t,v)          typedef t CtvType##v; CsvExtern(int,CtvOffs##v);
#define CtvAccess(v)            (*((CtvType##v *)(CthCpvAccess(CthData)+CsvAccess(CtvOffs##v))))
#define CtvInitialize(t,v)      (CsvAccess(CtvOffs##v)=CthRegister(sizeof(CtvType##v)));

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
CpmDestination CpmEnqueueBFIFO(int pe, int priobits, int *prioptr);
CpmDestination CpmEnqueueBLIFO(int pe, int priobits, int *prioptr);
CpmDestination CpmEnqueue(int pe,int qs,int priobits,int *prioptr);

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
void CldNodeEnqueue(int node, void *msg, int infofn);

/****** CMM: THE MESSAGE MANAGER ******/

typedef struct CmmTableStruct *CmmTable;

#define CmmWildCard (-1)

CmmTable   CmmNew();
void       CmmFree(CmmTable t);
void       CmmPut(CmmTable t, int ntags, int *tags, void *msg);
void      *CmmFind(CmmTable t, int ntags, int *tags, int *returntags, int del);
#define    CmmGet(t,nt,tg,rt)   (CmmFind((t),(nt),(tg),(rt),1))
#define    CmmProbe(t,nt,tg,rt) (CmmFind((t),(nt),(tg),(rt),0))

/******** ConverseInit and ConverseExit ********/

void ConverseInit(int, char**, CmiStartFn, int, int);
void ConverseExit(void);

void CmiAbort(const char *);

#ifndef CMK_OPTIMIZE
#define _MEMCHECK(p) do { \
                         if ((p)==0) CmiAbort("Memory Allocation Failure.\n");\
                     } while(0)
#else
#define _MEMCHECK(p) do{}while(0)
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

typedef void (*CcdVoidFn)();

#define CcdPROCESSORIDLE 1
#define CcdSIGUSR1 2
#define CcdSIGUSR2 3
#define CcdQUIESCENCE 4

void CcdCallFnAfter(CcdVoidFn fnp, void *arg, unsigned int msecs);
void CcdPeriodicallyCall(CcdVoidFn fnp, void *arg);

void CcdRaiseCondition(int condnum);
void CcdCallOnCondition(int condnum, CcdVoidFn fnp, void *arg);

void CcdCallBacks();

/******** Parallel Debugger *********/

#if CMK_DEBUG_MODE

#include "conv-ccs.h"

CpvExtern(void *, debugQueue);

void CpdInit(void);
void CpdFreeze(void);
void CpdUnFreeze(void);

void CpdInitializeObjectTable();
void CpdInitializeHandlerArray();
void CpdInitializeBreakPoints();

#define MAX_NUM_HANDLERS 1000
typedef char* (*hndlrIDFunction)(char *);
typedef hndlrIDFunction handlerType[MAX_NUM_HANDLERS][2];
void handlerArrayRegister(int, hndlrIDFunction, hndlrIDFunction);

typedef int (*indirectionFunction)(char *);
typedef indirectionFunction indirectionType[MAX_NUM_HANDLERS];

typedef char* (*symbolTableFunction)(void);
typedef symbolTableFunction symbolTableType[MAX_NUM_HANDLERS];

void symbolTableFnArrayRegister(int hndlrID, int noOfBreakPoints,
				symbolTableFunction f, indirectionFunction g);
char* getSymbolTableInfo();
int isBreakPoint(char *msg);
int isEntryPoint(char *msg);
void setBreakPoints(char *);
char *getBreakPoints();

char* getObjectList();
char* getObjectContents(int);

void msgListCache();
void msgListCleanup();

char* genericViewMsgFunction(char *msg, int type);
char* getMsgListSched();
char* getMsgListPCQueue();
char* getMsgListFIFO();
char* getMsgListDebug();
char* getMsgContentsSched(int index);
char* getMsgContentsPCQueue(int index);
char* getMsgContentsFIFO(int index);
char* getMsgContentsDebug(int index);

#endif

#if CMK_WEB_MODE

#include "conv-ccs.h"

void CWebInit (void);
void CWebPlateRegisterCell (void);
void CWebPlateDataDeposit (int timestep, int cellx, int celly, 
                           int rows, int columns, int **data);
#endif

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

/*****************************************************************************
 *
 *    Converse Random Numbers
 *
 *****************************************************************************/

typedef struct rngen_
{
  unsigned int prime;
  double state[3], multiplier[3];/* simulate 64 bit arithmetic */
} CrnStream;

void CrnInitStream(CrnStream *, int, int);
int CrnInt(CrnStream *);
double CrnDouble(CrnStream *);
float CrnFloat(CrnStream *);
void CrnSrand(int);
int CrnRand(void);
double CrnDrand(void);

#include "conv-trace.h"

/**** DEAL WITH DIFFERENCES: KERNIGHAN-RITCHIE-C, ANSI-C, AND C++ ****/

#if defined(__cplusplus)
}
#endif

#endif /* CONVERSE_H */
