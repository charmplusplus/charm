/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.77  1998-01-16 18:03:02  milind
 * Fixed Ctv bug on shared memory machines.
 * Made latencyBWtest conformant with Converse.
 * Added high resolution timers to Origin2000.
 * Removed starvation from Origin Pthreads version.
 *
 * Revision 2.76  1997/12/10 21:01:06  jyelon
 * *** empty log message ***
 *
 * Revision 2.75  1997/11/26 19:17:24  milind
 * Fixed some portability bugs due to varying integer and pointer sizes.
 *
 * Revision 2.74  1997/10/29 18:47:05  jyelon
 * Added CmiHandlerToFunction
 *
 * Revision 2.73  1997/07/31 00:28:25  jyelon
 * *** empty log message ***
 *
 * Revision 2.72  1997/07/30 17:31:06  jyelon
 * *** empty log message ***
 *
 *
 ***************************************************************************/

#ifndef CONVERSE_H
#define CONVERSE_H

#ifndef _conv_mach_h
#include "conv-mach.h"
#endif

/**** DEAL WITH DIFFERENCES: KERNIGHAN-RITCHIE-C, ANSI-C, AND C++ ****/

#if CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION
#define CMK_CONCAT(x,y) y
#endif

#if CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION
#define CMK_CONCAT(x,y) x##y
#endif

/* the following flags denote properties of the C compiler,  */
/* not the C++ compiler.  If this is C++, ignore them.       */
#ifdef __cplusplus
#undef CMK_PROTOTYPES_FAIL
#undef CMK_PROTOTYPES_WORK
#define CMK_PROTOTYPES_FAIL 0
#define CMK_PROTOTYPES_WORK 1
#undef CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION
#undef CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION
#define CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION 0
#define CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION 1
extern "C" {
#endif

#if CMK_PROTOTYPES_WORK
#define CMK_PROTO(x) x
#endif

#if CMK_PROTOTYPES_FAIL
#define CMK_PROTO(x) ()
#endif

#if CMK_STATIC_PROTO_WORKS
#define CMK_STATIC_PROTO static
#endif

#if CMK_STATIC_PROTO_FAILS
#define CMK_STATIC_PROTO extern
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
#define CpvInitialize(t,v) 
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)

#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v) 
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
#define CmiDestroyLock(lock) 0

#endif

#if CMK_SHARED_VARS_EXEMPLAR

#include <spp_prog_model.h>
#include <cps.h>

extern int Cmi_numpes;
extern int Cmi_mynodesize;

#define CmiMyPe()           (my_thread())
#define CmiMyRank()         (my_thread())
#define CmiNumPes()         Cmi_numpes
#define CmiMyNodeSize()     Cmi_mynodesize
#define CmiMyNode()         0
#define CmiNumNodes()       Cmi_numpes
#define CmiNodeFirst(node)  0
#define CmiNodeSize(node)   Cmi_numpes
#define CmiNodeOf(pe)       0
#define CmiRankOf(pe)       (pe)

#define SHARED_DECL
#define CpvDeclare(t,v) t* CMK_CONCAT(Cpv_Var_,v)
#define CpvExtern(t,v)  extern t* CMK_CONCAT(Cpv_Var_,v)
#define CpvStaticDeclare(t,v) static t* CMK_CONCAT(Cpv_Var_,v)
#define CpvInitialize(t,v)\
    { if (CmiMyRank()) while (CMK_CONCAT(Cpv_Var_,v)==0);\
    else { CMK_CONCAT(Cpv_Var_,v)=(t*)malloc(sizeof(t)*CmiMyNodeSize()); }}
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)[CmiMyRank()]

#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvExtern(t,v) extern t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v)
#define CsvAccess(v) CMK_CONCAT(Csv_Var_,v)


extern void CmiMemLock();
extern void CmiMemUnlock();
extern void CmiNodeBarrier CMK_PROTO((void));
extern void *CmiSvAlloc CMK_PROTO((int));

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
  { if (CmiMyRank()) while (CMK_CONCAT(Cpv_Var_,v)==0) thr_yield();\
    else { CMK_CONCAT(Cpv_Var_,v)=(t*)malloc(sizeof(t)*CmiMyNodeSize()); }}
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)[CmiMyRank()]

#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvExtern(t,v) extern t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v)
#define CsvAccess(v) CMK_CONCAT(Csv_Var_,v)

extern void CmiMemLock();
extern void CmiMemUnlock();
#define CmiNodeBarrier() 0
#define CmiSvAlloc CmiAlloc


typedef mutex_t *CmiNodeLock;
extern CmiNodeLock CmiCreateLock();
#define CmiLock(lock) (mutex_lock(lock))
#define CmiUnlock(lock) (mutex_unlock(lock))
#define CmiTryLock(lock) (mutex_trylock(lock))
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
    { if (CMK_CONCAT(Cpv_Var_,v)==0)\
        { CMK_CONCAT(Cpv_Var_,v) = (t *)CmiAlloc(CmiNumPes()*sizeof(t)); }}
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)[CmiMyPe()]

#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvExtern(t,v) extern t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v)
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
  { if (CmiMyRank()) while (CMK_CONCAT(Cpv_Var_,v)==0) sched_yield();\
    else { CMK_CONCAT(Cpv_Var_,v)=(t*)CmiAlloc(sizeof(t)*CmiMyNodeSize()); }}
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)[CmiMyRank()]

#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvExtern(t,v) extern t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v)
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

/******** CMI: TYPE DEFINITIONS ********/

#define CmiMsgHeaderSizeBytes CMK_MSG_HEADER_SIZE_BYTES

#if CMK_COMMHANDLE_IS_A_POINTER
typedef void  *CmiCommHandle;
#endif

#if CMK_COMMHANDLE_IS_AN_INTEGER
typedef int    CmiCommHandle;
#endif


typedef void (*CmiHandler)();

/******** Basic Types ********/

#ifdef CMK_NUMBERS_NORMAL

typedef short                   CInt2;
typedef int                     CInt4;
typedef long long               CInt8;
typedef unsigned short          CUInt2;
typedef unsigned int            CUInt4;
typedef unsigned long long      CUInt8;
typedef float                   CFloat4;
typedef double                  CFloat8;
typedef struct { char c[16]; }  CFloat16;

#endif

/******** CMI, CSD: MANY LOW-LEVEL OPERATIONS ********/

CpvExtern(CmiHandler*, CmiHandlerTable);
CpvExtern(void*,       CsdSchedQueue);
CpvExtern(int,         CsdStopFlag);
CpvExtern(CmiHandler, CsdNotifyIdle);
CpvExtern(CmiHandler, CsdNotifyBusy);
CpvExtern(int,        CsdStopNotifyFlag);

extern int CmiRegisterHandler CMK_PROTO((CmiHandler));
extern int CmiRegisterHandlerLocal CMK_PROTO((CmiHandler));
extern int CmiRegisterHandlerGlobal CMK_PROTO((CmiHandler));
extern void CmiNumberHandler CMK_PROTO((int, CmiHandler));

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

#define CmiHandlerAccess(m)\
  (*((int*)(((char *)(m))+CMK_MSG_HEADER_BLANK_SPACE)))

#define CmiGetHandler(env)  (CmiHandlerAccess(env))
#define CmiSetHandler(env,x)  (CmiHandlerAccess(env) = (x))
#define CmiHandlerToFunction(n) (CpvAccess(CmiHandlerTable)[n])
#define CmiGetHandlerFunction(env) (CmiHandlerToFunction(CmiGetHandler(env)))

void    *CmiAlloc  CMK_PROTO((int size));
int      CmiSize   CMK_PROTO((void *));
void     CmiFree   CMK_PROTO((void *));

double   CmiTimer      CMK_PROTO(());
double   CmiWallTimer  CMK_PROTO(());
double   CmiCpuTimer   CMK_PROTO(());

#define CsdEnqueueGeneral(x,s,i,p)\
    (CqsEnqueueGeneral(CpvAccess(CsdSchedQueue),(x),(s),(i),(p)))
#define CsdEnqueueFifo(x)     (CqsEnqueueFifo(CpvAccess(CsdSchedQueue),(x)))
#define CsdEnqueueLifo(x)     (CqsEnqueueLifo(CpvAccess(CsdSchedQueue),(x)))
#define CsdEnqueue(x)         (CqsEnqueueFifo(CpvAccess(CsdSchedQueue),(x)))
#define CsdEmpty()            (CqsEmpty(CpvAccess(CsdSchedQueue)))

#if CMK_CMIPRINTF_IS_A_BUILTIN
void  CmiPrintf CMK_PROTO((char *, ...));
void  CmiError  CMK_PROTO((char *, ...));
int   CmiScanf  CMK_PROTO((char *, ...));
#endif

#if CMK_CMIPRINTF_IS_JUST_PRINTF
#include <stdio.h>

#define CmiPrintf printf
#define CmiError  printf
#define CmiScanf  scanf
#endif

typedef void (*CmiStartFn) CMK_PROTO((int argc, char **argv));

/********* CSD - THE SCHEDULER ********/

extern  int CsdScheduler CMK_PROTO((int));
#define CsdSetNotifyIdle(f1,f2) {CpvAccess(CsdNotifyIdle)=(f1);\
                                 CpvAccess(CsdNotifyBusy)=(f2);}
#define CsdStartNotifyIdle() (CpvAccess(CsdStopNotifyFlag)=0)
#define CsdStopNotifyIdle() (CpvAccess(CsdStopNotifyFlag)=1)

#if CMK_CSDEXITSCHEDULER_IS_A_FUNCTION
extern void CsdExitScheduler CMK_PROTO((void));
#endif 

#if CMK_CSDEXITSCHEDULER_SET_CSDSTOPFLAG
#define CsdExitScheduler()  (CpvAccess(CsdStopFlag)++)
#endif

int      CmiSpanTreeRoot         CMK_PROTO(()) ;
int      CmiNumSpanTreeChildren  CMK_PROTO((int)) ;
int      CmiSpanTreeParent       CMK_PROTO((int)) ;
void     CmiSpanTreeChildren     CMK_PROTO((int node, int *children)) ;

/****** CMI MESSAGE TRANSMISSION ******/

void          CmiSyncSendFn        CMK_PROTO((int, int, char *));
CmiCommHandle CmiAsyncSendFn       CMK_PROTO((int, int, char *));
void          CmiFreeSendFn        CMK_PROTO((int, int, char *));

void          CmiSyncBroadcastFn      CMK_PROTO((int, char *));
CmiCommHandle CmiAsyncBroadcastFn     CMK_PROTO((int, char *));
void          CmiFreeBroadcastFn      CMK_PROTO((int, char *));

void          CmiSyncBroadcastAllFn   CMK_PROTO((int, char *));
CmiCommHandle CmiAsyncBroadcastAllFn  CMK_PROTO((int, char *));
void          CmiFreeBroadcastAllFn   CMK_PROTO((int, char *));


#define CmiSyncSend(p,s,m)              (CmiSyncSendFn((p),(s),(char *)(m)))
#define CmiAsyncSend(p,s,m)             (CmiAsyncSendFn((p),(s),(char *)(m)))
#define CmiSyncSendAndFree(p,s,m)       (CmiFreeSendFn((p),(s),(char *)(m)))

#define CmiSyncBroadcast(s,m)           (CmiSyncBroadcastFn((s),(char *)(m)))
#define CmiAsyncBroadcast(s,m)          (CmiAsyncBroadcastFn((s),(char *)(m)))
#define CmiSyncBroadcastAndFree(s,m)    (CmiFreeBroadcastFn((s),(char *)(m)))

#define CmiSyncBroadcastAll(s,m)        (CmiSyncBroadcastAllFn((s),(char *)(m)))
#define CmiAsyncBroadcastAll(s,m)       (CmiAsyncBroadcastAllFn((s),(char *)(m)))
#define CmiSyncBroadcastAllAndFree(s,m) (CmiFreeBroadcastAllFn((s),(char *)(m)))

/****** CMI VECTOR MESSAGE TRANSMISSION ******/

void CmiSyncVectorSend CMK_PROTO((int, int, int *, char **));
CmiCommHandle CmiAsyncVectorSend CMK_PROTO((int, int, int *, char **));
void CmiSyncVectorSendAndFree CMK_PROTO((int, int, int *, char **));

/******** CMI MESSAGE RECEPTION ********/

int    CmiDeliverMsgs          CMK_PROTO((int maxmsgs));
void   CmiDeliverSpecificMsg   CMK_PROTO((int handler));

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

int        CthImplemented  CMK_PROTO((void));

CthThread  CthSelf     CMK_PROTO((void));
CthThread  CthCreate   CMK_PROTO((CthVoidFn, void *, int));
void       CthResume   CMK_PROTO((CthThread));
void       CthFree     CMK_PROTO((CthThread));

void       CthSuspend             CMK_PROTO((void));
void       CthAwaken              CMK_PROTO((CthThread));
void       CthSetStrategy         CMK_PROTO((CthThread, CthVoidFn, CthThFn));
void       CthSetStrategyDefault  CMK_PROTO((CthThread));
void       CthYield               CMK_PROTO((void));

void       CthSetNext CMK_PROTO((CthThread t, CthThread next));
CthThread  CthGetNext CMK_PROTO((CthThread t));

void       CthAutoYield           CMK_PROTO((CthThread t, int flag));
double     CthAutoYieldFreq       CMK_PROTO((CthThread t));
void       CthAutoYieldBlock      CMK_PROTO((void));
void       CthAutoYieldUnblock    CMK_PROTO((void));

/****** CTH: THREAD-PRIVATE VARIABLES ******/

#if CMK_THREADS_REQUIRE_NO_CPV

#define CthCpvDeclare(t,v)    t v
#define CthCpvExtern(t,v)     extern t v
#define CthCpvStatic(t,v)     static t v
#define CthCpvInitialize(t,v) 
#define CthCpvAccess(x)       x

#else

#define CthCpvDeclare(t,v)    CpvDeclare(t,v)
#define CthCpvExtern(t,v)     CpvExtern(t,v)
#define CthCpvStatic(t,v)     CpvStaticDeclare(t,v)
#define CthCpvInitialize(t,v) CpvInitialize(t,v)
#define CthCpvAccess(x)       CpvAccess(x)

#endif

CthCpvExtern(char *,CthData);
extern int CthRegister CMK_PROTO((int));
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

typedef void *(*CpmSender) CMK_PROTO((CpmDestination, int, void *));

struct CpmDestinationStruct
{
  CpmSender sendfn;
  int envsize;
};

#define CpmPE(n) n
#define CpmALL (-1)
#define CpmOTHERS (-2)

CpmDestination CpmSend CMK_PROTO((int pe));
CpmDestination CpmMakeThread CMK_PROTO((int pe));
CpmDestination CpmMakeThreadSize CMK_PROTO((int pe, int size));
CpmDestination CpmEnqueueFIFO CMK_PROTO((int pe));
CpmDestination CpmEnqueueLIFO CMK_PROTO((int pe));
CpmDestination CpmEnqueueIFIFO CMK_PROTO((int pe, int prio));
CpmDestination CpmEnqueueILIFO CMK_PROTO((int pe, int prio));
CpmDestination CpmEnqueueBFIFO CMK_PROTO((int pe, int priobits, int *prioptr));
CpmDestination CpmEnqueueBLIFO CMK_PROTO((int pe, int priobits, int *prioptr));
CpmDestination CpmEnqueue CMK_PROTO((int pe,int qs,int priobits,int *prioptr));

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
#define CpmPack_char(v)
#define CpmUnpack_char(v)

CpmDeclareSimple(short);
#define CpmPack_short(v)
#define CpmUnpack_short(v)

CpmDeclareSimple(int);
#define CpmPack_int(v)
#define CpmUnpack_int(v)

CpmDeclareSimple(long);
#define CpmPack_long(v)
#define CpmUnpack_long(v)

CpmDeclareSimple(float);
#define CpmPack_float(v)
#define CpmUnpack_float(v)

CpmDeclareSimple(double);
#define CpmPack_double(v)
#define CpmUnpack_double(v)

typedef int CpmDim;
CpmDeclareSimple(CpmDim);
#define CpmPack_CpmDim(v)
#define CpmUnpack_CpmDim(v)

CpmDeclareSimple(Cfuture);
#define CpmPack_Cfuture(v)
#define CpmUnpack_Cfuture(v)

typedef char *CpmStr;
CpmDeclarePointer(CpmStr);
#define CpmPtrSize_CpmStr(v) (strlen(v)+1)
#define CpmPtrPack_CpmStr(p, v) (strcpy(p, v))
#define CpmPtrUnpack_CpmStr(v)
#define CpmPtrFree_CpmStr(v)

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

/****** CMM: THE MESSAGE MANAGER ******/

typedef struct CmmTableStruct *CmmTable;

#define CmmWildCard (-1)

CmmTable   CmmNew();
void       CmmFree CMK_PROTO((CmmTable t));
void       CmmPut CMK_PROTO((CmmTable t, int ntags, int *tags, void *msg));
void      *CmmFind CMK_PROTO((CmmTable t, int ntags, int *tags, int *returntags, int del));
#define    CmmGet(t,nt,tg,rt)   (CmmFind((t),(nt),(tg),(rt),1))
#define    CmmProbe(t,nt,tg,rt) (CmmFind((t),(nt),(tg),(rt),0))

/******** ConverseInit and ConverseExit ********/

void ConverseInit CMK_PROTO((int, char**, CmiStartFn, int, int));
void ConverseExit CMK_PROTO((void));

/******** CONVCONDS ********/

typedef void (*CcdVoidFn)();

#define CcdPROCESSORIDLE 1

void CcdCallFnAfter CMK_PROTO((CcdVoidFn fnp, void *arg, unsigned int msecs));
void CcdPeriodicallyCall CMK_PROTO((CcdVoidFn fnp, void *arg));

void CcdRaiseCondition CMK_PROTO((int condnum));
void CcdCallOnCondition CMK_PROTO((int condnum, CcdVoidFn fnp, void *arg));

void CcdCallBacks();

/**** DEAL WITH DIFFERENCES: KERNIGHAN-RITCHIE-C, ANSI-C, AND C++ ****/

#if defined(__cplusplus)
}
#endif

#endif /* CONVERSE_H */

