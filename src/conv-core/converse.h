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
 * Revision 2.36  1996-01-03 23:18:37  sanjeev
 * CmiSize prototype should have void * argument
 *
 * Revision 2.35  1995/10/31 19:53:21  jyelon
 * Added 'CMK_THREADS_USE_ALLOCA_WITH_PRAGMA'
 *
 * Revision 2.34  1995/10/27  22:37:08  jyelon
 * Changed NumPe -> NumPes
 *
 * Revision 2.33  1995/10/27  21:39:32  jyelon
 * changed NumPe --> NumPes
 *
 * Revision 2.32  1995/10/23  23:09:05  jyelon
 * *** empty log message ***
 *
 * Revision 2.31  1995/10/20  17:29:10  jyelon
 * *** empty log message ***
 *
 * Revision 2.30  1995/10/19  18:22:08  jyelon
 * *** empty log message ***
 *
 * Revision 2.29  1995/10/18  22:20:37  jyelon
 * Added 'eatstack' threads implementation.
 *
 * Revision 2.28  1995/10/13  22:34:07  jyelon
 * *** empty log message ***
 *
 * Revision 2.27  1995/10/13  18:14:10  jyelon
 * K&R changes, etc.
 *
 * Revision 2.26  1995/10/12  20:18:19  sanjeev
 * modified prototype for CmiPrintf etc.
 *
 * Revision 2.25  1995/10/12  18:14:18  jyelon
 * Added parentheses in macro defs.
 *
 * Revision 2.24  1995/10/11  00:36:39  jyelon
 * Added CmiInterrupt stuff.
 *
 * Revision 2.23  1995/09/30  15:03:15  jyelon
 * A few changes for threaded-uniprocessor version.
 *
 * Revision 2.22  1995/09/29  09:51:44  jyelon
 * Many corrections, added protos, CmiGet-->CmiDeliver, etc.
 *
 * Revision 2.21  1995/09/27  22:23:15  jyelon
 * Many bug-fixes.  Added Cpv macros to threads package.
 *
 * Revision 2.20  1995/09/26  18:26:00  jyelon
 * Added CthSetStrategyDefault, and cleaned up a bit.
 *
 * Revision 2.19  1995/09/20  17:22:14  jyelon
 * Added CthImplemented
 *
 * Revision 2.18  1995/09/20  16:36:56  jyelon
 * *** empty log message ***
 *
 * Revision 2.17  1995/09/20  15:09:42  sanjeev
 * fixed CmiFree, put int CmiSpanTree stuff
 *
 * Revision 2.16  1995/09/20  15:04:45  jyelon
 * *** empty log message ***
 *
 * Revision 2.15  1995/09/20  14:58:12  jyelon
 * Did some work on threads stuff.
 *
 * Revision 2.14  1995/09/20  13:16:33  jyelon
 * Added prototypes for Cth (thread) functions.
 *
 * Revision 2.13  1995/09/19  21:43:58  brunner
 * Moved declaration of CmiTimer here from c++interface.h
 *
 * Revision 2.12  1995/09/19  19:31:51  jyelon
 * Fixed a bug.
 *
 * Revision 2.11  1995/09/19  18:57:17  jyelon
 * added CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION and other goodies.
 *
 * Revision 2.10  1995/09/07  21:14:15  jyelon
 * Added prefix to Cpv and Csv macros, then fixed bugs thereby revealed.
 *
 * Revision 2.9  1995/07/19  22:18:54  jyelon
 * *** empty log message ***
 *
 * Revision 2.8  1995/07/11  18:10:26  jyelon
 * Added CsdEnqueueFifo, etc.
 *
 * Revision 2.7  1995/07/11  16:52:37  gursoy
 * CsdExitScheduler is a function for uniprocessor now
 *
 * Revision 2.6  1995/07/07  14:04:35  gursoy
 * redone uniprocessors changes again, somehow an old version
 * is checked in
 *
 * Revision 2.5  1995/07/06  23:22:58  narain
 * Put back the different components of converse.h in this file.
 *
 * Revision 2.4  1995/06/29  21:25:29  narain
 * Split up converse.h and moved out the functionality to converse?????.h
 *
 ***************************************************************************/

#ifndef CONVERSE_H
#define CONVERSE_H

#ifndef _conv_mach_h
#include "conv-mach.h"
#endif

/**** DEAL WITH DIFFERENCES: KERNIGHAN-RITCHIE-C, ANSI-C, AND C++ ****/

#ifdef CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION
#define CMK_CONCAT(x,y) y
#endif

#ifdef CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION
#define CMK_CONCAT(x,y) x##y
#endif

/* the following flags denote properties of the C compiler,  */
/* not the C++ compiler.  If this is C++, ignore them.       */
#ifdef __cplusplus
#undef CMK_COMPILER_HATES_PROTOTYPES
#define CMK_COMPILER_LIKES_PROTOTYPES
#undef CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION
#define CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION
extern "C" {
#endif

#ifdef CMK_COMPILER_LIKES_PROTOTYPES
#define CMK_PROTO(x) x
#endif

#ifdef CMK_COMPILER_HATES_PROTOTYPES
#define CMK_PROTO(x) ()
#endif

#ifdef CMK_COMPILER_LIKES_STATIC_PROTO
#define CMK_STATIC_PROTO static
#endif

#ifdef CMK_COMPILER_HATES_STATIC_PROTO
#define CMK_STATIC_PROTO extern
#endif

/******** CPV, CSV: PRIVATE AND SHARED VARIABLES *******/

#ifdef CMK_NO_SHARED_VARS_AT_ALL

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

#define CmiMyRank() 0
#define CmiNodeBarrier()
#define CmiSvAlloc CmiAlloc

#endif


#ifdef CMK_SHARED_VARS_EXEMPLAR
#include <spp_prog_model.h>
#include <memory.h>

#define SHARED_DECL node_private
#define CpvDeclare(t,v) thread_private t CMK_CONCAT(Cpv_Var_,v)
#define CpvExtern(t,v)  extern thread_private t CMK_CONCAT(Cpv_Var_,v)
#define CpvStaticDeclare(t,v) static thread_private t CMK_CONCAT(Cpv_Var_,v)
#define CpvInitialize(t,v)
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)

#define CsvDeclare(t,v) node_private t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static node_private t CMK_CONCAT(Csv_Var_,v)
#define CsvExtern(t,v) extern node_private t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v)
#define CsvAccess(v) CMK_CONCAT(Csv_Var_,v)

extern int CmiMyRank CMK_PROTO((void));
extern void CmiNodeBarrier CMK_PROTO((void));
extern void *CmiSvAlloc CMK_PROTO((int));

#endif



#ifdef CMK_SHARED_VARS_UNIPROCESSOR

#define SHARED_DECL
#define CpvDeclare(t,v) t* CMK_CONCAT(Cpv_Var_,v)
#define CpvExtern(t,v)  extern t* CMK_CONCAT(Cpv_Var_,v)
#define CpvStaticDeclare(t,v) static t* CMK_CONCAT(Cpv_Var_,v)
#define CpvInitialize(t,v)\
    { if (CMK_CONCAT(Cpv_Var_,v)==0)\
        { CMK_CONCAT(Cpv_Var_,v) = (t *)CmiAlloc(Cmi_numpes*sizeof(t)); }}
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)[Cmi_mype]

#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvExtern(t,v) extern t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v)
#define CsvAccess(v) CMK_CONCAT(Csv_Var_,v)

#define CmiMyRank() Cmi_mype
extern void CmiNodeBarrier();
#define CmiSvAlloc CmiAlloc

#endif

/******** CMI: TYPE DEFINITIONS ********/

#ifdef CMK_COMMHANDLE_IS_A_POINTER
typedef void  *CmiCommHandle;
#endif

#ifdef CMK_COMMHANDLE_IS_AN_INTEGER
typedef int    CmiCommHandle;
#endif


typedef void (*CmiHandler)();

/******** CMI, CSD: MANY LOW-LEVEL OPERATIONS ********/

#define CmiMsgHeaderSizeBytes 4

CpvExtern(CmiHandler*, CmiHandlerTable);
CpvExtern(void*,       CsdSchedQueue);
CpvExtern(int,         CsdStopFlag);

extern int CmiRegisterHandler CMK_PROTO((CmiHandler));

#define CmiGetHandler(env)  (*((int *)(env)))

#define CmiSetHandler(env,x)  (*((int *)(env)) = x)

#define CmiGetHandlerFunction(env)\
    (CpvAccess(CmiHandlerTable)[CmiGetHandler(env)])

void    *CmiAlloc  CMK_PROTO((int size));
int      CmiSize   CMK_PROTO((void *));
void     CmiFree   CMK_PROTO((void *));

double   CmiTimer  CMK_PROTO(());

#define CsdEnqueueGeneral(x,s,i,p)\
    (CqsEnqueueGeneral(CpvAccess(CsdSchedQueue),(x),(s),(i),(p)))
#define CsdEnqueueFifo(x)     (CqsEnqueueFifo(CpvAccess(CsdSchedQueue),(x)))
#define CsdEnqueueLifo(x)     (CqsEnqueueLifo(CpvAccess(CsdSchedQueue),(x)))
#define CsdEnqueue(x)         (CqsEnqueueFifo(CpvAccess(CsdSchedQueue),(x)))
#define CsdEmpty()            (CqsEmpty(CpvAccess(CsdSchedQueue)))

#ifdef CMK_CMIMYPE_IS_A_BUILTIN
int CmiMyPe CMK_PROTO((void));
int CmiNumPes CMK_PROTO((void));
#endif

#ifdef CMK_CMIMYPE_IS_A_VARIABLE
CpvExtern(int, Cmi_mype);
CpvExtern(int, Cmi_numpes);
#define CmiMyPe() CpvAccess(Cmi_mype)
#define CmiNumPes() CpvAccess(Cmi_numpes)
#endif

#ifdef CMK_CMIMYPE_UNIPROCESSOR
extern int Cmi_mype;
extern int Cmi_numpes;
#define CmiMyPe() Cmi_mype
#define CmiNumPes() Cmi_numpes
#endif

#ifdef CMK_CMIPRINTF_IS_A_BUILTIN
void  CmiPrintf CMK_PROTO((char *, ...));
void  CmiError  CMK_PROTO((char *, ...));
int   CmiScanf  CMK_PROTO((char *, ...));
#endif

#ifdef CMK_CMIPRINTF_IS_JUST_PRINTF
#define CmiPrintf printf
#define CmiError  printf
#define CmiScanf  scanf
#endif

/********* CSD - THE SCHEDULER ********/

extern  int CsdScheduler CMK_PROTO((int));

#ifdef CMK_CSDEXITSCHEDULER_IS_A_FUNCTION
extern void CsdExitScheduler CMK_PROTO((void));
#endif 

#ifdef CMK_CSDEXITSCHEDULER_SET_CSDSTOPFLAG
#define CsdExitScheduler()  (CpvAccess(CsdStopFlag)=1)
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
#define CmiASyncBroadcastAll(s,m)       (CmiAsyncBroadcastAllFn((s),(char *)(m)))
#define CmiSyncBroadcastAllAndFree(s,m) (CmiFreeBroadcastAllFn((s),(char *)(m)))

/******** CMI MESSAGE RECEPTION ********/

int    CmiDeliverMsgs          CMK_PROTO((int maxmsgs));
void   CmiDeliverSpecificMsg   CMK_PROTO((int handler));

/******** CQS: THE QUEUEING SYSTEM ********/

#define CQS_QUEUEING_FIFO 0
#define CQS_QUEUEING_LIFO 1
#define CQS_QUEUEING_IFIFO 2
#define CQS_QUEUEING_ILIFO 3
#define CQS_QUEUEING_BFIFO 4
#define CQS_QUEUEING_BLIFO 5

/****** CTH: THE THREADS PACKAGE ******/

#ifdef CMK_THREADS_USE_ALLOCA_WITH_HEADER_FILE
#define CMK_THREADS_USE_ALLOCA
#endif

#ifdef CMK_THREADS_USE_ALLOCA_WITH_PRAGMA
#define CMK_THREADS_USE_ALLOCA
#endif

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

/****** CTH: THREAD-PRIVATE VARIABLES (Geez, I hate C) ******/


#ifdef CMK_THREADS_UNAVAILABLE
#define CtvDeclare(t,v)         CpvDeclare(t,v)
#define CtvStaticDeclare(t,v)   CpvStaticDeclare(t,v)
#define CtvExtern(t,v)          CpvExtern(t,v)
#define CtvAccess(v)            CpvAccess(v)
#define CtvInitialize(t,v)      CpvInitialize(t,v)
#endif



#ifdef CMK_THREADS_USE_ALLOCA
#ifdef CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION
extern char *CthData;
extern int CthRegister CMK_PROTO((int));
#define CtvOffs(v) CtvOffs##v
#define CtvType(v) CtvType##v
#define CtvDeclare(t,v)         typedef t CtvType(v); int CtvOffs(v)
#define CtvStaticDeclare(t,v)   typedef t CtvType(v); static int CtvOffs(v)
#define CtvExtern(t,v)          typedef t CtvType(v); extern int CtvOffs(v)
#define CtvAccess(v)            (*((CtvType(v) *)(CthData+CtvOffs(v))))
#define CtvInitialize(t,v)      (CtvOffs(v)=CthRegister(sizeof(CtvType(v))))
#endif /* CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION */
#endif /* CMK_THREADS_USE_ALLOCA */



#ifdef CMK_THREADS_USE_ALLOCA
#ifdef CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION
extern char *CthData;
extern int CthRegister CMK_PROTO((int));
#define CtvDeclare(t,v)         int v;          struct v { t data; }
#define CtvStaticDeclare(t,v)   static int v;   struct v { t data; }
#define CtvExtern(t,v)          extern int v;   struct v { t data; }
#define CtvAccess(v)            (((struct v *)(CthData+(v)))->data)
#define CtvInitialize(t,v)      (v=CthRegister(sizeof(struct v)))
#endif /* CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION */
#endif /* CMK_THREADS_USE_ALLOCA */



#ifdef CMK_THREADS_USE_EATSTACK
#ifdef CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION
CpvExtern(char *, CthData);
extern int CthRegister CMK_PROTO((int));
#define CtvOffs(v) CtvOffs##v
#define CtvType(v) CtvType##v
#define CtvDeclare(t,v)      typedef t CtvType(v); int CtvOffs(v)
#define CtvStaticDeclare(t,v)typedef t CtvType(v); static int CtvOffs(v)
#define CtvExtern(t,v)       typedef t CtvType(v); extern int CtvOffs(v)
#define CtvAccess(v)         (*((CtvType(v) *)(CpvAccess(CthData)+CtvOffs(v))))
#define CtvInitialize(t,v)   (CtvOffs(v)=CthRegister(sizeof(CtvType(v))))
#endif /* CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION */
#endif /* CMK_THREADS_USE_EATSTACK */



#ifdef CMK_THREADS_USE_EATSTACK
#ifdef CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION
CpvDeclare(char *, CthData);
extern int CthRegister CMK_PROTO((int));
#define CtvDeclare(t,v)         int v;          struct v { t data; }
#define CtvStaticDeclare(t,v)   static int v;   struct v { t data; }
#define CtvExtern(t,v)          extern int v;   struct v { t data; }
#define CtvAccess(v)            (((struct v *)(CpvAccess(CthData)+(v)))->data)
#define CtvInitialize(t,v)      (v=CthRegister(sizeof(struct v)))
#endif /* CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION */
#endif /* CMK_THREADS_USE_EATSTACK */


/****** CMM: THE MESSAGE MANAGER ******/

typedef struct CmmTableStruct *CmmTable;

#define CmmWildCard (-1)

CmmTable   CmmNew();
void       CmmFree CMK_PROTO((CmmTable t));
void       CmmPut CMK_PROTO((CmmTable t, int ntags, int *tags, void *msg));
void      *CmmFind CMK_PROTO((CmmTable t, int ntags, int *tags, int *returntags, int del));
#define    CmmGet(t,nt,tg,rt)   (CmmFind((t),(nt),(tg),(rt),1))
#define    CmmProbe(t,nt,tg,rt) (CmmFind((t),(nt),(tg),(rt),0))


/****** FAST INTERRUPT BLOCKING FACILITY (NOT FOR CONVERSE USER) ********/

CpvExtern(int,       CmiInterruptsBlocked);
CpvExtern(CthVoidFn, CmiInterruptFuncSaved);

#define CmiInterruptHeader(fn) \
    if (CpvAccess(CmiInterruptsBlocked)) \
        { CpvAccess(CmiInterruptFuncSaved)=(CthVoidFn)(fn); return; }

#define CmiInterruptsBlock()\
    { CpvAccess(CmiInterruptsBlocked)++; }

#define CmiInterruptsRelease() { \
    CpvAccess(CmiInterruptsBlocked)--; \
    if (CpvAccess(CmiInterruptsBlocked)==0) {\
        CthVoidFn f = CpvAccess(CmiInterruptFuncSaved);\
        if (f) { CpvAccess(CmiInterruptFuncSaved)=0; (f)(); }\
    }\
}


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

