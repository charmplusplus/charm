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
 * Revision 2.21  1995-09-27 22:23:15  jyelon
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

#if defined(__cplusplus)
extern "C" {
#endif

#ifdef CMK_COMPILER_LIKES_PROTOTYPES
#define CMK_PROTO(x) x
#endif

#ifdef CMK_COMPILER_HATES_PROTOTYPES
#define CMK_PROTO(x) ()
#endif

#ifdef CMK_PREPROCESSOR_USES_K_AND_R_STANDARD_CONCATENATION
#define CMK_QUOTE(x)x
#define CMK_CONCAT(x,y) CMK_QUOTE(x)y
#endif

#ifdef CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION
#define CMK_CONCAT(x,y) x##y
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

#endif



#ifdef CMK_SHARED_VARS_UNIPROCESSOR

#define SHARED_DECL
#define CpvDeclare(t,v) t* CMK_CONCAT(Cpv_Var_,v)
#define CpvExtern(t,v)  extern t* CMK_CONCAT(Cpv_Var_,v)
#define CpvStaticDeclare(t,v) static t* CMK_CONCAT(Cpv_Var_,v)
#define CpvInitialize(t,v) if (Cmi_mype == 0) CMK_CONCAT(Cpv_Var_,v) = (t *) CmiAlloc(Cmi_numpe*sizeof(t)); else;
#define CpvAccess(v) CMK_CONCAT(Cpv_Var_,v)[Cmi_mype]

#define CsvDeclare(t,v) t CMK_CONCAT(Csv_Var_,v)
#define CsvStaticDeclare(t,v) static t CMK_CONCAT(Csv_Var_,v)
#define CsvExtern(t,v) extern t CMK_CONCAT(Csv_Var_,v)
#define CsvInitialize(t,v)
#define CsvAccess(v) CMK_CONCAT(Csv_Var_,v)

#define CmiMyRank() Cmi_mype
#define CmiNodeBarrier()
#define CmiSvAlloc CmiAlloc

#endif




/******** CMI, CSD: MANY LOW-LEVEL OPERATIONS ********/

#define CmiMsgHeaderSizeBytes 4

typedef void (*CmiHandler)();
typedef void  *CmiCommHandle;

CsvExtern(CmiHandler*, CmiHandlerTable);
CpvExtern(void*,       CsdSchedQueue);
CpvExtern(int,         CsdStopFlag);

extern int CmiRegisterHandler CMK_PROTO((CmiHandler));

#define CmiGetHandler(env)  (*((int *)(env)))

#define CmiSetHandler(env,x)  (*((int *)(env)) = x)

#define CmiGetHandlerFunction(env)\
    (CsvAccess(CmiHandlerTable)[CmiGetHandler(env)])

void    *CmiGetMsg CMK_PROTO(());

void    *CmiAlloc  CMK_PROTO((int size));
int      CmiSize   CMK_PROTO(());
void     CmiFree   CMK_PROTO((void *));

double   CmiTimer  CMK_PROTO(());

int      CmiSpanTreeRoot         CMK_PROTO(()) ;
int      CmiNumSpanTreeChildren  CMK_PROTO((int)) ;
int      CmiSpanTreeParent       CMK_PROTO((int)) ;
void     CmiSpanTreeChildren     CMK_PROTO((int node, int *children)) ;


#define CsdEnqueueGeneral(x,s,i,p)\
    (CqsEnqueueGeneral(CpvAccess(CsdSchedQueue),x,s,i,p))
#define CsdEnqueueFifo(x)     (CqsEnqueueFifo(CpvAccess(CsdSchedQueue),x))
#define CsdEnqueueLifo(x)     (CqsEnqueueLifo(CpvAccess(CsdSchedQueue),x))
#define CsdEnqueue(x)         (CqsEnqueueFifo(CpvAccess(CsdSchedQueue),x))
#define CsdEmpty()            (CqsEmpty(CpvAccess(CsdSchedQueue)))

extern  void  CsdScheduler CMK_PROTO((int));
extern  void *CsdGetMsg    CMK_PROTO(());

/* for uniprocessor CsdExitScheduler() is a function in machine.c */
#ifndef CMK_SHARED_VARS_UNIPROCESSOR
#define CsdExitScheduler()  (CpvAccess(CsdStopFlag)=1)
#endif 

#ifdef CMK_CMIMYPE_IS_A_BUILTIN
int CmiMyPe CMK_PROTO((void));
int CmiNumPe CMK_PROTO((void));
#endif

#ifdef CMK_CMIMYPE_IS_A_VARIABLE
CpvExtern(int, Cmi_mype);
CpvExtern(int, Cmi_numpe);
#define CmiMyPe() CpvAccess(Cmi_mype)
#define CmiNumPe() CpvAccess(Cmi_numpe)
#endif

#ifdef CMK_CMIMYPE_UNIPROCESSOR
extern int Cmi_mype;
extern int Cmi_numpe;
#define CmiMyPe() Cmi_mype
#define CmiNumPe() Cmi_numpe
#endif

#ifdef CMK_CMIPRINTF_IS_A_BUILTIN
void  CmiPrintf CMK_PROTO(());
void  CmiError  CMK_PROTO(());
int   CmiScanf  CMK_PROTO(());
#endif

#ifdef CMK_CMIPRINTF_IS_JUST_PRINTF
#define CmiPrintf printf
#define CmiError  printf
#define CmiScanf  scanf
#endif

/******** CQS: THE QUEUEING SYSTEM ********/

#define CQS_QUEUEING_FIFO 0
#define CQS_QUEUEING_LIFO 1
#define CQS_QUEUEING_IFIFO 2
#define CQS_QUEUEING_ILIFO 3
#define CQS_QUEUEING_BFIFO 4
#define CQS_QUEUEING_BLIFO 5

/****** CTH: THE THREADS PACKAGE ******/

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

void       CthSetVar   CMK_PROTO((CthThread, void **, void *));
void      *CthGetVar   CMK_PROTO((CthThread, void **));


/****** CMM: THE MESSAGE MANAGER ******/

typedef struct CmmTableStruct *CmmTable;

#define CmmWildCard (-1)

CmmTable   CmmNew();
void       CmmFree(CmmTable t);
void       CmmPut(CmmTable t, int ntags, int *tags, void *msg);
void      *CmmFind(CmmTable t, int ntags, int *tags, int *returntags, int del);
#define    CmmGet(t,nt,tg,rt)   (CmmFind(t,nt,tg,rt,1))
#define    CmmProbe(t,nt,tg,rt) (CmmFind(t,nt,tg,rt,0))


/**** DEAL WITH DIFFERENCES: KERNIGHAN-RITCHIE-C, ANSI-C, AND C++ ****/

#if defined(__cplusplus)
}
#endif

#endif /* CONVERSE_H */

