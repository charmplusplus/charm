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
 * Revision 2.10  1995-09-07 21:14:15  jyelon
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

#if defined(__cplusplus)||defined(__STDC__)
#define CMK_PROTO(x) x
#define CMK_CONCAT(x,y) x##y
#else
#define CMK_PROTO(x) ()
#define CMK_QUOTE(x) x
#define CMK_CONCAT(x,y) CMK_QUOTE(x)y
#endif


/******** MACROS AND PROTOTYPES FOR CPV AND CSV *******/

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

extern int Cmi_mype;
extern int Cmi_numpe;
#define CmiMyPe() Cmi_mype
#define CmiNumPe() Cmi_numpe

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










/******** PROTOTYPES FOR CONVERSE TYPES ********/

typedef void (*CmiHandler)();
typedef void  *CmiCommHandle;

CsvExtern(CmiHandler*, CmiHandlerTable);



/******** PROTOTYPES FOR CMI FUNCTIONS AND MACROS *******/


#define CmiMsgHeaderSizeBytes 4

extern int CmiRegisterHandler CMK_PROTO((CmiHandler));

#define CmiGetHandler(env)  (*((int *)(env)))

#define CmiSetHandler(env,x)  (*((int *)(env)) = x)

#define CmiGetHandlerFunction(env) (CsvAccess(CmiHandlerTable)[CmiGetHandler(env)])

void *CmiGetMsg CMK_PROTO(());

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

void *CmiAlloc  CMK_PROTO((int size));
int   CmiSize   CMK_PROTO(());
void  CmiFree   CMK_PROTO(());

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

/******** CONSTANTS FOR CONVERSE QUEUEING SYSTEM ********/

#define CQS_QUEUEING_FIFO 0
#define CQS_QUEUEING_LIFO 1
#define CQS_QUEUEING_IFIFO 2
#define CQS_QUEUEING_ILIFO 3
#define CQS_QUEUEING_BFIFO 4
#define CQS_QUEUEING_BLIFO 5

/******** PROTOTYPES FOR CSD FUNCTIONS AND MACROS ********/

CpvExtern(void*, CsdSchedQueue);
CpvExtern(int, CsdStopFlag);

/* for uniprocessor CsdExitScheduler() is a function in machine.c */
#ifndef CMK_SHARED_VARS_UNIPROCESSOR
#define CsdExitScheduler()  (CpvAccess(CsdStopFlag)=1)
#endif 

#define CsdEnqueueGeneral(x,s,i,p)\
    (CqsEnqueueGeneral(CpvAccess(CsdSchedQueue),x,s,i,p))
#define CsdEnqueueFifo(x)     (CqsEnqueueFifo(CpvAccess(CsdSchedQueue),x))
#define CsdEnqueueLifo(x)     (CqsEnqueueLifo(CpvAccess(CsdSchedQueue),x))
#define CsdEnqueue(x)         (CqsEnqueueFifo(CpvAccess(CsdSchedQueue),x))

#define CsdEmpty()     (CqsEmpty(CpvAccess(CsdSchedQueue)))

extern  void  CsdScheduler CMK_PROTO((int));
extern  void *CsdGetMsg CMK_PROTO(());

/**** DEAL WITH DIFFERENCES: KERNIGHAN-RITCHIE-C, ANSI-C, AND C++ ****/

#if defined(__cplusplus)
}
#endif

#endif /* CONVERSE_H */

