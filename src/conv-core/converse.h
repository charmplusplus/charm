
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
#else
#define CMK_PROTO(x) ()
#endif

/******** PROTOTYPES FOR CONVERSE TYPES ********/

typedef void (*CmiHandler)();
typedef void  *CmiCommHandle;


/******** MACROS AND PROTOTYPES FOR CPV AND CSV *******/

#ifdef CMK_NO_SHARED_VARS_AT_ALL

#define CpvDeclare(t,v) t v
#define CpvExtern(t,v)  extern t v
#define CpvStaticDeclare(t,v) static t v
#define CpvInitialize(t,v) 
#define CpvAccess(v) v

#define CsvDeclare(t,v) t v
#define CsvStaticDeclare(t,v) static t v
#define CsvInitialize(t,v) 
#define CsvExtern(t,v) extern t v
#define CsvAccess(v) v

#define CmiMyRank() 0
#define CmiNodeBarrier()
#define CmiSvAlloc CmiAlloc

#endif

/******** PROTOTYPES FOR CMI FUNCTIONS AND MACROS *******/


CsvExtern(CmiHandler*, CmiHandlerTable);

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
CpvExtern(int ,Cmi_numpe);
#define CmiMyPe() CpvAccess(Cmi_mype)
#define CmiNumPe() CpvAccess(Cmi_numpe)
#endif

void *CmiAlloc  CMK_PROTO((int size));
int   CmiSize   CMK_PROTO((...));
void  CmiFree   CMK_PROTO((...));

#ifdef CMK_CMIPRINTF_IS_A_BUILTIN
void  CmiPrintf CMK_PROTO((...));
void  CmiError  CMK_PROTO((...));
int   CmiScanf  CMK_PROTO((...));
#endif

#ifdef CMK_CMIPRINTF_IS_JUST_PRINTF
#define CmiPrintf printf
#define CmiError  printf
#define CmiScanf  scanf
#endif

/******** PROTOTYPES FOR CSD FUNCTIONS AND MACROS ********/

CpvExtern(void*, CsdSchedQueue);
CpvExtern(int, CsdStopFlag);

#define CsdExitScheduler()  (CpvAccess(CsdStopFlag)=1)

#define CsdEnqueue(x)  (CqsEnqueue(CpvAccess(CsdSchedQueue),x))

#define CsdEmpty()     (CqsEmpty(CpvAccess(CsdSchedQueue)))

extern  void  CsdScheduler CMK_PROTO((int));
extern  void *CsdGetMsg CMK_PROTO(());

/**** DEAL WITH DIFFERENCES: KERNIGHAN-RITCHIE-C, ANSI-C, AND C++ ****/

#if defined(__cplusplus)
}
#endif

#endif  /* CONVERSE_H */


