/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkPerf
*/
/*@{*/

#include <stdlib.h>

// cannot include charm++.h because trace-common.o is part of libconv-core.a
#include "charm.h"
#include "middle.h"
#include "cklists.h"

#include "trace.h"
#include "trace-common.h"
#include "allEvents.h"          //projector
#include "register.h" // for _entryTable

CpvCExtern(int, _traceCoreOn);   // projector

#define DEBUGF(x)          // CmiPrintf x

#define LogBufSize      10000

#ifdef CMK_OPTIMIZE
static int warned = 0;
#define OPTIMIZE_WARNING if (!warned) { warned=1;  CmiPrintf("\n\n!!!! Warning: tracing not available with CMK_OPTIMIZE!\n");  return;  }
#else
#define OPTIMIZE_WARNING /*empty*/
#endif

CkpvDeclare(TraceArray*, _traces);		// lists of all trace modules

/* trace for bluegene */
class TraceBluegene;
CkpvDeclare(TraceBluegene*, _tracebg);
int traceBluegeneLinked=0;			// if trace-bluegene is linked

CkpvDeclare(double, traceInitTime);
CpvDeclare(int, traceOn);
#if CMK_TRACE_IN_CHARM
CkpvDeclare(int, traceOnPe);
#endif

CkpvDeclare(int, CtrLogBufSize);
CkpvDeclare(char*, traceRoot);

int _threadMsg, _threadChare, _threadEP;
int _packMsg, _packChare, _packEP;
int _unpackMsg, _unpackChare, _unpackEP;
int _dummyMsg, _dummyChare, _dummyEP;

/// decide parameters from command line
static void traceCommonInit(char **argv)
{
  CmiArgGroup("Charm++","Tracing");
  DEBUGF(("[%d] in traceCommonInit.\n", CkMyPe()));
  CkpvInitialize(double, traceInitTime);
  CkpvAccess(traceInitTime) = TRACE_TIMER();
  CpvInitialize(int, traceOn);
  CpvInitialize(int, _traceCoreOn); //projector
  CkpvInitialize(int, CtrLogBufSize);
  CkpvInitialize(char*, traceRoot);
  CpvAccess(traceOn) = 0;
  CpvAccess(_traceCoreOn)=0; //projector
#if CMK_TRACE_IN_CHARM
  CkpvInitialize(int, traceOnPe);
  CkpvAccess(traceOnPe) = 1;
#if 0
  // example for choosing subset of processors for tracing
  CkpvAccess(traceOnPe) = 0;
  if (CkMyPe() < 20 && CkMyPe() > 15) CkpvAccess(traceOnPe) = 1;
  // must include pe 0
  if (CkMyPe()==0) CkpvAccess(traceOnPe) = 1;
#endif
#endif
  CkpvAccess(CtrLogBufSize) = LogBufSize;
  if (CmiGetArgIntDesc(argv,"+logsize",&CkpvAccess(CtrLogBufSize), "Log entries to buffer per I/O"))
    if (CkMyPe() == 0) 
      CmiPrintf("Trace: logsize: %d\n", CkpvAccess(CtrLogBufSize));
  char *root;
  if (CmiGetArgStringDesc(argv, "+traceroot", &root, "Directory to write trace files to")) {
    int i;
    for (i=strlen(argv[0])-1; i>=0; i--) if (argv[0][i] == '/') break;
    i++;
    CkpvAccess(traceRoot) = (char *)malloc(strlen(argv[0]+i) + strlen(root) + 2);    _MEMCHECK(CkpvAccess(traceRoot));
    strcpy(CkpvAccess(traceRoot), root);
    strcat(CkpvAccess(traceRoot), "/");
    strcat(CkpvAccess(traceRoot), argv[0]+i);
    if (CkMyPe() == 0) 
      CmiPrintf("Trace: traceroot: %s\n", CkpvAccess(traceRoot));
  }
  else {
    CkpvAccess(traceRoot) = (char *) malloc(strlen(argv[0])+1);
    _MEMCHECK(CkpvAccess(traceRoot));
    strcpy(CkpvAccess(traceRoot), argv[0]);
  }
  
#ifdef __BLUEGENE__
  if(BgNodeRank()==0) {
#else
  if(CkMyRank()==0) {
#endif
    _threadMsg = CkRegisterMsg("dummy_thread_msg", 0, 0, 0);
    _threadChare = CkRegisterChare("dummy_thread_chare", 0);
    _threadEP = CkRegisterEp("dummy_thread_ep", 0, _threadMsg,_threadChare, 0+CK_EP_INTRINSIC);

    _packMsg = CkRegisterMsg("dummy_pack_msg", 0, 0, 0);
    _packChare = CkRegisterChare("dummy_pack_chare", 0);
    _packEP = CkRegisterEp("dummy_pack_ep", 0, _packMsg,_packChare, 0+CK_EP_INTRINSIC);

    _unpackMsg = CkRegisterMsg("dummy_unpack_msg", 0, 0, 0);
    _unpackChare = CkRegisterChare("dummy_unpack_chare", 0);
    _unpackEP = CkRegisterEp("dummy_unpack_ep", 0, _unpackMsg,_unpackChare, 0+CK_EP_INTRINSIC);

    _dummyMsg = CkRegisterMsg("dummy_msg", 0, 0, 0);
    _dummyChare = CkRegisterChare("dummy_chare", 0);
    _dummyEP = CkRegisterEp("dummy_ep", 0, _dummyMsg,_dummyChare, 0+CK_EP_INTRINSIC);
  }
}

/** Write out the common parts of the .sts file. */
extern void traceWriteSTS(FILE *stsfp,int nUserEvents) {
  fprintf(stsfp, "MACHINE %s\n",CMK_MACHINE_NAME);
  fprintf(stsfp, "PROCESSORS %d\n", CkNumPes());
  fprintf(stsfp, "TOTAL_CHARES %d\n", _chareTable.size());
  fprintf(stsfp, "TOTAL_EPS %d\n", _entryTable.size());
  fprintf(stsfp, "TOTAL_MSGS %d\n", _msgTable.size());
  fprintf(stsfp, "TOTAL_PSEUDOS %d\n", 0);
  fprintf(stsfp, "TOTAL_EVENTS %d\n", nUserEvents);
  int i;
  for(i=0;i<_chareTable.size();i++)
    fprintf(stsfp, "CHARE %d %s\n", i, _chareTable[i]->name);
  for(i=0;i<_entryTable.size();i++)
    fprintf(stsfp, "ENTRY CHARE %d %s %d %d\n", i, _entryTable[i]->name,
                 _entryTable[i]->chareIdx, _entryTable[i]->msgIdx);
  for(i=0;i<_msgTable.size();i++)
    fprintf(stsfp, "MESSAGE %d %d\n", i, _msgTable[i]->size);
}

/*Install the beginIdle/endIdle condition handlers.*/
extern "C" void traceBegin(void) {
  OPTIMIZE_WARNING
  DEBUGF(("[%d] traceBegin called with %d at %f\n", CkMyPe(), CpvAccess(traceOn), TraceTimer()));
  if (CpvAccess(traceOn)==1) return;
  CkpvAccess(_traces)->traceBegin();
  CpvAccess(traceOn) = 1;
}

/*Cancel the beginIdle/endIdle condition handlers.*/
extern "C" void traceEnd(void) {
  OPTIMIZE_WARNING
  DEBUGF(("[%d] traceEnd called with %d at %f\n", CkMyPe(), CpvAccess(traceOn), TraceTimer()));
  if (CpvAccess(traceOn)==0) return;
  CkpvAccess(_traces)->traceEnd();
  CpvAccess(traceOn) = 0;
}

/// defined in moduleInit.C
void _createTraces(char **argv);

/**
    traceInit: 		called at Converse level
    traceCharmInit:	called at Charm++ level
*/
/// initialize trace framework, also create the trace module(s).
static inline void _traceInit(char **argv) 
{
  CkpvInitialize(TraceArray *, _traces);
  CkpvAccess(_traces) = new TraceArray;

  // common init
  traceCommonInit(argv);

  // in moduleInit.C
  _createTraces(argv);

  if (CkpvAccess(_traces)->length() && !CmiGetArgFlagDesc(argv,"+traceoff","Disable tracing"))
    traceBegin();
}

/// Converse version
extern "C" void traceInit(char **argv) 
{
#if ! CMK_TRACE_IN_CHARM
  _traceInit(argv);
#endif
  initTraceCore(argv);
}

/// Charm++ version
extern "C" void traceCharmInit(char **argv) 
{
#if CMK_TRACE_IN_CHARM
  _traceInit(argv);
#endif
}

// CMK_OPTIMIZE is already guarded in convcore.c
extern "C"
void traceMessageRecv(char *msg, int pe)
{
#if ! CMK_TRACE_IN_CHARM
  CkpvAccessOther(_traces, CmiRankOf(pe))->messageRecv(msg, pe);
#endif
}

// CMK_OPTIMIZE is already guarded in convcore.c
// converse thread tracing is not supported in blue gene simulator
// only AMPI threads are traced.
extern "C"
void traceResume(CmiObjId *tid)
{
    _TRACE_ONLY(CkpvAccess(_traces)->beginExecute(tid));
    if(CpvAccess(_traceCoreOn))
	    resumeTraceCore();
}

extern "C"
void traceSuspend(void)
{
  _TRACE_ONLY(CkpvAccess(_traces)->endExecute());
}

extern "C"
void traceAwaken(CthThread t)
{
  CkpvAccess(_traces)->creation(0, _threadEP);
}

extern "C"
void traceUserEvent(int e)
{
#ifndef CMK_OPTIMIZE
  if (CpvAccess(traceOn))
    CkpvAccess(_traces)->userEvent(e);
#endif
}

extern "C"
void traceUserBracketEvent(int e, double beginT, double endT)
{
#ifndef CMK_OPTIMIZE
  if (CpvAccess(traceOn) && CkpvAccess(_traces))
    CkpvAccess(_traces)->userBracketEvent(e, beginT, endT);
#endif
}

extern "C"
int traceRegisterUserEvent(const char*x, int e)
{
#ifndef CMK_OPTIMIZE
  return CkpvAccess(_traces)->traceRegisterUserEvent(x, e);
#else
  return 0;
#endif
}

extern "C"
void traceClearEps(void)
{
  OPTIMIZE_WARNING
  CkpvAccess(_traces)->traceClearEps();
}

extern "C"
void traceWriteSts(void)
{
  OPTIMIZE_WARNING
  CkpvAccess(_traces)->traceWriteSts();
}

/**
    traceClose: 	this function is called at Converse
    traceCharmClose:	called at Charm++ level
*/
extern "C"
void traceClose(void)
{
#if ! CMK_BLUEGENE_CHARM
  OPTIMIZE_WARNING
  CkpvAccess(_traces)->traceClose();
#endif   
}

extern "C"
void traceCharmClose(void)
{
#if CMK_BLUEGENE_CHARM
  OPTIMIZE_WARNING
  CkpvAccess(_traces)->traceClose();
#endif
}

#if 0
// helper functions
int CkIsCharmMessage(char *msg)
{
//CmiPrintf("getMsgtype: %d %d %d %d %d\n", ((envelope *)msg)->getMsgtype(), CmiGetHandler(msg), CmiGetXHandler(msg), _charmHandlerIdx, index_skipCldHandler);
  if ((CmiGetHandler(msg) == _charmHandlerIdx) &&
         (CmiGetHandlerFunction(msg) == (CmiHandler)_processHandler))
    return 1;
  if (CmiGetXHandler(msg) == _charmHandlerIdx) return 1;
  return 0;
}
#endif

// return 1 if any one of tracing modules is linked.
int  traceAvailable()
{
#ifdef CMK_OPTIMIZE
  return 0;
#else
  return CkpvAccess(_traces)->length()>0;
#endif
}

double CmiTraceTimer()
{
  return TraceTimer();
}

void TraceArray::creation(envelope *env, int ep, int num)
{ 
    if (_entryTable[ep]->traceEnabled)
        ALLDO(creation(env, ep, num));
}

void TraceArray::creationMulticast(envelope *env, int ep, int num,
				   int *pelist)
{
  if (_entryTable[ep]->traceEnabled)
    ALLDO(creationMulticast(env, ep, num, pelist));
}

/*@}*/
