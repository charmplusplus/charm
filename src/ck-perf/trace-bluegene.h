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

#ifndef _TRACE_BLUEGENE_H
#define _TRACE_BLUEGENE_H

#include "trace.h"

// TraceBluegene is subclass of Trace, 
// it defines Blue Gene specific tracing subroutines.
class TraceBluegene : public Trace {

 private:
    FILE* pfp;
 public:
    TraceBluegene(char** argv);
    ~TraceBluegene();
    virtual void setTraceOnPE(int flag) { _traceOn = 1; }  // always on
    int traceOnPE() { return 1; }
    void getForwardDep(void* log, void** fDepPtr);
    void getForwardDepForAll(void** logs1, void** logs2, int logsize,void* fDepPtr);
    void tlineEnd(void** parentLogPtr);
    void bgDummyBeginExec(char* name,void** parentLogPtr);
    void bgBeginExec(char* msg, char *str);
    void bgAmpiBeginExec(char *msg, char *str, void *log);
    void bgEndExec(int);
    void addBackwardDep(void *log);
    void userBracketEvent(int eventID, double bt, double et) {}	// from trace.h
    void userBracketEvent(char* name, double bt, double et, void** parentLogPtr);
    void userBracketEvent(char* name, double bt, double et, void** parentLogPtr, CkVec<void*> bgLogList);
    void bgPrint(char* str);
    void creatFiles();
    void writePrint(char *, double t);
    void traceClose();
};

CkpvExtern(TraceBluegene*, _tracebg);
extern int traceBluegeneLinked;

#ifndef CMK_OPTIMIZE
#  define _TRACE_BG_ONLY(code) do{if(traceBluegeneLinked && CpvAccess(traceOn)){ code; }} while(0)
#else
#  define _TRACE_BG_ONLY(code) /*empty*/
#endif

/* tracing for Blue Gene - before trace projector era */
#if !defined(CMK_OPTIMIZE) && CMK_TRACE_IN_CHARM
// for Sdag only
// fixme - think of better api for tracing sdag code
#define BgPrint(x)  _TRACE_BG_ONLY(CkpvAccess(_tracebg)->bgPrint(x))
#define _TRACE_BG_BEGIN_EXECUTE_NOMSG(x,pLogPtr)  _TRACE_BG_ONLY(CkpvAccess(_tracebg)->bgDummyBeginExec(x,pLogPtr))
#define _TRACE_BG_BEGIN_EXECUTE(msg, str)  _TRACE_BG_ONLY(CkpvAccess(_tracebg)->bgBeginExec(msg, str))
#define _TRACE_BG_AMPI_BEGIN_EXECUTE(msg, str, log)  _TRACE_BG_ONLY(CkpvAccess(_tracebg)->bgAmpiBeginExec(msg, str, log))
#define _TRACE_BG_END_EXECUTE(commit)   _TRACE_BG_ONLY(CkpvAccess(_tracebg)->bgEndExec(commit))
#define _TRACE_BG_TLINE_END(pLogPtr) _TRACE_BG_ONLY(CkpvAccess(_tracebg)->tlineEnd(pLogPtr))
#define _TRACE_BG_FORWARD_DEPS(logs1,logs2,size,fDep)  _TRACE_BG_ONLY(CkpvAccess(_tracebg)->getForwardDepForAll(logs1,logs2, size,fDep))
#define _TRACE_BG_ADD_BACKWARD_DEP(log)  _TRACE_BG_ONLY(CkpvAccess(_tracebg)->addBackwardDep(log))
#define _TRACE_BG_USER_EVENT_BRACKET(x,bt,et,pLogPtr) _TRACE_BG_ONLY(CkpvAccess(_tracebg)->userBracketEvent(x,bt,et,pLogPtr))
#define _TRACE_BGLIST_USER_EVENT_BRACKET(x,bt,et,pLogPtr,bgLogList) _TRACE_BG_ONLY(CkpvAccess(_tracebg)->userBracketEvent(x,bt,et,pLogPtr,bgLogList))

# define TRACE_BG_AMPI_SUSPEND()     \
        if(CpvAccess(traceOn)) traceSuspend();  \
        _TRACE_BG_END_EXECUTE(1);
# define TRACE_BG_AMPI_RESUME(t, msg, str, log)        \
	/* using the actual received message's time */	\
        _TRACE_BG_AMPI_BEGIN_EXECUTE((char *)UsrToEnv(msg), str, log); \
        if(CpvAccess(traceOn)) CthTraceResume(t);
# define TRACE_BG_AMPI_START(t, str)  { \
        void* _bgParentLog = NULL;      \
        /*_TRACE_BG_TLINE_END(&_bgParentLog);*/	\
        _TRACE_BG_BEGIN_EXECUTE_NOMSG(str, &_bgParentLog);      \
        if(CpvAccess(traceOn) && t) CthTraceResume(t);	\
        }
# define TRACE_BG_AMPI_NEWSTART(t, str, event, count)	\
	TRACE_BG_AMPI_SUSPEND();	\
	TRACE_BG_AMPI_START(t, str);	\
    	{	\
	for(int i=0;i<count;i++) {	\
                _TRACE_BG_ADD_BACKWARD_DEP(event);	\
        }	\
	}
#define TRACE_BG_AMPI_WAITALL(reqs) 	\
        {	\
	/* TRACE_BG_AMPI_SUSPEND(); */	\
	CthThread th = getAmpiInstance(MPI_COMM_WORLD)->getThread();	\
  	TRACE_BG_AMPI_START(th, "AMPI_WAITALL")	\
    	_TRACE_BG_ADD_BACKWARD_DEP(curLog);	\
  	for(int i=0;i<count;i++) {	\
	  if (request[i] == MPI_REQUEST_NULL) continue;	\
    	  void *log = (*reqs)[request[i]]->event;	\
    	  _TRACE_BG_ADD_BACKWARD_DEP(log);	\
        }	\
	}
#define TRACE_BG_AMPI_BARRIER_START(barrierLog)	\
	{	\
  	_TRACE_BG_TLINE_END(&barrierLog);	\
  	TRACE_BG_AMPI_SUSPEND();	\
  	_TRACE_BG_BEGIN_EXECUTE_NOMSG("AMPI_Barrier", &barrierLog);	\
	}
#define TRACE_BG_AMPI_BARRIER_END(barrierLog)	\
	{	\
	void *curLog;    /* store current log in timeline */	\
  	_TRACE_BG_TLINE_END(&curLog);	\
  	TRACE_BG_AMPI_SUSPEND();	\
  	_TRACE_BG_BEGIN_EXECUTE_NOMSG("AMPI_Barrier_END", &curLog);	\
  	_TRACE_BG_ADD_BACKWARD_DEP(barrierLog);	\
	}
extern "C" void BgSetStartEvent();
#else
# define BgPrint(x)  
# define _TRACE_BG_TLINE_END(x)
#define _TRACE_BG_BEGIN_EXECUTE_NOMSG(x,pLogPtr)
#define _TRACE_BG_USER_EVENT_BRACKET(x,bt,et,pLogPtr)
#define _TRACE_BGLIST_USER_EVENT_BRACKET(x,bt,et,pLogPtr,bgLogList)
#define _TRACE_BG_END_EXECUTE(commit)
#define _TRACE_BG_FORWARD_DEPS(logs1,logs2,size,fDep)

# define TRACE_BG_AMPI_SUSPEND()
# define TRACE_BG_AMPI_RESUME(t, msg, str, log)
# define TRACE_BG_AMPI_START(t, str)
# define TRACE_BG_NEWSTART(t, str, events, count)
#define TRACE_BG_AMPI_WAITALL(reqs)
#endif   /* CMK_TRACE_IN_CHARM */
extern "C" void BgPrintf(char *str);

#endif

/*@}*/
