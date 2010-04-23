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

// Bigsim emulator specific tracing subroutines.
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
    void bgAddTag(const char *str);
    void bgDummyBeginExec(const char* name,void** parentLogPtr, int split);
    void bgBeginExec(char* msg, char *str);
    void bgAmpiBeginExec(char *msg, char *str, void **logs, int count);
    void bgAmpiSetSize(int size);
    void bgSetInfo(char *msg, const char *str, void **logs, int count);
    void bgEndExec(int);
    virtual void beginExecute(envelope *);
    virtual void beginExecute(CmiObjId *tid) {}
    virtual void beginExecute(int event,int msgType,int ep,int srcPe, int mlen,CmiObjId *idx);
    void addBackwardDep(void *log);
    void userBracketEvent(int eventID, double bt, double et) {}	// from trace.h
    void userBracketEvent(const char* name, double bt, double et, void** parentLogPtr);
    void userBracketEvent(const char* name, double bt, double et, void** parentLogPtr, CkVec<void*> bgLogList);
    void bgPrint(const char* str);
    void bgMark(char* str);
    void creatFiles();
    void writePrint(char *, double t);
    void traceClose();
};

CkpvExtern(TraceBluegene*, _tracebg);
extern int traceBluegeneLinked;

#if ! CMK_TRACE_DISABLED
#  define _TRACE_BG_ONLY(code) do{ BgGetTime(); if(traceBluegeneLinked && CpvAccess(traceOn)){ code; } resetVTime(); } while(0)
#else
#  define _TRACE_BG_ONLY(code) /*empty*/
#endif

/* tracing for Blue Gene - before trace projector era */
#if !CMK_TRACE_DISABLED && CMK_TRACE_IN_CHARM
// for Sdag only
// fixme - think of better api for tracing sdag code
#define BgPrint(x)  _TRACE_BG_ONLY(CkpvAccess(_tracebg)->bgPrint(x))
#define BgMark_(x)  _TRACE_BG_ONLY(CkpvAccess(_tracebg)->bgMark(x))
#define _TRACE_BG_BEGIN_EXECUTE_NOMSG(x,pLogPtr,split)  _TRACE_BG_ONLY(CkpvAccess(_tracebg)->bgDummyBeginExec(x,pLogPtr,split))
#define _TRACE_BG_BEGIN_EXECUTE(msg, str)  _TRACE_BG_ONLY(CkpvAccess(_tracebg)->bgBeginExec(msg, str))
#define _TRACE_BG_SET_INFO(msg, str, logs, count)  _TRACE_BG_ONLY(CkpvAccess(_tracebg)->bgSetInfo(msg, str, logs, count))
#define _TRACE_BG_AMPI_BEGIN_EXECUTE(msg, str, logs, count)  _TRACE_BG_ONLY(CkpvAccess(_tracebg)->bgAmpiBeginExec(msg, str, logs, count))
#define _TRACE_BG_END_EXECUTE(commit)   _TRACE_BG_ONLY(CkpvAccess(_tracebg)->bgEndExec(commit))
#define _TRACE_BG_TLINE_END(pLogPtr) _TRACE_BG_ONLY(CkpvAccess(_tracebg)->tlineEnd(pLogPtr))
#define _TRACE_BG_FORWARD_DEPS(logs1,logs2,size,fDep)  _TRACE_BG_ONLY(CkpvAccess(_tracebg)->getForwardDepForAll(logs1,logs2, size,fDep))
#define _TRACE_BG_ADD_BACKWARD_DEP(log)  _TRACE_BG_ONLY(CkpvAccess(_tracebg)->addBackwardDep(log))
#define _TRACE_BG_USER_EVENT_BRACKET(x,bt,et,pLogPtr) _TRACE_BG_ONLY(CkpvAccess(_tracebg)->userBracketEvent(x,bt,et,pLogPtr))
#define _TRACE_BGLIST_USER_EVENT_BRACKET(x,bt,et,pLogPtr,bgLogList) _TRACE_BG_ONLY(CkpvAccess(_tracebg)->userBracketEvent(x,bt,et,pLogPtr,bgLogList))
#define TRACE_BG_ADD_TAG(str)	_TRACE_BG_ONLY(CkpvAccess(_tracebg)->bgAddTag(str))

# define TRACE_BG_AMPI_SET_SIZE(size) \
    _TRACE_BG_ONLY(CkpvAccess(_tracebg)->bgAmpiSetSize(size))

# define TRACE_BG_AMPI_SUSPEND()     \
	_TRACE_BG_END_EXECUTE(1); \
        /* if(CpvAccess(traceOn)) traceSuspend(); */

# define TRACE_BG_AMPI_START(t, str)  { \
        void* _bgParentLog = NULL;      \
        /*_TRACE_BG_TLINE_END(&_bgParentLog);*/ \
        if(CpvAccess(traceOn) && t) CthTraceResume(t);  \
        _TRACE_BG_BEGIN_EXECUTE_NOMSG(str, &_bgParentLog, 1);      \
        }

# define TRACE_BG_AMPI_BREAK(t, str, event, count, connect)  	\
	{	\
	void *curLog;    /* store current log in timeline */	\
  	_TRACE_BG_TLINE_END(&curLog);	\
	TRACE_BG_AMPI_SUSPEND();        \
        /* TRACE_BG_AMPI_START(t, str);  */  \
	void * _bgParentLog = NULL;      \
         _TRACE_BG_BEGIN_EXECUTE_NOMSG(str, &_bgParentLog, 1);  \
        for(int i=0;i<count;i++) {      \
                _TRACE_BG_ADD_BACKWARD_DEP(((void**)event)[i]);      \
        }	\
        if (connect) _TRACE_BG_ADD_BACKWARD_DEP(curLog);      \
	}
	

#define TRACE_BG_AMPI_WAITALL(reqs) 	\
        {	\
	/* TRACE_BG_AMPI_SUSPEND(); */	\
	CthThread th = getAmpiInstance(MPI_COMM_WORLD)->getThread();	\
 	TRACE_BG_AMPI_BREAK(th, "AMPI_WAITALL", NULL, 0, 0);	\
    	_TRACE_BG_ADD_BACKWARD_DEP(curLog);	\
  	for(int i=0;i<count;i++) {	\
	  if (request[i] == MPI_REQUEST_NULL) continue;	\
    	  void *log = (*reqs)[request[i]]->event;	\
    	  _TRACE_BG_ADD_BACKWARD_DEP(log);	\
        }	\
	}
extern "C" void BgSetStartEvent();
#else
# define BgPrint(x)  
# define BgMark_(x)  
#define _TRACE_BG_BEGIN_EXECUTE_NOMSG(x,pLogPtr,split)
#define _TRACE_BG_BEGIN_EXECUTE(msg, str)
#define _TRACE_BG_SET_INFO(msg, str, logs, count) 
#define _TRACE_BG_AMPI_BEGIN_EXECUTE(msg, str, logs, count)
#define _TRACE_BG_END_EXECUTE(commit)
#define _TRACE_BG_TLINE_END(x)	
#define _TRACE_BG_FORWARD_DEP(logs1,logs2,size,fDep)
#define _TRACE_BG_BACKWARD_DEP(log)	
#define _TRACE_BG_USER_EVENT_BRACKET(x,bt,et,pLogPtr)
#define _TRACE_BGLIST_USER_EVENT_BRACKET(x,bt,et,pLogPtr,bgLogList)
#define _TRACE_BG_ADD_TAG(str)
	
# define TRACE_BG_AMPI_SUSPEND()
# define TRACE_BG_AMPI_RESUME(t, msg, str, log)
# define TRACE_BG_AMPI_START(t, str)
# define TRACE_BG_NEWSTART(t, str, events, count)
# define TRACE_BG_AMPI_BREAK(t, str, event, count)
# define TRACE_BG_AMPI_WAITALL(reqs)
# define TRACE_BG_AMPI_SET_SIZE(size)
#endif   /* CMK_TRACE_IN_CHARM */

extern "C" void BgPrintf(const char *str);
extern "C" void BgMark(char *str);

#endif

/*@}*/
