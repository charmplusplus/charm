/***********Projector tracing file added by Sayantan ****************/

#include "charm++.h"
#include "trace-projector.h"
#include "trace-projections.h"

#define DEBUGF(x)           // CmiPrintf x

CkpvStaticDeclare(Trace*, _traceproj);
class UsrEvent {
public:
  int e;
  char *str;
  UsrEvent(int _e, char* _s): e(_e),str(_s) {}
};
typedef CkVec<UsrEvent *>   UsrEventVec;
CkpvStaticDeclare(UsrEventVec, usrEvents);


#if ! CMK_TRACE_ENABLED
static int warned=0;
#define OPTIMIZED_VERSION 	\
	if (!warned) { warned=1; 	\
	CmiPrintf("\n\n!!!! Warning: traceUserEvent not available in optimized version!!!!\n\n\n"); }
#else
#define OPTIMIZED_VERSION /*empty*/
#endif

/**
  For each TraceFoo module, _createTraceFoo() must be defined.
  This function is called in _createTraces() generated in moduleInit.C
*/
void _createTraceprojector(char **argv)
{
  DEBUGF(("%d createTraceProjector\n", CkMyPe()));
  CkpvInitialize(Trace*, _traceproj);
  CkpvInitialize(CkVec<UsrEvent *>, usrEvents);
  CkpvAccess(_traceproj) = new  TraceProjector(argv);
  CkpvAccess(_traces)->addTrace(CkpvAccess(_traceproj));
}

TraceProjector::TraceProjector(char **argv)
{
	
		CpvInitialize(int, _traceCoreOn);
		CpvAccess(_traceCoreOn)=1;
		traceCoreOn = 1;

}

int TraceProjector::traceRegisterUserEvent(const char* evt, int e)
{
CkAssert(e==-1 || e>=0);
  CkAssert(evt != NULL);
  int event;
  int biggest = -1;
  for (int i=0; i<CkpvAccess(usrEvents).length(); i++) {
    int cur = CkpvAccess(usrEvents)[i]->e;
    if (cur == e) 
      CmiAbort("UserEvent double registered!");
    if (cur > biggest) biggest = cur;
  }
  // if biggest is -1, it means no user events were previously registered
  // hence automatically assigned events will start from id of 0.
  if (e==-1) event = biggest+1; // automatically assign new event id
  else event = e;
  CkpvAccess(usrEvents).push_back(new UsrEvent(event,(char *)evt));
  return event;
}

void TraceProjector::traceClearEps(void)
{
  // In trace-summary, this zeros out the EP bins, to eliminate noise
  // from startup.  Here, this isn't useful, since we can do that in
  // post-processing
}


extern "C" void writeSts(){
	FILE *stsfp;
	char *fname = new char[strlen(CkpvAccess(traceRoot))+strlen(".sts")+1];
	sprintf(fname, "%s.sts", CkpvAccess(traceRoot));
	do{
		stsfp = fopen(fname, "w");
	} while (!stsfp && (errno == EINTR || errno == EMFILE));
	if(stsfp==0)
		CmiAbort("Cannot open projections sts file for writing.\n");
	delete[] fname;
		    
	 fprintf(stsfp, "VERSION %s\n", PROJECTION_VERSION);
	 traceWriteSTS(stsfp,CkpvAccess(usrEvents).length());
	 int i;
	 for(i=0;i<CkpvAccess(usrEvents).length();i++)
	      fprintf(stsfp, "EVENT %d %s\n", CkpvAccess(usrEvents)[i]->e, CkpvAccess(usrEvents)[i]->str);
	 fprintf(stsfp, "END\n");
	fclose(stsfp);
			     
}


void TraceProjector::traceWriteSts(void)
{
	if(CkMyPe()==0)
		writeSts();
}

void TraceProjector::traceClose(void)
{
    if(CkMyPe()==0){
	    writeSts();
    }
    CkpvAccess(_traceproj)->endComputation();   
    closeTraceCore(); 
}

void TraceProjector::traceBegin(void)
{
}

void TraceProjector::traceEnd(void) 
{
}

void TraceProjector::userEvent(int e)
{
	_LOG_E_USER_EVENT_CHARM(e);
}

void TraceProjector::userBracketEvent(int e, double bt, double et)
{
	_LOG_E_USER_EVENT_PAIR_CHARM(e,bt,et);
}

void TraceProjector::creation(envelope *e, int ep,int num)
{
	_LOG_E_CREATION_N(e, ep, num);
}

void TraceProjector::beginExecute(envelope *e)
{
	//_LOG_E_BEGIN_EXECUTE(e);
	
	charm_beginExecute(e);
}

void TraceProjector::beginExecute(CmiObjId  *tid)
{
	// FIXME-- log this
	
	_LOG_E_BEGIN_EXECUTE(0);
}


void TraceProjector::beginExecute(int event,int msgType,int ep,int srcPe,int mlen,CmiObjId *idx)
{
	//CmiPrintf("TraceProjector:iData in beginExecuteDetailed %d %d \n",event,srcPe);
	_LOG_E_BEGIN_EXECUTE_DETAILED(event, msgType, ep, srcPe, mlen);
}

void TraceProjector::endExecute(void)
{
	_LOG_E_END_EXECUTE();
}

void TraceProjector::messageRecv(char *env, int pe)
{
	_LOG_E_MSG_RECV_CHARM(env, pe);
}

void TraceProjector::beginIdle(double curWallTime)
{
	_LOG_E_PROC_IDLE();
}

void TraceProjector::endIdle(double curWallTime)
{
	_LOG_E_PROC_BUSY();
}

void TraceProjector::beginPack(void)
{
	_LOG_E_BEGIN_PACK();
}

void TraceProjector::endPack(void)
{
	_LOG_E_END_PACK();
}

void TraceProjector::beginUnpack(void)
{
	_LOG_E_BEGIN_UNPACK();
}

void TraceProjector::endUnpack(void)
{
	_LOG_E_END_UNPACK();
}

void TraceProjector::enqueue(envelope *env) 
{
	_LOG_E_ENQUEUE(env);
}

void TraceProjector::dequeue(envelope *env) 
{
	_LOG_E_DEQUEUE(env);
}

void TraceProjector::beginComputation(void)
{
	_LOG_E_BEGIN_COMPUTATION();
}

void TraceProjector::endComputation(void)
{
	_LOG_E_END_COMPUTATION();
}

