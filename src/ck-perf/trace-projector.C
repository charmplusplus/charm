/***********Projector tracing file added by Sayantan ****************/

#include "charm++.h"
#include "trace-projector.h"
#include "trace-projections.h"

#define DEBUGF(x)           // CmiPrintf x

CkpvStaticDeclare(Trace*, _traceproj);


#ifdef CMK_OPTIMIZE
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
	/** Projector doesn't have it at the moment **/
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
	 fprintf(stsfp, "MACHINE %s\n",CMK_MACHINE_NAME);
	 fprintf(stsfp, "PROCESSORS %d\n", CkNumPes());
	 fprintf(stsfp, "TOTAL_CHARES %d\n", _numChares);
         fprintf(stsfp, "TOTAL_EPS %d\n", _numEntries);
	 fprintf(stsfp, "TOTAL_MSGS %d\n", _numMsgs);
	 fprintf(stsfp, "TOTAL_PSEUDOS %d\n", 0);
	 
	 //fprintf(stsfp, "TOTAL_EVENTS %d\n", CkpvAccess(usrEvents).length()); --- bad hack
	 fprintf(stsfp, "TOTAL_EVENTS 0\n");
	 
	 int i;
	 for(i=0;i<_numChares;i++)
	      fprintf(stsfp, "CHARE %d %s\n", i, _chareTable[i]->name);
	 for(i=0;i<_numEntries;i++)
	      fprintf(stsfp, "ENTRY CHARE %d %s %d %d\n", i, _entryTable[i]->name,
		 _entryTable[i]->chareIdx, _entryTable[i]->msgIdx);
	 for(i=0;i<_numMsgs;i++)
	      fprintf(stsfp, "MESSAGE %d %d\n", i, _msgTable[i]->size);
	 /*for(i=0;i<CkpvAccess(usrEvents).length();i++)
	      fprintf(stsfp, "EVENT %d %s\n", CkpvAccess(usrEvents)[i]->e, CkpvAccess(usrEvents)[i]->str);*/
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
}

void TraceProjector::creation(envelope *e)
{
	_LOG_E_CREATION_1(e);
}
void TraceProjector::creation(envelope *e, int num)
{
	_LOG_E_CREATION_N(e, num);
}

void TraceProjector::beginExecute(envelope *e)
{
	//_LOG_E_BEGIN_EXECUTE(e);
	//CmiPrintf("TraceProjector:iData in beginExecute %d %d \n",e->getEvent(),e->getSrcPe());
	charm_beginExecute(e);
}

void TraceProjector::beginExecute(int event,int msgType,int ep,int srcPe, int mlen)
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

void TraceProjector::beginIdle(void)
{
	_LOG_E_PROC_IDLE();
}

void TraceProjector::endIdle(void)
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

