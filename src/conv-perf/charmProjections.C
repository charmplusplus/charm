
#include "envelope.h"
#include "converse.h"
#include "charmProjections.h"
#include "traceCoreCommon.h"
#include "charmEvents.h"

CtvStaticDeclare(int,curThreadEvent);

static int _numEvents = 0;
static int _threadMsg, _threadChare, _threadEP;
static int curEvent;
static int execEvent;
static int execEp;
static int execPe;

extern "C" void initCharmProjections() 
{
  CtvInitialize(int,curThreadEvent);
  CtvAccess(curThreadEvent) = 0;
  curEvent = 0;
}

extern "C" int  traceRegisterUserEvent(const char*) {}	//TODO

extern "C" void creation(envelope *e, int num)
{
  if(e==0) {
    CtvAccess(curThreadEvent)=curEvent;
	int* iData = (int*)malloc(sizeof(int)*4); 
	iData[0] = ForChareMsg;
	iData[1] = _threadEP;
	iData[2] = curEvent++;
	iData[3] = CkMyPe();
	LogEvent(_CHARM_LANG_ID, _E_CREATION, 4, iData); 
  } else {
    e->setEvent(curEvent);
    for(int i=0; i<num; i++) {
		int* iData = (int*)malloc(sizeof(int)*5); 
		iData[0] = e->getMsgtype();
		iData[1] = e->getEpIdx();
		iData[2] = curEvent+i;
		iData[3] = CkMyPe();
		iData[4] = e->getTotalsize();
		LogEvent(_CHARM_LANG_ID, _E_CREATION, 5, iData); 
    }
    curEvent += num;
  }
}

extern "C" void beginExecute(envelope *e)
{
  if(e==0) {
    execEvent = CtvAccess(curThreadEvent);
    execEp = (-1);
	int* iData = (int*)malloc(sizeof(int)*4); 
	iData[0] = ForChareMsg;
	iData[1] = _threadEP;
	iData[2] = execEvent;
	iData[3] = CkMyPe();
	LogEvent(_CHARM_LANG_ID, _E_BEGIN_PROCESSING, 4, iData); 
  } else {
    beginExecute(e->getEvent(),e->getMsgtype(),e->getEpIdx(),e->getSrcPe(),e->getTotalsize());
  }
}

extern "C" void beginExecute(int event,int msgType,int ep,int srcPe,int ml)
{
  execEvent=event;
  execEp=ep;
  execPe=srcPe;
  int* iData = (int*)malloc(sizeof(int)*5); 
  iData[0] = msgType;
  iData[1] = ep;
  iData[2] = event;
  iData[3] = srcPe;
  iData[4] = ml;
  LogEvent(_CHARM_LANG_ID, _E_BEGIN_PROCESSING, 5, iData); 
}

extern "C" void endExecute(void)
{
  if(execEp == (-1)) {
	int* iData = (int*)malloc(sizeof(int)*4); 
	iData[0] = 0;
	iData[1] = _threadEP;
	iData[2] = execEvent;
	iData[3] = CkMyPe();
	LogEvent(_CHARM_LANG_ID, _E_END_PROCESSING, 4, iData); 
  } else {
	int* iData = (int*)malloc(sizeof(int)*4); 
	iData[0] = 0;
	iData[1] = execEp;
	iData[2] = execEvent;
	iData[3] = execPe;
	LogEvent(_CHARM_LANG_ID, _E_END_PROCESSING, 4, iData); 
  }
}

extern "C" void enqueue(envelope *e) {}	//TODO

extern "C" void dequeue(envelope *e) {}	//TODO

extern "C" void beginComputation(void)
{
  	if(CkMyRank()==0) {
		//DOUBT: what are these registrations ? ... cocurrently running with projections => problem 
    	//_threadMsg = CkRegisterMsg("dummy_thread_msg", 0, 0, 0, 0);
    	//_threadChare = CkRegisterChare("dummy_thread_chare", 0);
    	//_threadEP = CkRegisterEp("dummy_thread_ep", 0, _threadMsg,_threadChare);
  	}
	int* iData = (int*)malloc(sizeof(int)*4); 
	iData[0] = iData[1] = 0;
	iData[2] = iData[3] = -1;
	LogEvent(_CHARM_LANG_ID, _E_BEGIN_COMPUTATION, 4, iData); 
}

extern "C" void endComputation(void)
{
	int* iData = (int*)malloc(sizeof(int)*4); 
	iData[0] = iData[1] = 0;
	iData[2] = iData[3] = -1;
	LogEvent(_CHARM_LANG_ID, _E_END_COMPUTATION, 4, iData); 
}

extern "C" void messageRecv(char *env, int pe) {} //TODO

extern "C" void userEvent(int e)
extern "C" void beginPack(void)
{
	int* iData = (int*)malloc(sizeof(int)*4); 
	iData[0] = iData[1] = iData[2] = 0;
	iData[3] = CkMyPe();
	LogEvent(_CHARM_LANG_ID, _E_BEGIN_PACK, 4, iData); 
}

extern "C" void endPack(void)
{
	int* iData = (int*)malloc(sizeof(int)*4); 
	iData[0] = iData[1] = iData[2] = 0;
	iData[3] = CkMyPe();
	LogEvent(_CHARM_LANG_ID, _E_END_PACK, 4, iData); 
}

extern "C" void beginUnpack(void)
{
	int* iData = (int*)malloc(sizeof(int)*4); 
	iData[0] = iData[1] = iData[2] = 0;
	iData[3] = CkMyPe();
	LogEvent(_CHARM_LANG_ID, _E_BEGIN_UNPACK, 4, iData); 
}

extern "C" void endUnpack(void)
{
	int* iData = (int*)malloc(sizeof(int)*4); 
	iData[0] = iData[1] = iData[2] = 0;
	iData[3] = CkMyPe();
	LogEvent(_CHARM_LANG_ID, _E_END_UNPACK, 4, iData); 
}


