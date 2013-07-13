extern "C" void CkExit(void);

#include "mpi-interoperate.h"

static int   _libExitStarted = 0;
int    _libExitHandlerIdx;

#if CMK_CONVERSE_MPI
extern  "C" { extern MPI_Comm charmComm ;}
#endif

extern int _ringexit;		    // for charm exit
extern int _ringtoken;
extern void _initCharm(int unused_argc, char **argv);


extern "C"
void StartCharmScheduler() {
  CsdScheduler(-1);
}

// triger LibExit on PE 0,
extern "C"
void LibCkExit(void)
{
	// always send to PE 0
	envelope *env = _allocEnv(StartExitMsg);
	env->setSrcPe(CkMyPe());
	CmiSetHandler(env, _libExitHandlerIdx);
	CmiSyncSendAndFree(0, env->getTotalsize(), (char *)env);
}

void _libExitHandler(envelope *env)
{
	switch(env->getMsgtype()) {
		case StartExitMsg:
			CkAssert(CkMyPe()==0);
			// else goto next
		case ExitMsg:
			CkAssert(CkMyPe()==0);
			if(_libExitStarted) {
				CmiFree(env);
				return;
			}
			_libExitStarted = 1;
			env->setMsgtype(ReqStatMsg);
			env->setSrcPe(CkMyPe());
			// if exit in ring, instead of broadcasting, send in ring
			if (_ringexit){
				const int stride = CkNumPes()/_ringtoken;
				int pe = 0; while (pe<CkNumPes()) {
					CmiSyncSend(pe, env->getTotalsize(), (char *)env);
					pe += stride;
				}
				CmiFree(env);
			}else{
				CmiSyncBroadcastAllAndFree(env->getTotalsize(), (char *)env);
			}	
			break;
		case ReqStatMsg:
			if (_ringexit) {
				int stride = CkNumPes()/_ringtoken;
				int pe = CkMyPe()+1;
				if (pe < CkNumPes() && pe % stride != 0)
					CmiSyncSendAndFree(pe, env->getTotalsize(), (char *)env);
				else
					CmiFree(env);
			}
			else
				CmiFree(env);
			//everyone exits here - there may be issues with leftover messages in the queue
			_libExitStarted = 0;
			CpvAccess(charmLibExitFlag) = 1;
			break;
		default:
			CmiAbort("Internal Error(_libExitHandler): Unknown-msg-type. Contact Developers.\n");
	}
}

#if CMK_CONVERSE_MPI
extern "C"
void CharmLibInit(MPI_Comm userComm, int argc, char **argv){
	//note CmiNumNodes and CmiMyNode should just be macros
  charmComm = userComm;
  MPI_Comm_size(charmComm, &_Cmi_numnodes);
  MPI_Comm_rank(charmComm, &_Cmi_mynode);

	CharmLibInterOperate = 1;
	ConverseInit(argc, argv, (CmiStartFn)_initCharm, 1, 0);
}
#else
extern "C"
void CharmLibInit(int userComm, int argc, char **argv){
    CmiAbort("mpi-interoperate only supports MPI machine layer");
}
#endif

#undef CkExit
#define CkExit CkExit
extern "C"
void CharmLibExit() {
	if(CkMyPe() == 0) {
		CkExit();
	}
	CsdScheduler(-1);
}

