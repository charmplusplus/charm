/**
  * Fast Message Logging Fault Tolerance Protocol.
  * Features:
 	* Reduces the latency overhead of causal message-logging approach. It does NOT use determinants at all.
 	* Supports multiple concurrent failures.
 	* Assumes the application is iterative and structured in communication.
  */

#include "charm.h"
#include "ck.h"
#include "ckmessagelogging.h"
#include "queueing.h"
#include <sys/types.h>
#include <signal.h>
#include "CentralLB.h"

#ifdef _FAULT_MLOG_

// Collects some statistics about message logging. 
#define COLLECT_STATS_MSGS 0
#define COLLECT_STATS_MSGS_TOTAL 0
#define COLLECT_STATS_MSG_COUNT 0
#define COLLECT_STATS_MEMORY 0
#define COLLECT_STATS_TEAM 0

#define RECOVERY_SEND "SEND"
#define RECOVERY_PROCESS "PROCESS"

#define DEBUG_MEM(x)  //x
#define DEBUG(x) // x
#define DEBUG_RESTART(x)  //x
#define DEBUGLB(x)   // x
#define DEBUG_TEAM(x)  // x
#define DEBUG_PERF(x) // x
#define DEBUG_CHECKPOINT 1
#define DEBUG_NOW(x) x
#define DEBUG_PE(x,y) // if(CkMyPe() == x) y
#define DEBUG_PE_NOW(x,y)  if(CkMyPe() == x) y
#define DEBUG_RECOVERY(x) //x

#define FAIL_DET_THRESHOLD 10

extern const char *idx2str(const CkArrayIndex &ind);
extern const char *idx2str(const ArrayElement *el);

void getGlobalStep(CkGroupID gID);

bool fault_aware(CkObjID &recver);
void createObjIDList(void *data,ChareMlogData *mlogData);
inline bool isLocal(int destPE);
inline bool isTeamLocal(int destPE);
void printLog(CkObjID *log);

int _restartFlag=0;
int _numRestartResponses=0;

char *checkpointDirectory=".";
int unAckedCheckpoint=0;
int countUpdateHomeAcks=0;

extern int teamSize;
extern int chkptPeriod;
extern bool fastRecovery;
extern int parallelRecovery;

char *killFile;
char *faultFile;
int killFlag=0;
int faultFlag=0;
int restartingMlogFlag=0;
void readKillFile();
double killTime=0.0;
double faultMean;
int checkpointCount=0;
int diskCkptFlag = 0;
static char fName[100];

/***** VARIABLES FOR MESSAGE LOGGING *****/
// stores the id of current object sending a message
CpvDeclare(Chare *,_currentObj);
// stores checkpoint from buddy
CpvDeclare(StoredCheckpoint *,_storedCheckpointData);
// stores the incarnation number from every other processor
CpvDeclare(char *, _incarnation);
/***** *****/

/***** VARIABLES FOR PARALLEL RECOVERY *****/
CpvDeclare(int, _numEmigrantRecObjs);
CpvDeclare(int, _numImmigrantRecObjs);
CpvDeclare(CkVec<CkLocation *> *, _immigrantRecObjs);
/***** *****/

#if COLLECT_STATS_MSGS
int *numMsgsTarget;
int *sizeMsgsTarget;
int totalMsgsTarget;
float totalMsgsSize;
#endif
#if COLLECT_STATS_MEMORY
int msgLogSize;
int bufferedDetsSize;
int storedDetsSize;
#endif

/***** IDS FOR MESSAGE LOGGING HANDLERS *****/
int _pingHandlerIdx;
int _checkpointRequestHandlerIdx;
int _storeCheckpointHandlerIdx;
int _checkpointAckHandlerIdx;
int _getCheckpointHandlerIdx;
int _recvCheckpointHandlerIdx;
int _dummyMigrationHandlerIdx;
int	_getGlobalStepHandlerIdx;
int	_recvGlobalStepHandlerIdx;
int _updateHomeRequestHandlerIdx;
int _updateHomeAckHandlerIdx;
int _resendMessagesHandlerIdx;
int _receivedDetDataHandlerIdx;
int _distributedLocationHandlerIdx;
int _sendBackLocationHandlerIdx;

void setTeamRecovery(void *data, ChareMlogData *mlogData);
void unsetTeamRecovery(void *data, ChareMlogData *mlogData);
int _falseRestart =0; /**
													For testing on clusters we might carry out restarts on 
													a porcessor without actually starting it
													1 -> false restart
													0 -> restart after an actual crash
												*/

//Load balancing globals
int onGoingLoadBalancing=0;
void *centralLb;
void (*resumeLbFnPtr)(void *);
int _receiveMlogLocationHandlerIdx;
int _checkpointBarrierHandlerIdx;
int _checkpointBarrierAckHandlerIdx;
int _startCheckpointIdx;
int _endCheckpointIdx;
int donotCountMigration=0;
int countLBMigratedAway=0;
int countLBToMigrate=0;
int migrationDoneCalled=0;
int checkpointBarrierCount=0;
int globalResumeCount=0;
CkGroupID globalLBID;
int restartDecisionNumber=-1;
double lastCompletedAlarm=0;
double lastRestart=0;
CkCallback ckptCallback;

//update location globals
int _receiveLocationHandlerIdx;

#if CMK_CONVERSE_MPI
static int heartBeatHandlerIdx;
static int heartBeatCheckHandlerIdx;
static int partnerFailureHandlerIdx;
static double lastPingTime = -1;

extern "C" void mpi_restart_crashed(int pe, int rank);
extern "C" int  find_spare_mpirank(int pe, int partition);

void heartBeatPartner();
void heartBeatHandler(void *msg);
void heartBeatCheckHandler();
void partnerFailureHandler(char *msg);
int getReverseCheckPointPE();
int inCkptFlag = 0;
#endif

static void *doNothingMsg(int * size, void * data, void ** remote, int count){
	return data;
}

/***** *****/

/** 
 * @brief Initialize message logging data structures and register handlers
 */
void _messageLoggingInit(){
	if(CkMyPe() == 0)
		CkPrintf("[%d] Fast Message Logging Support \n",CkMyPe());

	//current object
	CpvInitialize(Chare *,_currentObj);
	
	//registering handlers for message logging
	_pingHandlerIdx = CkRegisterHandler(_pingHandler);
		
	//handlers for checkpointing
	_storeCheckpointHandlerIdx = CkRegisterHandler(_storeCheckpointHandler);
	_checkpointAckHandlerIdx = CkRegisterHandler( _checkpointAckHandler);
	_checkpointRequestHandlerIdx =  CkRegisterHandler(_checkpointRequestHandler);

	//handlers for restart
	_getCheckpointHandlerIdx = CkRegisterHandler(_getCheckpointHandler);
	_recvCheckpointHandlerIdx = CkRegisterHandler(_recvCheckpointHandler);
	_resendMessagesHandlerIdx = CkRegisterHandler(_resendMessagesHandler);
	_distributedLocationHandlerIdx=CkRegisterHandler(_distributedLocationHandler);
	_sendBackLocationHandlerIdx=CkRegisterHandler(_sendBackLocationHandler);
	_dummyMigrationHandlerIdx = CkRegisterHandler(_dummyMigrationHandler);

	//handlers for load balancing
	_receiveMlogLocationHandlerIdx=CkRegisterHandler(_receiveMlogLocationHandler);
	_getGlobalStepHandlerIdx=CkRegisterHandler(_getGlobalStepHandler);
	_recvGlobalStepHandlerIdx=CkRegisterHandler(_recvGlobalStepHandler);
	_checkpointBarrierHandlerIdx=CkRegisterHandler(_checkpointBarrierHandler);
	_checkpointBarrierAckHandlerIdx=CkRegisterHandler(_checkpointBarrierAckHandler);
	_startCheckpointIdx = CkRegisterHandler(_startCheckpointHandler);
	_endCheckpointIdx = CkRegisterHandler(_endCheckpointHandler);
	
	//handlers for updating locations
	_receiveLocationHandlerIdx=CkRegisterHandler(_receiveLocationHandler);

	// handlers for failure detection in MPI layer
#if CMK_CONVERSE_MPI
	heartBeatHandlerIdx = CkRegisterHandler(heartBeatHandler);
	heartBeatCheckHandlerIdx = CkRegisterHandler(heartBeatCheckHandler);
	partnerFailureHandlerIdx = CkRegisterHandler(partnerFailureHandler);
#endif
	
	// Cpv variables for message-logging protocol
	CpvInitialize(char *, _incarnation);
	CpvAccess(_incarnation) = (char *) CmiAlloc(CmiNumPes() * sizeof(int));
	for(int i=0; i<CmiNumPes(); i++){
		CpvAccess(_incarnation)[i] = 0;
	}

	// Cpv variables for parallel recovery
	CpvInitialize(int, _numEmigrantRecObjs);
    CpvAccess(_numEmigrantRecObjs) = 0;
    CpvInitialize(int, _numImmigrantRecObjs);
    CpvAccess(_numImmigrantRecObjs) = 0;

    CpvInitialize(CkVec<CkLocation *> *, _immigrantRecObjs);
    CpvAccess(_immigrantRecObjs) = new CkVec<CkLocation *>;

	//Cpv variables for checkpoint
	CpvInitialize(StoredCheckpoint *,_storedCheckpointData);
	CpvAccess(_storedCheckpointData) = new StoredCheckpoint;

	// disk checkpoint
	if(diskCkptFlag){
#if CMK_USE_MKSTEMP
		sprintf(fName, "/tmp/ckpt%d-XXXXXX", CkMyPe());
		mkstemp(fName);
#else
		fName=tmpnam(NULL);
#endif
	}

	// registering user events for projections	
	traceRegisterUserEvent("Remove Logs", 20);
	traceRegisterUserEvent("Ticket Request Handler", 21);
	traceRegisterUserEvent("Ticket Handler", 22);
	traceRegisterUserEvent("Local Message Copy Handler", 23);
	traceRegisterUserEvent("Local Message Ack Handler", 24);	
	traceRegisterUserEvent("Preprocess current message",25);
	traceRegisterUserEvent("Preprocess past message",26);
	traceRegisterUserEvent("Preprocess future message",27);
	traceRegisterUserEvent("Checkpoint",28);
	traceRegisterUserEvent("Checkpoint Store",29);
	traceRegisterUserEvent("Checkpoint Ack",30);
	traceRegisterUserEvent("Send Ticket Request",31);
	traceRegisterUserEvent("Generalticketrequest1",32);
	traceRegisterUserEvent("TicketLogLocal",33);
	traceRegisterUserEvent("next_ticket and SN",34);
	traceRegisterUserEvent("Timeout for buffered remote messages",35);
	traceRegisterUserEvent("Timeout for buffered local messages",36);
	traceRegisterUserEvent("Inform Location Home",37);
	traceRegisterUserEvent("Receive Location Handler",38);
	
	lastCompletedAlarm=CmiWallTimer();
	lastRestart = CmiWallTimer();

	// fault detection for MPI layer
#if CMK_CONVERSE_MPI
  void heartBeatPartner();
  void heartBeatCheckHandler();
  CcdCallOnCondition(CcdPERIODIC_100ms,(CcdVoidFn)heartBeatPartner,NULL);
  CcdCallOnCondition(CcdPERIODIC_5s,(CcdVoidFn)heartBeatCheckHandler,NULL);
#endif

#if COLLECT_STATS_MSGS
#if COLLECT_STATS_MSGS_TOTAL
	totalMsgsTarget = 0;
	totalMsgsSize = 0.0;
#else
	numMsgsTarget = (int *)CmiAlloc(sizeof(int) * CmiNumPes());
	sizeMsgsTarget = (int *)CmiAlloc(sizeof(int) * CmiNumPes());
	for(int i=0; i<CmiNumPes(); i++){
		numMsgsTarget[i] = 0;
		sizeMsgsTarget[i] = 0;
	}
#endif
#endif
#if COLLECT_STATS_MEMORY
	msgLogSize = 0;
	bufferedDetsSize = 0;
	storedDetsSize = 0;
#endif

}

#if CMK_CONVERSE_MPI

/**
 * Receives the notification of a failure and updates pe-to-rank mapping.
 */
void partnerFailureHandler(char *msg)
{
   int diepe = *(int *)(msg+CmiMsgHeaderSizeBytes);

   // send message to crash pe to let it restart
   int newrank = find_spare_mpirank(diepe, CmiMyPartition());
   int buddy = getReverseCheckPointPE();
   if (buddy == diepe)  {
     mpi_restart_crashed(diepe, newrank);
     CcdCallOnCondition(CcdPERIODIC_5s,(CcdVoidFn)heartBeatCheckHandler,NULL);
   }
}

/**
 * Registers last time it knew about the PE that checkpoints on it.
 */
void heartBeatHandler(void *msg)
{
	lastPingTime = CmiWallTimer();
	CmiFree(msg);
}

/**
 * Checks whether the PE that checkpoints on it is still alive.
 */
void heartBeatCheckHandler()
{
	double now = CmiWallTimer();
	if (lastPingTime > 0 && now - lastPingTime > FAIL_DET_THRESHOLD && !inCkptFlag) {
		int i, pe, buddy;
		// tell everyone that PE is down
		buddy = getReverseCheckPointPE();
		CmiPrintf("[%d] detected buddy processor %d died %f %f. \n", CmiMyPe(), buddy, now, lastPingTime);
		
		for (int pe = 0; pe < CmiNumPes(); pe++) {
			if (pe == buddy) continue;
			char *msg = (char*)CmiAlloc(CmiMsgHeaderSizeBytes+sizeof(int));
			*(int *)(msg+CmiMsgHeaderSizeBytes) = buddy;
			CmiSetHandler(msg, partnerFailureHandlerIdx);
			CmiSyncSendAndFree(pe, CmiMsgHeaderSizeBytes+sizeof(int), (char *)msg);
		}
	}
	else 
		CcdCallOnCondition(CcdPERIODIC_5s,(CcdVoidFn)heartBeatCheckHandler,NULL);
}

/**
 * Pings buddy to let it know this PE is alive. Used for failure detection.
 */
void heartBeatPartner()
{
	int buddy = getCheckPointPE();
	//printf("[%d] heartBeatPartner %d\n", CmiMyPe(), buddy);
	char *msg = (char*)CmiAlloc(CmiMsgHeaderSizeBytes+sizeof(int));
	*(int *)(msg+CmiMsgHeaderSizeBytes) = CmiMyPe();
	CmiSetHandler(msg, heartBeatHandlerIdx);
	CmiSyncSendAndFree(buddy, CmiMsgHeaderSizeBytes+sizeof(int), (char *)msg);
	CcdCallOnCondition(CcdPERIODIC_100ms,(CcdVoidFn)heartBeatPartner,NULL);
}
#endif

void killLocal(void *_dummy,double curWallTime);	

void readKillFile(){
	FILE *fp=fopen(killFile,"r");
	if(!fp){
		return;
	}
	int proc;
	double sec;
	while(fscanf(fp,"%d %lf",&proc,&sec)==2){
		if(proc == CkMyPe()){
			killTime = CmiWallTimer()+sec;
			printf("[%d] To be killed after %.6lf s (MLOG) \n",CkMyPe(),sec);
			CcdCallFnAfter(killLocal,NULL,sec*1000);	
		}
	}
	fclose(fp);
}

/**
 * @brief: reads the PE that will be failing throughout the execution and the mean time between failures.
 * We assume an exponential distribution for the mean-time-between-failures.
 */
void readFaultFile(){
        FILE *fp=fopen(faultFile,"r");
        if(!fp){
                return;
        }
        int proc;
        double sec;
        fscanf(fp,"%d %lf",&proc,&sec);
	faultMean = sec;
	if(proc == CkMyPe()){
	        printf("[%d] PE %d to be killed every %.6lf s (MEMCKPT) \n",CkMyPe(),proc,sec);
        	CcdCallFnAfter(killLocal,NULL,sec*1000);
	}
        fclose(fp);
}

void killLocal(void *_dummy,double curWallTime){
	printf("[%d] KillLocal called at %.6lf \n",CkMyPe(),CmiWallTimer());
	if(CmiWallTimer()<killTime-1){
		CcdCallFnAfter(killLocal,NULL,(killTime-CmiWallTimer())*1000);	
	}else{
#if CMK_CONVERSE_MPI
		CkDieNow();
#else
		kill(getpid(),SIGKILL);
#endif
	}
}

#if ! CMK_CONVERSE_MPI
void CkDieNow()
{
	// kill -9 for non-mpi version
	CmiPrintf("[%d] die now.\n", CmiMyPe());
	killTime = CmiWallTimer()+0.001;
	CcdCallFnAfter(killLocal,NULL,1);
}
#endif


/*** Auxiliary Functions ***/

/************************ Message logging methods ****************/

/**
 * Sends a group message that might be a broadcast.
 */
void sendGroupMsg(envelope *env, int destPE, int _infoIdx){
	if(destPE == CLD_BROADCAST || destPE == CLD_BROADCAST_ALL){
		DEBUG(printf("[%d] Group Broadcast \n",CkMyPe()));
		void *origMsg = EnvToUsr(env);
		for(int i=0;i<CmiNumPes();i++){
			if(!(destPE == CLD_BROADCAST && i == CmiMyPe())){
				void *copyMsg = CkCopyMsg(&origMsg);
				envelope *copyEnv = UsrToEnv(copyMsg);
				copyEnv->SN=0;
				copyEnv->sender.type = TypeInvalid;
				DEBUG(printf("[%d] Sending group broadcast message to proc %d \n",CkMyPe(),i));
				sendGroupMsg(copyEnv,i,_infoIdx);
			}
		}
		return;
	}

	// initializing values of envelope
	env->SN=0;
	env->sender.type = TypeInvalid;

	CkObjID recver;
	recver.type = TypeGroup;
	recver.data.group.id = env->getGroupNum();
	recver.data.group.onPE = destPE;
	sendCommonMsg(recver,env,destPE,_infoIdx);
}

/**
 * Sends a nodegroup message that might be a broadcast.
 */
void sendNodeGroupMsg(envelope *env, int destNode, int _infoIdx){
	if(destNode == CLD_BROADCAST || destNode == CLD_BROADCAST_ALL){
		DEBUG(printf("[%d] NodeGroup Broadcast \n",CkMyPe()));
		void *origMsg = EnvToUsr(env);
		for(int i=0;i<CmiNumNodes();i++){
			if(!(destNode == CLD_BROADCAST && i == CmiMyNode())){
				void *copyMsg = CkCopyMsg(&origMsg);
				envelope *copyEnv = UsrToEnv(copyMsg);
				copyEnv->SN=0;
				copyEnv->sender.type = TypeInvalid;
				sendNodeGroupMsg(copyEnv,i,_infoIdx);
			}
		}
		return;
	}

	// initializing values of envelope
	env->SN=0;
	env->sender.type = TypeInvalid;

	CkObjID recver;
	recver.type = TypeNodeGroup;
	recver.data.group.id = env->getGroupNum();
	recver.data.group.onPE = destNode;
	sendCommonMsg(recver,env,destNode,_infoIdx);
}

/**
 * Sends a message to an array element.
 */
void sendArrayMsg(envelope *env,int destPE,int _infoIdx){
	CkObjID recver;
	recver.type = TypeArray;
	recver.data.array.id = env->getArrayMgr();
	recver.data.array.idx.asChild() = *(&env->getsetArrayIndex());

	if(CpvAccess(_currentObj)!=NULL &&  CpvAccess(_currentObj)->mlogData->objID.type != TypeArray){
		char recverString[100],senderString[100];
		
		DEBUG(printf("[%d] %s being sent message from non-array %s \n",CkMyPe(),recver.toString(recverString),CpvAccess(_currentObj)->mlogData->objID.toString(senderString)));
	}

	// initializing values of envelope
	env->SN = 0;

	sendCommonMsg(recver,env,destPE,_infoIdx);
};

/**
 * Sends a message to a singleton chare.
 */
void sendChareMsg(envelope *env,int destPE,int _infoIdx, const CkChareID *pCid){
	CkObjID recver;
	recver.type = TypeChare;
	recver.data.chare.id = *pCid;

	if(CpvAccess(_currentObj)!=NULL &&  CpvAccess(_currentObj)->mlogData->objID.type != TypeArray){
		char recverString[100],senderString[100];
		
		DEBUG(printf("[%d] %s being sent message from non-array %s \n",CkMyPe(),recver.toString(recverString),CpvAccess(_currentObj)->mlogData->objID.toString(senderString)));
	}

	// initializing values of envelope
	env->SN = 0;

	sendCommonMsg(recver,env,destPE,_infoIdx);
};

/**
 * A method to generate the actual ticket requests for groups, nodegroups or arrays.
 */
void sendCommonMsg(CkObjID &recver,envelope *_env,int destPE,int _infoIdx){
	envelope *env = _env;
	int resend=0; //is it a resend
	DEBUG(char recverName[100]);
	DEBUG(char senderString[100]);
	
	DEBUG_MEM(CmiMemoryCheck());

	if(CpvAccess(_currentObj) == NULL){
//		CkAssert(0);
		DEBUG(printf("[%d] !!!!WARNING: _currentObj is NULL while message is being sent\n",CkMyPe());)
		generalCldEnqueue(destPE,env,_infoIdx);
		return;
	}

	// checking if this message should bypass determinants in message-logging
	if(env->flags & CK_BYPASS_DET_MLOG){
	 	env->sender = CpvAccess(_currentObj)->mlogData->objID;
		env->recver = recver;
		DEBUG(CkPrintf("[%d] Bypassing determinants from %s to %s PE %d\n",CkMyPe(),CpvAccess(_currentObj)->mlogData->objID.toString(senderString),recver.toString(recverName),destPE));
		generalCldEnqueue(destPE,env,_infoIdx);
		return;
	}
	
	// setting message logging data in the envelope
	env->incarnation = CpvAccess(_incarnation)[CkMyPe()];
	env->sender = CpvAccess(_currentObj)->mlogData->objID;
	env->SN = 0;
	
	DEBUG_MEM(CmiMemoryCheck());

	CkObjID &sender = env->sender;
	env->recver = recver;

	Chare *obj = (Chare *)env->sender.getObject();
	  
	if(env->SN == 0){
		DEBUG_MEM(CmiMemoryCheck());
		env->SN = obj->mlogData->nextSN(recver);
	}else{
		resend = 1;
	}

	// uses the proper sending mechanism for local or remote messages
	if(isLocal(destPE)){
		sendLocalMsg(env, _infoIdx);
	}else{
		MlogEntry *mEntry = new MlogEntry(env,destPE,_infoIdx);
		sendRemoteMsg(sender,recver,destPE,mEntry,env->SN,resend);
	}
}

/**
 * Determines if the message is local or not. A message is local if:
 * 1) Both the destination and origin are the same PE.
 */
inline bool isLocal(int destPE){
	// both the destination and the origin are the same PE
	if(destPE == CkMyPe())
		return true;

	return false;
}

/**
 * Determines if the message is group local or not. A message is group local if:
 * 1) They belong to the same team in the team-based message logging.
 */
inline bool isTeamLocal(int destPE){

	// they belong to the same group
	if(teamSize > 1 && destPE/teamSize == CkMyPe()/teamSize)
		return true;

	return false;
}

/**
 * Method that does the actual send by creating a ticket request filling it up and sending it.
 */
void sendRemoteMsg(CkObjID &sender,CkObjID &recver,int destPE,MlogEntry *entry,MCount SN,int resend){
	DEBUG(char recverString[100]);
	DEBUG(char senderString[100]);

	int totalSize;

	envelope *env = entry->env;
	DEBUG_PE(3,printf("[%d] Sending message to %s from %s PE %d SN %d time %.6lf \n",CkMyPe(),env->recver.toString(recverString),env->sender.toString(senderString),destPE,env->SN,CkWallTimer()));

	// setting all the information
	Chare *obj = (Chare *)entry->env->sender.getObject();
	entry->env->recver = recver;
	entry->env->SN = SN;
	if(!resend){
		obj->mlogData->addLogEntry(entry);
#if COLLECT_STATS_TEAM
		MLOGFT_totalMessages += 1.0;
		MLOGFT_totalLogSize += entry->env->getTotalsize();
#endif
	}

	// sending the message
	generalCldEnqueue(destPE, entry->env, entry->_infoIdx);

	DEBUG_MEM(CmiMemoryCheck());
#if COLLECT_STATS_MSGS
#if COLLECT_STATS_MSGS_TOTAL
	totalMsgsTarget++;
	totalMsgsSize += (float)env->getTotalsize();
#else
	numMsgsTarget[destPE]++;
	sizeMsgsTarget[destPE] += env->getTotalsize();
#endif
#endif
#if COLLECT_STATS_MEMORY
	msgLogSize += env->getTotalsize();
#endif
};


/**
 * @brief Function to send a local message. 
 */
void sendLocalMsg(envelope *env, int _infoIdx){
	DEBUG_PERF(double _startTime=CkWallTimer());
	DEBUG_MEM(CmiMemoryCheck());
	DEBUG(Chare *senderObj = (Chare *)env->sender.getObject();)
	DEBUG(char senderString[100]);
	DEBUG(char recverString[100]);

	DEBUG(printf("[%d] Local Message being sent for SN %d sender %s recver %s \n",CmiMyPe(),env->SN,env->sender.toString(senderString),env->recver.toString(recverString)));

	// setting flag to free allocated memory space for this message
	//env->flags = env->flags | CK_FREE_MSG_MLOG;

	// getting the receiver local object
	Chare *recverObj = (Chare *)env->recver.getObject();

	// if receiver object is not NULL, we will ask it for a ticket
	if(recverObj){

		// sends the local message
		_skipCldEnqueue(CmiMyPe(),env,_infoIdx);	

		DEBUG_MEM(CmiMemoryCheck());
	}else{
		DEBUG(printf("[%d] Local recver object is NULL \n",CmiMyPe()););
	}
};

/****
	The handler functions
*****/

bool fault_aware(CkObjID &recver){
	switch(recver.type){
		case TypeChare:
			return true;
		case TypeMainChare:
			return false;
		case TypeGroup:
		case TypeNodeGroup:
		case TypeArray:
			return true;
		default:
			return false;
	}
};

/* Preprocesses a received message */
int preProcessReceivedMessage(envelope *env, Chare **objPointer, MlogEntry **logEntryPointer){
	DEBUG(char recverString[100]);
	DEBUG(char senderString[100]);
	DEBUG_MEM(CmiMemoryCheck());
	int flag;
	bool ticketSuccess;

	// getting the receiver object
	CkObjID recver = env->recver;

	// checking for determinants bypass in message logging
	if(env->flags & CK_BYPASS_DET_MLOG){
		DEBUG(printf("[%d] Bypassing message sender %s recver %s \n",CkMyPe(),env->sender.toString(senderString), recver.toString(recverString)));
		return 1;	
	}

	// checking if receiver is fault aware
	if(!fault_aware(recver)){
		CkPrintf("[%d] Receiver NOT fault aware\n",CkMyPe());
		return 1;
	}

	// checking if object is NULL 
	Chare *obj = (Chare *)recver.getObject();
	*objPointer = obj;
	if(obj == NULL){
		// the objects does not exist on this PE, we let Charm forward the message and update the location manager.
		DEBUG(printf("[%d] Message SN %d sender %s for NULL recver %s\n",CkMyPe(),env->SN,env->sender.toString(senderString),recver.toString(recverString)));
		return 1;
	}

	// checking if message comes from an old incarnation, message must be discarded
	if(env->incarnation < CpvAccess(_incarnation)[env->getSrcPe()]){
		DEBUG(printf("[%d] Stale message SN %d sender %s for recver %s being ignored\n",CkMyPe(),env->SN,env->sender.toString(senderString),recver.toString(recverString)));
		CmiFree(env);
		return 0;
	}

	DEBUG_MEM(CmiMemoryCheck());
	DEBUG_PE(2,printf("[%d] Message received, sender = %s SN %d TN %d tProcessed %d for recver %s at %.6lf \n",CkMyPe(),env->sender.toString(senderString),env->SN,env->TN,obj->mlogData->tProcessed, recver.toString(recverString),CkWallTimer()));

	// checking if message has already been processed, message must be discarded	
	if(obj->mlogData->checkAndStoreSsn(env->sender,env->SN)){
		DEBUG(printf("[%d] Duplicate message SN %d sender %s for recver %s being ignored\n",CkMyPe(),env->SN,env->sender.toString(senderString),recver.toString(recverString)));
		CmiFree(env);
		return 0;
	}

	// message can be processed at this point
	DEBUG(printf("[%d] Message SN %d sender %s for recver %s being delivered\n",CkMyPe(),env->SN,env->sender.toString(senderString),recver.toString(recverString)));
	return 1;
}

/**
 * @brief Updates a few variables once a message has been processed.
 */
void postProcessReceivedMessage(Chare *obj, CkObjID &sender, MCount SN, MlogEntry *entry){
}

/***
	Helpers for the handlers and message logging methods
***/

void generalCldEnqueue(int destPE, envelope *env, int _infoIdx){
//	double _startTime = CkWallTimer();
	if(env->recver.type != TypeNodeGroup){
	//This repeats a step performed in skipCldEnq for messages sent to
	//other processors. I do this here so that messages to local processors
	//undergo the same transformation.. It lets the resend be uniform for 
	//all messages
//		CmiSetXHandler(env,CmiGetHandler(env));
		_skipCldEnqueue(destPE,env,_infoIdx);
	}else{
		_noCldNodeEnqueue(destPE,env);
	}
//	traceUserBracketEvent(22,_startTime,CkWallTimer());
}
//extern "C" int CmiGetNonLocalLength();

void _pingHandler(CkPingMsg *msg){
	printf("[%d] Received Ping from %d\n",CkMyPe(),msg->PE);
	CmiFree(msg);
}


/*****************************************************************************
	Checkpointing methods..
		Pack all the data on a processor and send it to the buddy periodically
		Also used to throw away message logs
*****************************************************************************/
void buildProcessedTicketLog(void *data,ChareMlogData *mlogData);
void clearUpMigratedRetainedLists(int PE);

void checkpointAlarm(void *_dummy,double curWallTime){
	double diff = curWallTime-lastCompletedAlarm;
	DEBUG(printf("[%d] calling for checkpoint %.6lf after last one\n",CkMyPe(),diff));
/*	if(CkWallTimer()-lastRestart < 50){
		CcdCallFnAfter(checkpointAlarm,NULL,chkptPeriod);
		return;
	}*/
	if(diff < ((chkptPeriod) - 2)){
		CcdCallFnAfter(checkpointAlarm,NULL,(chkptPeriod-diff)*1000);
		return;
	}
	CheckpointRequest request;
	request.PE = CkMyPe();
	CmiSetHandler(&request,_checkpointRequestHandlerIdx);
	CmiSyncBroadcastAll(sizeof(CheckpointRequest),(char *)&request);
};

void _checkpointRequestHandler(CheckpointRequest *request){
	startMlogCheckpoint(NULL,CmiWallTimer());
}

/**
 * @brief Starts checkpoint phase at PE 0
 */
void CkStartMlogCheckpoint(CkCallback &cb){

	// storing callback
	ckptCallback = cb;

	// sending a broadcast to start checkpoint
	CheckpointBarrierMsg *msg = (CheckpointBarrierMsg *)CmiAlloc(sizeof(CheckpointBarrierMsg));
	CmiSetHandler(msg,_startCheckpointIdx);
	CmiSyncBroadcastAllAndFree(sizeof(CheckpointBarrierMsg),(char *)msg);
}

/**
 * @brief Starts checkpoint: send its checkpoint to its partner. This checkpointing strategy is NOT connected to the load balancer, 
 * hence onGoingLoadBalancer==0.
 */
void _startCheckpointHandler(CheckpointBarrierMsg *startMsg){
	CmiFree(startMsg);

	// removing messages from the log and cleaning up other data structures
	garbageCollectMlog();

	// increasing the checkpoint counter
	checkpointCount++;

	// setting globalLBID to NULL
	globalLBID.idx = 0;
	
#if CMK_CONVERSE_MPI
	inCkptFlag = 1;
#endif

#if DEBUG_CHECKPOINT
	if(CmiMyPe() == 0){
		printf("[%d] starting checkpoint handler at %.6lf CmiTimer %.6lf \n",CkMyPe(),CmiWallTimer(),CmiTimer());
	}
#endif

	DEBUG_MEM(CmiMemoryCheck());

	PUP::sizer psizer;
	psizer | checkpointCount;
	for(int i=0; i<CmiNumPes(); i++){
		psizer | CpvAccess(_incarnation)[i];
	}
	CkPupROData(psizer);
	DEBUG_MEM(CmiMemoryCheck());
	CkPupGroupData(psizer,true);
	DEBUG_MEM(CmiMemoryCheck());
	CkPupNodeGroupData(psizer,true);
	DEBUG_MEM(CmiMemoryCheck());
	pupArrayElementsSkip(psizer,true,NULL);
	DEBUG_MEM(CmiMemoryCheck());

	int dataSize = psizer.size();
	int totalSize = sizeof(CheckPointDataMsg)+dataSize;
	char *msg = (char *)CmiAlloc(totalSize);
	CheckPointDataMsg *chkMsg = (CheckPointDataMsg *)msg;
	chkMsg->PE = CkMyPe();
	chkMsg->dataSize = dataSize;
	
	char *buf = &msg[sizeof(CheckPointDataMsg)];
	PUP::toMem pBuf(buf);

	pBuf | checkpointCount;
	for(int i=0; i<CmiNumPes(); i++){
		pBuf | CpvAccess(_incarnation)[i];
	}
	CkPupROData(pBuf);
	CkPupGroupData(pBuf,true);
	CkPupNodeGroupData(pBuf,true);
	pupArrayElementsSkip(pBuf,true,NULL);

	unAckedCheckpoint=1;
	CmiSetHandler(msg,_storeCheckpointHandlerIdx);
	CmiSyncSendAndFree(getCheckPointPE(),totalSize,msg);

#if DEBUG_CHECKPOINT
	if(CmiMyPe() == 0){
		printf("[%d] finishing checkpoint at %.6lf CmiTimer %.6lf with dataSize %d\n",CkMyPe(),CmiWallTimer(),CmiTimer(),dataSize);
	}
#endif

#if COLLECT_STATS_MEMORY
	CkPrintf("[%d] CKP=%d BUF_DET=%d STO_DET=%d MSG_LOG=%d\n",CkMyPe(),totalSize,bufferedDetsSize*sizeof(Determinant),storedDetsSize*sizeof(Determinant),msgLogSize);
	msgLogSize = 0;
	bufferedDetsSize = 0;
	storedDetsSize = 0;
#endif

};

/**
 * @brief Finishes checkpoint process by making the callback
 */
void _endCheckpointHandler(char *msg){
	CmiFree(msg);

	// making the callback to resume execution after checkpoint
	ckptCallback.send();
}

/**
 * @brief Starts the checkpoint phase after migration.
 */
void startMlogCheckpoint(void *_dummy, double curWallTime){
	double _startTime = CkWallTimer();

	// increasing the checkpoint counter
	checkpointCount++;
	
#if CMK_CONVERSE_MPI
	inCkptFlag = 1;
#endif

#if DEBUG_CHECKPOINT
	if(CmiMyPe() == 0){
		printf("[%d] starting checkpoint at %.6lf CmiTimer %.6lf \n",CkMyPe(),CmiWallTimer(),CmiTimer());
	}
#endif

	DEBUG_MEM(CmiMemoryCheck());

	PUP::sizer psizer;
	psizer | checkpointCount;
	for(int i=0; i<CmiNumPes(); i++){
		psizer | CpvAccess(_incarnation)[i];
	}
	CkPupROData(psizer);
	DEBUG_MEM(CmiMemoryCheck());
	CkPupGroupData(psizer,true);
	DEBUG_MEM(CmiMemoryCheck());
	CkPupNodeGroupData(psizer,true);
	DEBUG_MEM(CmiMemoryCheck());
	pupArrayElementsSkip(psizer,true,NULL);
	DEBUG_MEM(CmiMemoryCheck());

	int dataSize = psizer.size();
	int totalSize = sizeof(CheckPointDataMsg)+dataSize;
	char *msg = (char *)CmiAlloc(totalSize);
	CheckPointDataMsg *chkMsg = (CheckPointDataMsg *)msg;
	chkMsg->PE = CkMyPe();
	chkMsg->dataSize = dataSize;
	
	char *buf = &msg[sizeof(CheckPointDataMsg)];
	PUP::toMem pBuf(buf);

	pBuf | checkpointCount;
	for(int i=0; i<CmiNumPes(); i++){
		pBuf | CpvAccess(_incarnation)[i];
	}
	CkPupROData(pBuf);
	CkPupGroupData(pBuf,true);
	CkPupNodeGroupData(pBuf,true);
	pupArrayElementsSkip(pBuf,true,NULL);

	unAckedCheckpoint=1;
	CmiSetHandler(msg,_storeCheckpointHandlerIdx);
	CmiSyncSendAndFree(getCheckPointPE(),totalSize,msg);

#if DEBUG_CHECKPOINT
	if(CmiMyPe() == 0){
		printf("[%d] finishing checkpoint at %.6lf CmiTimer %.6lf with dataSize %d\n",CkMyPe(),CmiWallTimer(),CmiTimer(),dataSize);
	}
#endif

#if COLLECT_STATS_MEMORY
	CkPrintf("[%d] CKP=%d BUF_DET=%d STO_DET=%d MSG_LOG=%d\n",CkMyPe(),totalSize,bufferedDetsSize*sizeof(Determinant),storedDetsSize*sizeof(Determinant),msgLogSize);
	msgLogSize = 0;
	bufferedDetsSize = 0;
	storedDetsSize = 0;
#endif

	if(CkMyPe() ==  0 && onGoingLoadBalancing==0 ){
		lastCompletedAlarm = curWallTime;
		CcdCallFnAfter(checkpointAlarm,NULL,chkptPeriod);
	}
	traceUserBracketEvent(28,_startTime,CkWallTimer());
};


class ElementPacker : public CkLocIterator {
private:
	CkLocMgr *locMgr;
	PUP::er &p;
public:
	ElementPacker(CkLocMgr* mgr_, PUP::er &p_):locMgr(mgr_),p(p_){};
	void addLocation(CkLocation &loc) {
		CkArrayIndexMax idx=loc.getIndex();
		CkGroupID gID = locMgr->ckGetGroupID();
		p|gID;	    // store loc mgr's GID as well for easier restore
		p|idx;
		p|loc;
	}
};

/**
 * Pups all the array elements in this processor.
 */
void pupArrayElementsSkip(PUP::er &p, bool create, MigrationRecord *listToSkip,int listsize){
	int numElements,i;
	int numGroups = CkpvAccess(_groupIDTable)->size();	
	if(!p.isUnpacking()){
		numElements = CkCountArrayElements();
	}	
	//cppcheck-suppress uninitvar
	p | numElements;
	DEBUG(printf("[%d] Number of arrayElements %d \n",CkMyPe(),numElements));
	if(!p.isUnpacking()){
		CKLOCMGR_LOOP(ElementPacker packer(mgr, p); mgr->iterate(packer););
	}else{
		//Flush all recs of all LocMgrs before putting in new elements
//		CKLOCMGR_LOOP(mgr->flushAllRecs(););
		for(int j=0;j<listsize;j++){
			if(listToSkip[j].ackFrom == 0 && listToSkip[j].ackTo == 1){
				printf("[%d] Array element to be skipped gid %d idx",CmiMyPe(),listToSkip[j].gID.idx);
				listToSkip[j].idx.print();
			}
		}
		
		printf("numElements = %d\n",numElements);
	
		for (int i=0; i<numElements; i++) {
			CkGroupID gID;
			CkArrayIndexMax idx;
			p|gID;
	    	p|idx;
			int flag=0;
			int matchedIdx=0;
			for(int j=0;j<listsize;j++){
				if(listToSkip[j].ackFrom == 0 && listToSkip[j].ackTo == 1){
					if(listToSkip[j].gID == gID && listToSkip[j].idx == idx){
						matchedIdx = j;
						flag = 1;
						break;
					}
				}
			}
			if(flag == 1){
				printf("[%d] Array element being skipped gid %d idx %s\n",CmiMyPe(),gID.idx,idx2str(idx));
			}else{
				printf("[%d] Array element being recovered gid %d idx %s\n",CmiMyPe(),gID.idx,idx2str(idx));
			}
				
			CkLocMgr *mgr = (CkLocMgr*)CkpvAccess(_groupTable)->find(gID).getObj();
			CkPrintf("numLocalElements = %d\n",mgr->numLocalElements());
			mgr->resume(idx,p,create,flag);
			if(flag == 1){
				int homePE = mgr->homePe(idx);
				informLocationHome(gID,idx,homePE,listToSkip[matchedIdx].toPE);
			}
	  	}
	}
};

/**
 * Reads a checkpoint from disk. Assumes variable fName contains the name of the file.
 */
void readCheckpointFromDisk(int size, char *data){
	FILE *f = fopen(fName,"rb");
	fread(data,1,size,f);
	fclose(f);
}

/**
 * Writes a checkpoint to disk. Assumes variable fName contains the name of the file.
 */
void writeCheckpointToDisk(int size, char *data){
	FILE *f = fopen(fName,"wb");
	fwrite(data, 1, size, f);
	fclose(f);
}

//handler that receives the checkpoint from a processor
//it stores it and acks it
void _storeCheckpointHandler(char *msg){
	double _startTime=CkWallTimer();
		
	CheckPointDataMsg *chkMsg = (CheckPointDataMsg *)msg;
	DEBUG(printf("[%d] Checkpoint Data from %d stored with datasize %d\n",CkMyPe(),chkMsg->PE,chkMsg->dataSize);)
	
	char *chkpt = &msg[sizeof(CheckPointDataMsg)];	
	
	char *oldChkpt = 	CpvAccess(_storedCheckpointData)->buf;
	if(oldChkpt != NULL){
		char *oldmsg = oldChkpt - sizeof(CheckPointDataMsg);
		CmiFree(oldmsg);
	}
	
	int sendingPE = chkMsg->PE;
	
	CpvAccess(_storedCheckpointData)->buf = chkpt;
	CpvAccess(_storedCheckpointData)->bufSize = chkMsg->dataSize;
	CpvAccess(_storedCheckpointData)->PE = sendingPE;

	// storing checkpoint in disk
	if(diskCkptFlag){
		writeCheckpointToDisk(chkMsg->dataSize,chkpt);
		CpvAccess(_storedCheckpointData)->buf = NULL;
		CmiFree(msg);
	}

	CheckPointAck ackMsg;
	ackMsg.PE = CkMyPe();
	ackMsg.dataSize = CpvAccess(_storedCheckpointData)->bufSize;
	CmiSetHandler(&ackMsg,_checkpointAckHandlerIdx);
	CmiSyncSend(sendingPE,sizeof(CheckPointAck),(char *)&ackMsg);
	
	traceUserBracketEvent(29,_startTime,CkWallTimer());
};


void _checkpointAckHandler(CheckPointAck *ackMsg){
	DEBUG_MEM(CmiMemoryCheck());
	unAckedCheckpoint=0;
	DEBUGLB(printf("[%d] CheckPoint Acked from PE %d with size %d onGoingLoadBalancing %d \n",CkMyPe(),ackMsg->PE,ackMsg->dataSize,onGoingLoadBalancing));
	DEBUGLB(CkPrintf("[%d] ACK HANDLER with %d\n",CkMyPe(),onGoingLoadBalancing));	
	if(onGoingLoadBalancing) {
		onGoingLoadBalancing = 0;
		finishedCheckpointLoadBalancing();
	} else {
#if CMK_CONVERSE_MPI
	inCkptFlag = 0;
#endif
		char *msg = (char*)CmiAlloc(CmiMsgHeaderSizeBytes);
		CmiSetHandler(msg, _endCheckpointIdx);
		CmiReduce(msg,CmiMsgHeaderSizeBytes,doNothingMsg);
	}
	CmiFree(ackMsg);
};

/***************************************************************
	Restart Methods and handlers
***************************************************************/	

/**
 * Function for restarting the crashed processor.
 * It sets the restart flag and contacts the buddy
 * processor to get the latest checkpoint.
 */
void CkMlogRestart(const char * dummy, CkArgMsg * dummyMsg){
	RestartRequest msg;

	fprintf(stderr,"[%d] Restart started at %.6lf \n",CkMyPe(),CmiWallTimer());

	// setting the restart flag
	_restartFlag = 1;
	_numRestartResponses = 0;

	// requesting the latest checkpoint from its buddy
	msg.PE = CkMyPe();
	CmiSetHandler(&msg,_getCheckpointHandlerIdx);
	CmiSyncSend(getCheckPointPE(),sizeof(RestartRequest),(char *)&msg);
};

void CkMlogRestartDouble(void *,double){
	CkMlogRestart(NULL,NULL);
};

/**
 * Gets the stored checkpoint for its buddy processor.
 */
void _getCheckpointHandler(RestartRequest *restartMsg){

	// retrieving the stored checkpoint
	StoredCheckpoint *storedChkpt =	CpvAccess(_storedCheckpointData);

	// making sure it is its buddy who is requesting the checkpoint
	CkAssert(restartMsg->PE == storedChkpt->PE);

	int totalSize = sizeof(RestartProcessorData) + storedChkpt->bufSize;
	
	DEBUG_RESTART(CkPrintf("[%d] Sending out checkpoint for processor %d size %d \n",CkMyPe(),restartMsg->PE,totalSize);)
	CkPrintf("[%d] Sending out checkpoint for processor %d size %d \n",CkMyPe(),restartMsg->PE,totalSize);
	
	char *msg = (char *)CmiAlloc(totalSize);
	
	RestartProcessorData *dataMsg = (RestartProcessorData *)msg;
	dataMsg->PE = CkMyPe();
	dataMsg->restartWallTime = CmiTimer();
	dataMsg->checkPointSize = storedChkpt->bufSize;
	dataMsg->lbGroupID = globalLBID;
	
	//store checkpoint data
	char *buf = &msg[sizeof(RestartProcessorData)];

	if(diskCkptFlag){
		readCheckpointFromDisk(storedChkpt->bufSize,buf);
	} else {
		memcpy(buf,storedChkpt->buf,storedChkpt->bufSize);
	}

	CmiSetHandler(msg,_recvCheckpointHandlerIdx);
	CmiSyncSendAndFree(restartMsg->PE,totalSize,msg);
	CmiFree(restartMsg);
}

// this list is used to create a vector of the object ids of all
//the chares on this processor currently and the highest TN processed by them 
//the first argument is actually a CkVec<TProcessedLog> *
void createObjIDList(void *data, ChareMlogData *mlogData){
	CkVec<CkObjID> *list = (CkVec<CkObjID> *)data;
	CkObjID entry;
	entry = mlogData->objID;
	list->push_back(entry);
	DEBUG_RECOVERY(printLog(&entry));
}


/**
 * Receives the checkpoint data from its buddy, restores the state of all the objects
 * and asks everyone else to update its home.
 */
void _recvCheckpointHandler(char *_restartData){
	RestartProcessorData *restartData = (RestartProcessorData *)_restartData;

	globalLBID = restartData->lbGroupID;
	
	printf("[%d] Restart Checkpointdata received from PE %d at %.6lf with checkpointSize %d\n",CkMyPe(),restartData->PE,CmiWallTimer(),restartData->checkPointSize);
	char *buf = &_restartData[sizeof(RestartProcessorData)];
	
	PUP::fromMem pBuf(buf);

	pBuf | checkpointCount;
	for(int i=0; i<CmiNumPes(); i++){
		pBuf | CpvAccess(_incarnation)[i];
	}
	CkPupROData(pBuf);
	CkPupGroupData(pBuf,true);
	CkPupNodeGroupData(pBuf,true);
	pupArrayElementsSkip(pBuf,true,NULL);
	CkAssert(pBuf.size() == restartData->checkPointSize);
	printf("[%d] Restart Objects created from CheckPointData at %.6lf \n",CkMyPe(),CmiWallTimer());

	// increases the incarnation number
	CpvAccess(_incarnation)[CmiMyPe()]++;
	
	forAllCharesDo(initializeRestart,NULL);
	
	CmiFree(_restartData);
	
	_initDone();

	getGlobalStep(globalLBID);
}


/**
 * @brief Initializes variables and flags for restarting procedure.
 */
void initializeRestart(void *data, ChareMlogData *mlogData){
	mlogData->resendReplyRecvd = 0;
	mlogData->restartFlag = 1;
};

/**
 * Updates the homePe of chare array elements.
 */
void updateHomePE(void *data,ChareMlogData *mlogData){
	RestartRequest *updateRequest = (RestartRequest *)data;
	int PE = updateRequest->PE; //restarted PE
	//if this object is an array Element and its home is the restarted processor
	// the home processor needs to know its current location
	if(mlogData->objID.type == TypeArray){
		//it is an array element
		CkGroupID myGID = mlogData->objID.data.array.id;
		CkArrayIndexMax myIdx =  mlogData->objID.data.array.idx.asChild();
		CkArrayID aid(mlogData->objID.data.array.id);		
		//check if the restarted processor is the home processor for this object
		CkLocMgr *locMgr = aid.ckLocalBranch()->getLocMgr();
		if(locMgr->homePe(myIdx) == PE){
			DEBUG_RESTART(printf("[%d] Tell %d of current location of array element",CkMyPe(),PE));
			DEBUG_RESTART(myIdx.print());
			informLocationHome(locMgr->getGroupID(),myIdx,PE,CkMyPe());
		}
	}
};

/**
 * Prints a processed log.
 */
void printLog(CkObjID &recver){
	char recverString[100];
	CkPrintf("[RECOVERY] [%d] OBJECT=\"%s\" \n",CkMyPe(),recver.toString(recverString));
}

/**
 * Prints information about a message.
 */
void printMsg(envelope *env, const char* par){
	char senderString[100];
	char recverString[100];
	CkPrintf("[RECOVERY] [%d] MSG-%s FROM=\"%s\" TO=\"%s\" SN=%d\n",CkMyPe(),par,env->sender.toString(senderString),env->recver.toString(recverString),env->SN);
}

/**
 * @brief Resends all the logged messages to a particular chare list.
 * @param data is of type ResendData which contains the array of objects on  the restartedProcessor.
 * @param mlogData a particular chare living in this processor.
 */
void resendMessageForChare(void *data, ChareMlogData *mlogData){
	DEBUG_RESTART(char nameString[100]);
	DEBUG_RESTART(char recverString[100]);
	DEBUG_RESTART(char senderString[100]);

	ResendData *resendData = (ResendData *)data;
	int PE = resendData->PE; //restarted PE
	int count=0;
	int ticketRequests=0;
	CkQ<MlogEntry *> *log = mlogData->getMlog();

	DEBUG_RESTART(printf("[%d] Resend message from %s to processor %d \n",CkMyPe(),mlogData->objID.toString(nameString),PE);)

	// traversing the message log to see if we must resend a message	
	for(int i=0;i<log->length();i++){
		MlogEntry *logEntry = (*log)[i];
		
		// if we sent out the logs of a local message to buddy and it crashed
		//before acknowledging 
		envelope *env = logEntry->env;
		if(env == NULL){
			continue;
		}
	
		// resend if type is not invalid	
		if(env->recver.type != TypeInvalid){
			for(int j=0;j<resendData->numberObjects;j++){
				if(env->recver == (resendData->listObjects)[j]){
					if(PE != CkMyPe()){
						DEBUG_RECOVERY(printMsg(env,RECOVERY_SEND));
						if(env->recver.type == TypeNodeGroup){
							CmiSyncNodeSend(PE,env->getTotalsize(),(char *)env);
						}else{
							CmiSetHandler(env,CmiGetXHandler(env));
							CmiSyncSend(PE,env->getTotalsize(),(char *)env);
						}
					}else{
						envelope *copyEnv = copyEnvelope(env);
						CqsEnqueueGeneral((Queue)CpvAccess(CsdSchedQueue),copyEnv, copyEnv->getQueueing(),copyEnv->getPriobits(),(unsigned int *)copyEnv->getPrioPtr());
					}
					DEBUG_RESTART(printf("[%d] Resent message sender %s recver %s SN %d TN %d \n",CkMyPe(),env->sender.toString(senderString),env->recver.toString(nameString),env->SN,env->TN));
					count++;
				}
			}//end of for loop of objects
			
		}	
	}
	DEBUG_RESTART(printf("[%d] Resent  %d/%d (%d) messages  from %s to processor %d \n",CkMyPe(),count,log->length(),ticketRequests,mlogData->objID.toString(nameString),PE);)	
}

/**
 * Resends messages since last checkpoint to the list of objects included in the 
 * request. It also sends stored remote determinants to the particular failed PE.
 */
void _resendMessagesHandler(char *msg){
	ResendData d;
	ResendRequest *resendReq = (ResendRequest *)msg;

	// cleaning global reduction sequence number
	CmiResetGlobalReduceSeqID();

	// building the reply message
	char *listObjects = &msg[sizeof(ResendRequest)];
	d.numberObjects = resendReq->numberObjects;
	d.PE = resendReq->PE;
	d.listObjects = (CkObjID *)listObjects;
	
	DEBUG(printf("[%d] Received request to Resend Messages to processor %d numberObjects %d at %.6lf\n",CkMyPe(),resendReq->PE,resendReq->numberObjects,CmiWallTimer()));

	// resends messages for the list of objects
	forAllCharesDo(resendMessageForChare,&d);

	DEBUG_MEM(CmiMemoryCheck());

	CmiFree(msg);
}

/*
	Method to do parallel restart. Distribute some of the array elements to other processors.
	The problem is that we cant use to charm entry methods to do migration as it will get
	stuck in the protocol that is going to restart
	Note: in order to avoid interference between the objects being recovered, the current PE
    will NOT keep any object. It will be devoted to forward the messages to recovering objects.    Otherwise, the current PE has to do both things, recover objects and forward messages and 
    objects end up stepping into each other's shoes (interference).
*/

class ElementDistributor: public CkLocIterator{
	CkLocMgr *locMgr;
	int *targetPE;

	void pupLocation(CkLocation &loc,PUP::er &p){
		CkArrayIndexMax idx=loc.getIndex();
		CkGroupID gID = locMgr->ckGetGroupID();
		p|gID;	    // store loc mgr's GID as well for easier restore
		p|idx;
		p|loc;
	};
public:
	ElementDistributor(CkLocMgr *mgr_,int *toPE_):locMgr(mgr_),targetPE(toPE_){};

	void addLocation(CkLocation &loc){

		// leaving object on this PE
		if(*targetPE == CkMyPe()){
			*targetPE = (*targetPE +1)%CkNumPes();
			return;
		}
			
		CkArrayIndexMax idx = loc.getIndex();
		CkLocRec *rec = loc.getLocalRecord();
		CkLocMgr *locMgr = loc.getManager();
		CkVec<CkMigratable *> eltList;
			
		CkPrintf("[%d] Distributing objects to Processor %d: ",CkMyPe(),*targetPE);
		idx.print();

		// incrementing number of emigrant objects
		CpvAccess(_numEmigrantRecObjs)++;
    	locMgr->migratableList(rec,eltList);
		CkReductionMgr *reductionMgr = (CkReductionMgr*)CkpvAccess(_groupTable)->find(eltList[0]->mlogData->objID.data.array.id).getObj();
		
		// let everybody else know the object is leaving
		locMgr->callMethod(rec,&CkMigratable::ckAboutToMigrate);
		reductionMgr->incNumEmigrantRecObjs();
	
		//pack up this location and send it across
		PUP::sizer psizer;
		pupLocation(loc,psizer);
		int totalSize = psizer.size() + sizeof(DistributeObjectMsg);
		char *msg = (char *)CmiAlloc(totalSize);
		DistributeObjectMsg *distributeMsg = (DistributeObjectMsg *)msg;
		distributeMsg->PE = CkMyPe();
		char *buf = &msg[sizeof(DistributeObjectMsg)];
		PUP::toMem pmem(buf);
		pmem.becomeDeleting();
		pupLocation(loc,pmem);
			
		locMgr->setDuringMigration(true);
		delete rec;
		locMgr->setDuringMigration(false);
		locMgr->inform(idx,*targetPE);

		CmiSetHandler(msg,_distributedLocationHandlerIdx);
		CmiSyncSendAndFree(*targetPE,totalSize,msg);

		CmiAssert(locMgr->lastKnown(idx) == *targetPE);

		//decide on the target processor for the next object
		*targetPE = *targetPE + 1;
		if(*targetPE > (CkMyPe() + parallelRecovery)){
			*targetPE = CkMyPe() + 1;
		}
	}

};

/**
 * Distributes objects to accelerate recovery after a failure.
 */
void distributeRestartedObjects(){
	int numGroups = CkpvAccess(_groupIDTable)->size();	
	int i;
	int targetPE=CkMyPe()+1;
	CKLOCMGR_LOOP(ElementDistributor distributor(mgr,&targetPE);mgr->iterate(distributor););
};

/**
 * Handler to receive back a location.
 */
void _sendBackLocationHandler(char *receivedMsg){
	printf("Array element received at processor %d after recovery\n",CkMyPe());
	DistributeObjectMsg *distributeMsg = (DistributeObjectMsg *)receivedMsg;
	int sourcePE = distributeMsg->PE;
	char *buf = &receivedMsg[sizeof(DistributeObjectMsg)];
	PUP::fromMem pmem(buf);
	CkGroupID gID;
	CkArrayIndexMax idx;
	pmem |gID;
	pmem |idx;
	CkLocMgr *mgr = (CkLocMgr*)CkpvAccess(_groupTable)->find(gID).getObj();
	donotCountMigration=1;
	mgr->resume(idx,pmem,true);
	donotCountMigration=0;
	informLocationHome(gID,idx,mgr->homePe(idx),CkMyPe());
	printf("Array element inserted at processor %d after parallel recovery\n",CkMyPe());
	idx.print();

	// decrementing number of emigrant objects at reduction manager
	CkVec<CkMigratable *> eltList;
	CkLocRec *rec = mgr->elementRec(idx);
	mgr->migratableList(rec,eltList);
	CkReductionMgr *reductionMgr = (CkReductionMgr*)CkpvAccess(_groupTable)->find(eltList[0]->mlogData->objID.data.array.id).getObj();
	reductionMgr->decNumEmigrantRecObjs();
	reductionMgr->decGCount();

	// checking if it has received all emigrant recovering objects
	CpvAccess(_numEmigrantRecObjs)--;
	if(CpvAccess(_numEmigrantRecObjs) == 0){
		(*resumeLbFnPtr)(centralLb);
	}

}

/**
 * Handler to update information about an object just received.
 */
void _distributedLocationHandler(char *receivedMsg){
	printf("Array element received at processor %d after distribution at restart\n",CkMyPe());
	DistributeObjectMsg *distributeMsg = (DistributeObjectMsg *)receivedMsg;
	int sourcePE = distributeMsg->PE;
	char *buf = &receivedMsg[sizeof(DistributeObjectMsg)];
	PUP::fromMem pmem(buf);
	CkGroupID gID;
	CkArrayIndexMax idx;
	pmem |gID;
	pmem |idx;
	CkLocMgr *mgr = (CkLocMgr*)CkpvAccess(_groupTable)->find(gID).getObj();
	donotCountMigration=1;
	mgr->resume(idx,pmem,true);
	donotCountMigration=0;
	informLocationHome(gID,idx,mgr->homePe(idx),CkMyPe());
	printf("Array element inserted at processor %d after distribution at restart ",CkMyPe());
	idx.print();

	CkLocRec *rec = mgr->elementRec(idx);
	CmiAssert(rec != NULL);

	// adding object to the list of immigrant recovery objects
	CpvAccess(_immigrantRecObjs)->push_back(new CkLocation(mgr,rec));
	CpvAccess(_numImmigrantRecObjs)++;
	
	CkVec<CkMigratable *> eltList;
	mgr->migratableList((CkLocRec *)rec,eltList);
	for(int i=0;i<eltList.size();i++){
		if(eltList[i]->mlogData->toResumeOrNot == 1 && eltList[i]->mlogData->resumeCount < globalResumeCount){
			CpvAccess(_currentObj) = eltList[i];
			eltList[i]->mlogData->immigrantRecFlag = 1;
			eltList[i]->mlogData->immigrantSourcePE = sourcePE;

			// incrementing immigrant counter at reduction manager
			CkReductionMgr *reductionMgr = (CkReductionMgr*)CkpvAccess(_groupTable)->find(eltList[i]->mlogData->objID.data.array.id).getObj();
			reductionMgr->incNumImmigrantRecObjs();
			reductionMgr->decGCount();

			eltList[i]->ResumeFromSync();
		}
	}
}


/** this method is used to send messages to a restarted processor to tell
 * it that a particular expected object is not going to get to it */
void sendDummyMigration(int restartPE,CkGroupID lbID,CkGroupID locMgrID,CkArrayIndexMax &idx,int locationPE){
	DummyMigrationMsg buf;
	buf.flag = MLOG_OBJECT;
	buf.lbID = lbID;
	buf.mgrID = locMgrID;
	buf.idx = idx;
	buf.locationPE = locationPE;
	CmiSetHandler(&buf,_dummyMigrationHandlerIdx);
	CmiSyncSend(restartPE,sizeof(DummyMigrationMsg),(char *)&buf);
};


/**this method is used by a restarted processor to tell other processors
 * that they are not going to receive these many objects.. just the count
 * not the objects themselves ***/

void sendDummyMigrationCounts(int *dummyCounts){
	DummyMigrationMsg buf;
	buf.flag = MLOG_COUNT;
	buf.lbID = globalLBID;
	CmiSetHandler(&buf,_dummyMigrationHandlerIdx);
	for(int i=0;i<CmiNumPes();i++){
		if(i != CmiMyPe() && dummyCounts[i] != 0){
			buf.count = dummyCounts[i];
			CmiSyncSend(i,sizeof(DummyMigrationMsg),(char *)&buf);
		}
	}
}


/** this handler is used to process a dummy migration msg.
 * it looks up the load balancer and calls migrated for it */

void _dummyMigrationHandler(DummyMigrationMsg *msg){
	CentralLB *lb = (CentralLB *)CkpvAccess(_groupTable)->find(msg->lbID).getObj();
	if(msg->flag == MLOG_OBJECT){
		DEBUG_RESTART(CmiPrintf("[%d] dummy Migration received from pe %d for %d:%s \n",CmiMyPe(),msg->locationPE,msg->mgrID.idx,idx2str(msg->idx)));
		LDObjHandle h;
		lb->Migrated(h,1);
	}
	if(msg->flag == MLOG_COUNT){
		DEBUG_RESTART(CmiPrintf("[%d] dummyMigration count %d received from restarted processor\n",CmiMyPe(),msg->count));
		for(int i=0;i<msg->count;i++){
			LDObjHandle h;
			lb->Migrated(h,1);
		}
	}
	CmiFree(msg);
};

/*****************************************************
	Implementation of a method that can be used to call
	any method on the ChareMlogData of all the chares on
	a processor currently
******************************************************/


class ElementCaller :  public CkLocIterator {
private:
	CkLocMgr *locMgr;
	MlogFn fnPointer;
	void *data;
public:
	ElementCaller(CkLocMgr * _locMgr, MlogFn _fnPointer,void *_data){
		locMgr = _locMgr;
		fnPointer = _fnPointer;
		data = _data;
	};
	void addLocation(CkLocation &loc){
		CkVec<CkMigratable *> list;
		CkLocRec *local = loc.getLocalRecord();
		locMgr->migratableList (local,list);
		for(int i=0;i<list.size();i++){
			CkMigratable *migratableElement = list[i];
			fnPointer(data,migratableElement->mlogData);
		}
	}
};

/**
 * Map function pointed by fnPointer over all the chares living in this processor.
 */
void forAllCharesDo(MlogFn fnPointer, void *data){
	int numGroups = CkpvAccess(_groupIDTable)->size();
	for(int i=0;i<numGroups;i++){
		Chare *obj = (Chare *)CkpvAccess(_groupTable)->find((*CkpvAccess(_groupIDTable))[i]).getObj();
		fnPointer(data,obj->mlogData);
	}
	int numNodeGroups = CksvAccess(_nodeGroupIDTable).size();
	for(int i=0;i<numNodeGroups;i++){
		Chare *obj = (Chare *)CksvAccess(_nodeGroupTable)->find(CksvAccess(_nodeGroupIDTable)[i]).getObj();
		fnPointer(data,obj->mlogData);
	}
	int i;
	CKLOCMGR_LOOP(ElementCaller caller(mgr, fnPointer,data); mgr->iterate(caller););
};


/******************************************************************
 Load Balancing
******************************************************************/

/**
 * This is the first time Converse is called after AtSync method has been called by every local object.
 * It is a good place to insert some optimizations for synchronized checkpoint. In the case of causal
 * message logging, we can take advantage of this situation and garbage collect at this point.
 */
void initMlogLBStep(CkGroupID gid){
	DEBUGLB(CkPrintf("[%d] INIT MLOG STEP\n",CkMyPe()));
	countLBMigratedAway = 0;
	countLBToMigrate=0;
	onGoingLoadBalancing=1;
	migrationDoneCalled=0;
	checkpointBarrierCount=0;
	if(globalLBID.idx != 0){
		CmiAssert(globalLBID.idx == gid.idx);
	}
	globalLBID = gid;
#if SYNCHRONIZED_CHECKPOINT
	garbageCollectMlog();
#endif
}

/**
 * Pups a location
 */
void pupLocation(CkLocation *loc, CkLocMgr *locMgr, PUP::er &p){
	CkArrayIndexMax idx = loc->getIndex();
	CkGroupID gID = locMgr->ckGetGroupID();
	p|gID;	    // store loc mgr's GID as well for easier restore
	p|idx;
	p|*loc;
};

/**
 * Sends back the immigrant recovering object to their origin PE.
 */
void sendBackImmigrantRecObjs(){
	CkLocation *loc;
	CkLocMgr *locMgr;
	CkArrayIndexMax idx;
	CkLocRec *rec;
	PUP::sizer psizer;
	int targetPE;
	CkVec<CkMigratable *> eltList;
	CkReductionMgr *reductionMgr;
 
	// looping through all elements in immigrant recovery objects vector
	for(int i=0; i<CpvAccess(_numImmigrantRecObjs); i++){

		// getting the components of each location
		loc = (*CpvAccess(_immigrantRecObjs))[i];
		idx = loc->getIndex();
		rec = loc->getLocalRecord();
		locMgr = loc->getManager();
    	locMgr->migratableList(rec,eltList);
		targetPE = eltList[i]->mlogData->immigrantSourcePE;

		// decrement counter at array manager
		reductionMgr = (CkReductionMgr*)CkpvAccess(_groupTable)->find(eltList[i]->mlogData->objID.data.array.id).getObj();
		reductionMgr->decNumImmigrantRecObjs();

		CkPrintf("[%d] Sending back object to %d: ",CkMyPe(),targetPE);
		idx.print();

		// let everybody else know the object is leaving
		locMgr->callMethod(rec,&CkMigratable::ckAboutToMigrate);
			
		//pack up this location and send it across
		pupLocation(loc,locMgr,psizer);
		int totalSize = psizer.size() + sizeof(DistributeObjectMsg);
		char *msg = (char *)CmiAlloc(totalSize);
		DistributeObjectMsg *distributeMsg = (DistributeObjectMsg *)msg;
		distributeMsg->PE = CkMyPe();
		char *buf = &msg[sizeof(DistributeObjectMsg)];
		PUP::toMem pmem(buf);
		pmem.becomeDeleting();
		pupLocation(loc,locMgr,pmem);
		
		locMgr->setDuringMigration(true);
		delete rec;
		locMgr->setDuringMigration(false);
		locMgr->inform(idx,targetPE);

		// sending the object
		CmiSetHandler(msg,_sendBackLocationHandlerIdx);
		CmiSyncSendAndFree(targetPE,totalSize,msg);

		// freeing memory
		delete loc;

		CmiAssert(locMgr->lastKnown(idx) == targetPE);
		
	}

	// cleaning up all data structures
	CpvAccess(_immigrantRecObjs)->removeAll();
	CpvAccess(_numImmigrantRecObjs) = 0;

}

/**
 * Restores objects after parallel recovery, either by sending back the immigrant objects or 
 * by waiting for all emigrant objects to be back.
 */
void restoreParallelRecovery(void (*_fnPtr)(void *),void *_centralLb){
	resumeLbFnPtr = _fnPtr;
	centralLb = _centralLb;

	// sending back the immigrant recovering objects
	if(CpvAccess(_numImmigrantRecObjs) > 0){
		sendBackImmigrantRecObjs();	
	}

	// checking whether it needs to wait for emigrant recovery objects
	if(CpvAccess(_numEmigrantRecObjs) > 0)
		return;

	// otherwise, load balancing process is finished
	(*resumeLbFnPtr)(centralLb);
}

void startLoadBalancingMlog(void (*_fnPtr)(void *),void *_centralLb){
	DEBUGLB(printf("[%d] start Load balancing section of message logging \n",CmiMyPe()));
	DEBUG_TEAM(printf("[%d] start Load balancing section of message logging \n",CmiMyPe()));

	resumeLbFnPtr = _fnPtr;
	centralLb = _centralLb;
	migrationDoneCalled = 1;
	if(countLBToMigrate == countLBMigratedAway){
		DEBUGLB(printf("[%d] calling startMlogCheckpoint in startLoadBalancingMlog countLBToMigrate %d countLBMigratedAway %d \n",CmiMyPe(),countLBToMigrate,countLBMigratedAway));
		startMlogCheckpoint(NULL,CmiWallTimer());
	}
};

void finishedCheckpointLoadBalancing(){
      DEBUGLB(printf("[%d] finished checkpoint after lb \n",CmiMyPe());)
  
      char *msg = (char*)CmiAlloc(CmiMsgHeaderSizeBytes);
      CmiSetHandler(msg,_checkpointBarrierHandlerIdx);
      CmiReduce(msg,CmiMsgHeaderSizeBytes,doNothingMsg);
};

void _receiveMlogLocationHandler(void *buf){
	envelope *env = (envelope *)buf;
	DEBUG(printf("[%d] Location received in message of size %d\n",CkMyPe(),env->getTotalsize()));
	CkUnpackMessage(&env);
	void *_msg = EnvToUsr(env);
	CkArrayElementMigrateMessage *msg = (CkArrayElementMigrateMessage *)_msg;
	CkGroupID gID= msg->gid;
	DEBUG(printf("[%d] Object to be inserted into location manager %d\n",CkMyPe(),gID.idx));
	CkLocMgr *mgr = (CkLocMgr*)CkpvAccess(_groupTable)->find(gID).getObj();
	CpvAccess(_currentObj)=mgr;
	mgr->immigrate(msg);
};

void _checkpointBarrierHandler(CheckpointBarrierMsg *barrierMsg){

	// deleting the received message
	CmiFree(barrierMsg);

	// sending a broadcast to resume execution
	CheckpointBarrierMsg *msg = (CheckpointBarrierMsg *)CmiAlloc(sizeof(CheckpointBarrierMsg));
	CmiSetHandler(msg,_checkpointBarrierAckHandlerIdx);
	CmiSyncBroadcastAllAndFree(sizeof(CheckpointBarrierMsg),(char *)msg);
}

void _checkpointBarrierAckHandler(CheckpointBarrierMsg *msg){
	DEBUG(CmiPrintf("[%d] _checkpointBarrierAckHandler \n",CmiMyPe()));
	DEBUGLB(CkPrintf("[%d] Reaching this point\n",CkMyPe()));

#if CMK_CONVERSE_MPI
	inCkptFlag = 0;
#endif

	// resuming LB function pointer
	(*resumeLbFnPtr)(centralLb);

	// deleting message
	CmiFree(msg);
}

/**
 * @brief Function to remove all messages in the message log of a particular chare.
 */
void garbageCollectMlogForChare(void *data, ChareMlogData *mlogData){
	int total;
	MlogEntry *logEntry;
	CkQ<MlogEntry *> *mlog = mlogData->getMlog();

	// traversing the whole message log and removing all elements
	total = mlog->length();
	for(int i=0; i<total; i++){
		logEntry = mlog->deq();
		delete logEntry;
	}

}

/**
 * @brief Garbage collects the message log and other data structures.
 */
void garbageCollectMlog(){
	DEBUG(CkPrintf("[%d] Garbage collecting message log and data structures\n", CkMyPe()));

	// removing all messages in message log for every chare
	forAllCharesDo(garbageCollectMlogForChare, NULL);
}

/**
	method that informs an array elements home processor of its current location
	It is a converse method to bypass the charm++ message logging framework
*/

void informLocationHome(CkGroupID locMgrID,CkArrayIndexMax idx,int homePE,int currentPE){
	double _startTime = CmiWallTimer();
	CurrentLocationMsg msg;
	msg.mgrID = locMgrID;
	msg.idx = idx;
	msg.locationPE = currentPE;
	msg.fromPE = CkMyPe();

	DEBUG(CmiPrintf("[%d] informing home %d of location %d of gid %d idx %s \n",CmiMyPe(),homePE,currentPE,locMgrID.idx,idx2str(idx)));
	CmiSetHandler(&msg,_receiveLocationHandlerIdx);
	CmiSyncSend(homePE,sizeof(CurrentLocationMsg),(char *)&msg);
	traceUserBracketEvent(37,_startTime,CmiWallTimer());
}


void _receiveLocationHandler(CurrentLocationMsg *data){
	double _startTime = CmiWallTimer();
	CkLocMgr *mgr =  (CkLocMgr*)CkpvAccess(_groupTable)->find(data->mgrID).getObj();
	if(mgr == NULL){
		CmiFree(data);
		return;
	}
	CkLocRec *rec = mgr->elementNrec(data->idx);
	DEBUG(CmiPrintf("[%d] location from %d is %d for gid %d idx %s rec %p \n",CkMyPe(),data->fromPE,data->locationPE,data->mgrID,idx2str(data->idx),rec));
	if(rec != NULL){
		if(mgr->lastKnown(data->idx) == CmiMyPe() && data->locationPE != CmiMyPe()){
			if(data->fromPE == data->locationPE){
				CmiAbort("Another processor has the same object");
			}
		}
	}
	if(rec!= NULL && data->fromPE != CmiMyPe()){
		int targetPE = data->fromPE;
		data->fromPE = CmiMyPe();
		data->locationPE = CmiMyPe();
		DEBUG(printf("[%d] WARNING!! informing proc %d of current location\n",CmiMyPe(),targetPE));
		CmiSyncSend(targetPE,sizeof(CurrentLocationMsg),(char *)data);
	}else{
		mgr->inform(data->idx,data->locationPE);
	}
	CmiFree(data);
	traceUserBracketEvent(38,_startTime,CmiWallTimer());
}



void getGlobalStep(CkGroupID gID){
	LBStepMsg msg;
	int destPE = 0;
	msg.lbID = gID;
	msg.fromPE = CmiMyPe();
	msg.step = -1;
	CmiSetHandler(&msg,_getGlobalStepHandlerIdx);
	CmiSyncSend(destPE,sizeof(LBStepMsg),(char *)&msg);
};

void _getGlobalStepHandler(LBStepMsg *msg){
	CentralLB *lb;
	if(msg->lbID.idx != 0){
		lb = (CentralLB *)CkpvAccess(_groupTable)->find(msg->lbID).getObj();
		msg->step = lb->step();
	} else {
		msg->step = -1;
	}
	CmiAssert(msg->fromPE != CmiMyPe());
	CmiPrintf("[%d] getGlobalStep called from %d step %d gid %d \n",CmiMyPe(),msg->fromPE,msg->step,msg->lbID.idx);
	CmiSetHandler(msg,_recvGlobalStepHandlerIdx);
	CmiSyncSend(msg->fromPE,sizeof(LBStepMsg),(char *)msg);
};

/**
 * @brief Receives the global step handler from PE 0
 */
void _recvGlobalStepHandler(LBStepMsg *msg){
	
	// updating restart decision number
	restartDecisionNumber = msg->step;
	CmiFree(msg);

	CmiPrintf("[%d] recvGlobalStepHandler \n",CmiMyPe());

	// continuing with restart process; send out the request to resend logged messages to all other processors
	CkVec<CkObjID> objectVec;
	forAllCharesDo(createObjIDList, (void *)&objectVec);
	int numberObjects = objectVec.size();
	
	//	resendMsg layout: |ResendRequest|Array of CkObjID|
	int totalSize = sizeof(ResendRequest) + numberObjects * sizeof(CkObjID);
	char *resendMsg = (char *)CmiAlloc(totalSize);	

	ResendRequest *resendReq = (ResendRequest *)resendMsg;
	resendReq->PE = CkMyPe(); 
	resendReq->numberObjects = numberObjects;
	char *objList = &resendMsg[sizeof(ResendRequest)];
	memcpy(objList,objectVec.getVec(),numberObjects * sizeof(CkObjID));	

	if(restartDecisionNumber != -1){
		CentralLB *lb = (CentralLB *)CkpvAccess(_groupTable)->find(globalLBID).getObj();
		CpvAccess(_currentObj) = lb;
		lb->ReceiveDummyMigration(restartDecisionNumber);
	}

	CmiSetHandler(resendMsg,_resendMessagesHandlerIdx);
	CmiSyncBroadcastAllAndFree(totalSize, resendMsg);

	/* test for parallel restart migrate away object**/
	if(fastRecovery){
		distributeRestartedObjects();
		printf("[%d] Redistribution of objects done at %.6lf \n",CkMyPe(),CmiWallTimer());
	}

};

/**
 * @brief Function to wrap up performance information.
 */
void _messageLoggingExit(){
	
	// printing the signature for causal message logging
	if(CkMyPe() == 0)
		printf("[%d] FastMessageLoggingExit \n",CmiMyPe());

#if COLLECT_STATS_MSGS
#if COLLECT_STATS_MSGS_TOTAL
	printf("[%d] TOTAL MESSAGES SENT: %d\n",CmiMyPe(),totalMsgsTarget);
	printf("[%d] TOTAL MESSAGES SENT SIZE: %.2f MB\n",CmiMyPe(),totalMsgsSize/(float)MEGABYTE);
#else
	printf("[%d] TARGETS: ",CmiMyPe());
	for(int i=0; i<CmiNumPes(); i++){
#if COLLECT_STATS_MSG_COUNT
		printf("%d ",numMsgsTarget[i]);
#else
		printf("%d ",sizeMsgsTarget[i]);
#endif
	}
	printf("\n");
#endif
#endif
}

void MlogEntry::pup(PUP::er &p){
	p | destPE;
	p | _infoIdx;
	int size;
	if(!p.isUnpacking()){
/*		CkAssert(env);
		if(!env->isPacked()){
			CkPackMessage(&env);
		}*/
		if(env == NULL){
			//message was probably local and has been removed from logs
			size = 0;
		}else{
			size = env->getTotalsize();
		}	
	}
	//cppcheck-suppress uninitvar
	p | size;
	if(p.isUnpacking()){
		if(size > 0){
			env = (envelope *)_allocEnv(ForChareMsg,size);
		}else{
			env = NULL;
		}
	}
	if(size > 0){
		p((char *)env,size);
	}
};


/**********************************
	* The methods of the message logging
	* data structure stored in each chare
	********************************/

MCount ChareMlogData::nextSN(const CkObjID &recver){
	MCount *SN = ssnTable.getPointer(recver);
	if(SN==NULL){
		ssnTable.put(recver) = 1;
		return 1;
	}else{
		(*SN)++;
		return *SN;
	}
};
 
/**
 * Adds an entry into the message log.
 */
void ChareMlogData::addLogEntry(MlogEntry *entry){
	DEBUG(char nameString[100]);
	DEBUG(printf("[%d] Adding logEntry %p to the log of %s with SN %d\n",CkMyPe(),entry,objID.toString(nameString),entry->env->SN));
	DEBUG_MEM(CmiMemoryCheck());

	// enqueuing the entry in the message log
	mlog.enq(entry);
};

/**
 * Checks whether a ssn has been already received. The collateral effect is the ssn get added to the list.
 */
int ChareMlogData::checkAndStoreSsn(const CkObjID &sender, MCount ssn){
	RSSN *rssn;
	rssn = receivedSsnTable.get(sender);
	if(rssn == NULL){
		rssn = new RSSN();
		receivedSsnTable.put(sender) = rssn;
	}
	return rssn->checkAndStore(ssn);
}

/**
 * Pup method for the metadata.
 * We are preventing the whole message log to be stored (as proposed by Sayantan for dealing with multiple failures).
 * Then, we only support one failure at a time. Read Sayantan's thesis, sections 4.2 and 4.3 for more details.
 */
void ChareMlogData::pup(PUP::er &p){
	int startSize=0;
	char nameStr[100];
	if(p.isSizing()){
		PUP::sizer *sizep = (PUP::sizer *)&p;
		startSize = sizep->size();
	}
	p | objID;
	if(p.isUnpacking()){
		DEBUG(CmiPrintf("[%d] Obj %s being unpacked with tCount %d tProcessed %d \n",CmiMyPe(),objID.toString(nameStr),tCount,tProcessed));
	}
	p | toResumeOrNot;
	p | resumeCount;
	DEBUG(CmiPrintf("[%d] Obj %s toResumeOrNot %d resumeCount %d \n",CmiMyPe(),objID.toString(nameStr),toResumeOrNot,resumeCount));
	
	ssnTable.pup(p);
	
	// pupping receivedSsnTable
	int rssnTableSize;
	if(!p.isUnpacking()){
		rssnTableSize = receivedSsnTable.numObjects();
	}
	//cppcheck-suppress uninitvar
	p | rssnTableSize;
	if(!p.isUnpacking()){
		CkHashtableIterator *iter = receivedSsnTable.iterator();
		while(iter->hasNext()){
			CkObjID *objID;
			RSSN **row = (RSSN **)iter->next((void **)&objID);
			p | (*objID);
			(*row)->pup(p);
		}
		delete iter;
	}else{
		for(int i=0; i<rssnTableSize; i++){
			CkObjID objID;
			p | objID;
			RSSN *row = new RSSN;
			row->pup(p);
			receivedSsnTable.put(objID) = row;
		}
	}
	
	p | resendReplyRecvd;
	p | restartFlag;

	if(p.isSizing()){
		PUP::sizer *sizep = (PUP::sizer *)&p;
		int pupSize = sizep->size()-startSize;
		DEBUG(char name[40]);
		DEBUG(CkPrintf("[%d]PUP::sizer of %s shows size %d\n",CkMyPe(),objID.toString(name),pupSize));
	//	CkAssert(pupSize <100000000);
	}
	
	double _finTime = CkWallTimer();
	DEBUG(CkPrintf("[%d] Pup took %.6lf\n",CkMyPe(),_finTime - _startTime));
};

/****************
*****************/

/**
 * Getting the pe number of the current processor's buddy.
 * In the team-based approach each processor might checkpoint in the next team, but currently
 * teams are only meant to reduce memory overhead.
 * Note: function getReverseCheckPointPE performs the reverse map. It must be changed accordingly.
 */
int getCheckPointPE(){
	return (CmiMyPe() + 1) % CmiNumPes();
}

/**
 * Getting the pe that checkpoints on this pe.
 */
int getReverseCheckPointPE(){
	return (CmiMyPe() - 1 + CmiNumPes()) % CmiNumPes();
}

//assume it is a packed envelope
envelope *copyEnvelope(envelope *env){
	envelope *newEnv = (envelope *)CmiAlloc(env->getTotalsize());
	memcpy(newEnv,env,env->getTotalsize());
	return newEnv;
}

#endif
