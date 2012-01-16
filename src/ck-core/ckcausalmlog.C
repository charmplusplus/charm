/**
 * Message Logging Fault Tolerance Protocol
 * It includes the main functions for the basic and team-based schemes.
 */

#include "charm.h"
#include "ck.h"
#include "ckmessagelogging.h"
#include "queueing.h"
#include <sys/types.h>
#include <signal.h>
#include "CentralLB.h"

#ifdef _FAULT_CAUSAL_

//#define DEBUG(x)  if(_restartFlag) {x;}
#define DEBUG_MEM(x) //x
#define DEBUG(x)  //x
#define DEBUGRESTART(x)  //x
#define DEBUGLB(x) // x
#define DEBUG_TEAM(x)  // x

#define BUFFERED_LOCAL
#define BUFFERED_REMOTE 

extern const char *idx2str(const CkArrayIndex &ind);
extern const char *idx2str(const ArrayElement *el);
const char *idx2str(const CkArrayIndex &ind){
	return idx2str((const CkArrayIndex &)ind);
};

void getGlobalStep(CkGroupID gID);

bool fault_aware(CkObjID &recver);
void sendCheckpointData(int mode);
void createObjIDList(void *data,ChareMlogData *mlogData);
inline bool isLocal(int destPE);
inline bool isTeamLocal(int destPE);

int _restartFlag=0;
//ERASE int restarted=0; // it's not being used anywhere

//TML: variables for measuring savings with teams in message logging
float MLOGFT_totalLogSize = 0.0;
float MLOGFT_totalMessages = 0.0;
float MLOGFT_totalObjects = 0.0;

//TODO: remove for perf runs
int countHashRefs=0; //count the number of gets
int countHashCollisions=0;

//#define CHECKPOINT_DISK
char *checkpointDirectory=".";
int unAckedCheckpoint=0;

int countLocal=0,countBuffered=0;
int countPiggy=0;
int countClearBufferedLocalCalls=0;

int countUpdateHomeAcks=0;

extern int teamSize;
extern int chkptPeriod;
extern bool parallelRestart;

char *killFile;
int killFlag=0;
int restartingMlogFlag=0;
void readKillFile();
double killTime=0.0;
int checkpointCount=0;


CpvDeclare(Chare *,_currentObj);
CpvDeclare(CkQ<LocalMessageLog> *,_localMessageLog);
CpvDeclare(CkQ<TicketRequest *> *,_delayedTicketRequests);
CpvDeclare(StoredCheckpoint *,_storedCheckpointData);
CpvDeclare(CkQ<MlogEntry *> *,_delayedLocalTicketRequests);
CpvDeclare(Queue, _outOfOrderMessageQueue);
CpvDeclare(CkQ<LocalMessageLog>*,_bufferedLocalMessageLogs);
//CpvDeclare(CkQ<TicketRequest>**,_bufferedTicketRequests);
CpvDeclare(char **,_bufferedTicketRequests);
CpvDeclare(int *,_numBufferedTicketRequests);
CpvDeclare(char *,_bufferTicketReply);



static double adjustChkptPeriod=0.0; //in ms
static double nextCheckpointTime=0.0;//in seconds

double lastBufferedLocalMessageCopyTime;

int _maxBufferedMessages;
int _maxBufferedTicketRequests;
int BUFFER_TIME=2; // in ms


int _ticketRequestHandlerIdx;
int _ticketHandlerIdx;
int _localMessageCopyHandlerIdx;
int _localMessageAckHandlerIdx;
int _pingHandlerIdx;
int _bufferedLocalMessageCopyHandlerIdx;
int _bufferedLocalMessageAckHandlerIdx;
int _bufferedTicketRequestHandlerIdx;
int _bufferedTicketHandlerIdx;


char objString[100];
int _checkpointRequestHandlerIdx;
int _storeCheckpointHandlerIdx;
int _checkpointAckHandlerIdx;
int _getCheckpointHandlerIdx;
int _recvCheckpointHandlerIdx;
int _removeProcessedLogHandlerIdx;

int _verifyAckRequestHandlerIdx;
int _verifyAckHandlerIdx;
int _dummyMigrationHandlerIdx;


int	_getGlobalStepHandlerIdx;
int	_recvGlobalStepHandlerIdx;

int _updateHomeRequestHandlerIdx;
int _updateHomeAckHandlerIdx;
int _resendMessagesHandlerIdx;
int _resendReplyHandlerIdx;
int _receivedTNDataHandlerIdx;
int _distributedLocationHandlerIdx;

//TML: integer constants for team-based message logging
int _restartHandlerIdx;
int _getRestartCheckpointHandlerIdx;
int _recvRestartCheckpointHandlerIdx;
void setTeamRecovery(void *data, ChareMlogData *mlogData);
void unsetTeamRecovery(void *data, ChareMlogData *mlogData);

int verifyAckTotal;
int verifyAckCount;

int verifyAckedRequests=0;

RestartRequest *storedRequest;



int _falseRestart =0; /**
													For testing on clusters we might carry out restarts on 
													a porcessor without actually starting it
													1 -> false restart
													0 -> restart after an actual crash
												*/																

//lock for the ticketRequestHandler and ticketLogLocalMessage methods;
int _lockNewTicket=0;


//Load balancing globals
int onGoingLoadBalancing=0;
void *centralLb;
void (*resumeLbFnPtr)(void *);
int _receiveMlogLocationHandlerIdx;
int _receiveMigrationNoticeHandlerIdx;
int _receiveMigrationNoticeAckHandlerIdx;
int _checkpointBarrierHandlerIdx;
int _checkpointBarrierAckHandlerIdx;

CkVec<MigrationRecord> migratedNoticeList;
CkVec<RetainedMigratedObject *> retainedObjectList;
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


//update location globals
int _receiveLocationHandlerIdx;



// initialize message logging datastructures and register handlers
void _messageLoggingInit(){
	//current object
	CpvInitialize(Chare *,_currentObj);
	
	//registering handlers for message logging
	_ticketRequestHandlerIdx = CkRegisterHandler((CmiHandler)_ticketRequestHandler);
	_ticketHandlerIdx = CkRegisterHandler((CmiHandler)_ticketHandler);
	_localMessageCopyHandlerIdx = CkRegisterHandler((CmiHandler)_localMessageCopyHandler);
	_localMessageAckHandlerIdx = CkRegisterHandler((CmiHandler)_localMessageAckHandler);
	_pingHandlerIdx = CkRegisterHandler((CmiHandler)_pingHandler);
	_bufferedLocalMessageCopyHandlerIdx = CkRegisterHandler((CmiHandler)_bufferedLocalMessageCopyHandler);
	_bufferedLocalMessageAckHandlerIdx = CkRegisterHandler((CmiHandler)_bufferedLocalMessageAckHandler);
	_bufferedTicketRequestHandlerIdx =  CkRegisterHandler((CmiHandler)_bufferedTicketRequestHandler);
	_bufferedTicketHandlerIdx = CkRegisterHandler((CmiHandler)_bufferedTicketHandler);

		
	//handlers for checkpointing
	_storeCheckpointHandlerIdx = CkRegisterHandler((CmiHandler)_storeCheckpointHandler);
	_checkpointAckHandlerIdx = CkRegisterHandler((CmiHandler) _checkpointAckHandler);
	_removeProcessedLogHandlerIdx  = CkRegisterHandler((CmiHandler)_removeProcessedLogHandler);
	_checkpointRequestHandlerIdx =  CkRegisterHandler((CmiHandler)_checkpointRequestHandler);


	//handlers for restart
	_getCheckpointHandlerIdx = CkRegisterHandler((CmiHandler)_getCheckpointHandler);
	_recvCheckpointHandlerIdx = CkRegisterHandler((CmiHandler)_recvCheckpointHandler);
	_updateHomeRequestHandlerIdx =CkRegisterHandler((CmiHandler)_updateHomeRequestHandler);
	_updateHomeAckHandlerIdx =  CkRegisterHandler((CmiHandler) _updateHomeAckHandler);
	_resendMessagesHandlerIdx = CkRegisterHandler((CmiHandler)_resendMessagesHandler);
	_resendReplyHandlerIdx = CkRegisterHandler((CmiHandler)_resendReplyHandler);
	_receivedTNDataHandlerIdx=CkRegisterHandler((CmiHandler)_receivedTNDataHandler);
	_distributedLocationHandlerIdx=CkRegisterHandler((CmiHandler)_distributedLocationHandler);
	_verifyAckRequestHandlerIdx = CkRegisterHandler((CmiHandler)_verifyAckRequestHandler);
	_verifyAckHandlerIdx = CkRegisterHandler((CmiHandler)_verifyAckHandler);
	_dummyMigrationHandlerIdx = CkRegisterHandler((CmiHandler)_dummyMigrationHandler);

	//TML: handlers for team-based message logging
	_restartHandlerIdx = CkRegisterHandler((CmiHandler)_restartHandler);
	_getRestartCheckpointHandlerIdx = CkRegisterHandler((CmiHandler)_getRestartCheckpointHandler);
	_recvRestartCheckpointHandlerIdx = CkRegisterHandler((CmiHandler)_recvRestartCheckpointHandler);

	
	//handlers for load balancing
	_receiveMlogLocationHandlerIdx=CkRegisterHandler((CmiHandler)_receiveMlogLocationHandler);
	_receiveMigrationNoticeHandlerIdx=CkRegisterHandler((CmiHandler)_receiveMigrationNoticeHandler);
	_receiveMigrationNoticeAckHandlerIdx=CkRegisterHandler((CmiHandler)_receiveMigrationNoticeAckHandler);
	_getGlobalStepHandlerIdx=CkRegisterHandler((CmiHandler)_getGlobalStepHandler);
	_recvGlobalStepHandlerIdx=CkRegisterHandler((CmiHandler)_recvGlobalStepHandler);
	_receiveMigrationNoticeHandlerIdx=CkRegisterHandler((CmiHandler)_receiveMigrationNoticeHandler);
	_receiveMigrationNoticeAckHandlerIdx=CkRegisterHandler((CmiHandler)_receiveMigrationNoticeAckHandler);
	_checkpointBarrierHandlerIdx=CkRegisterHandler((CmiHandler)_checkpointBarrierHandler);
	_checkpointBarrierAckHandlerIdx=CkRegisterHandler((CmiHandler)_checkpointBarrierAckHandler);

	
	//handlers for updating locations
	_receiveLocationHandlerIdx=CkRegisterHandler((CmiHandler)_receiveLocationHandler);
	
	//Cpv variables for message logging
	CpvInitialize(CkQ<LocalMessageLog>*,_localMessageLog);
	CpvAccess(_localMessageLog) = new CkQ<LocalMessageLog>(10000);
	CpvInitialize(CkQ<TicketRequest *> *,_delayedTicketRequests);
	CpvAccess(_delayedTicketRequests) = new CkQ<TicketRequest *>;
	CpvInitialize(CkQ<MlogEntry *>*,_delayedLocalTicketRequests);
	CpvAccess(_delayedLocalTicketRequests) = new CkQ<MlogEntry *>;
	CpvInitialize(Queue, _outOfOrderMessageQueue);
	CpvAccess(_outOfOrderMessageQueue) = CqsCreate();
	CpvInitialize(CkQ<LocalMessageLog>*,_bufferedLocalMessageLogs);
	CpvAccess(_bufferedLocalMessageLogs) = new CkQ<LocalMessageLog>;
	
	CpvInitialize(char **,_bufferedTicketRequests);
	CpvAccess(_bufferedTicketRequests) = new char *[CkNumPes()];
	CpvAccess(_numBufferedTicketRequests) = new int[CkNumPes()];
	for(int i=0;i<CkNumPes();i++){
		CpvAccess(_bufferedTicketRequests)[i]=NULL;
		CpvAccess(_numBufferedTicketRequests)[i]=0;
	}
  CpvInitialize(char *,_bufferTicketReply);
	CpvAccess(_bufferTicketReply) = (char *)CmiAlloc(sizeof(BufferedTicketRequestHeader)+_maxBufferedTicketRequests*sizeof(TicketReply));
	
//	CcdCallOnConditionKeep(CcdPERIODIC_100ms,retryTicketRequest,NULL);
	CcdCallFnAfter(retryTicketRequest,NULL,100);	
	
	
	//Cpv variables for checkpoint
	CpvInitialize(StoredCheckpoint *,_storedCheckpointData);
	CpvAccess(_storedCheckpointData) = new StoredCheckpoint;
	
//	CcdCallOnConditionKeep(CcdPERIODIC_10s,startMlogCheckpoint,NULL);
//	printf("[%d] Checkpoint Period is %d s\n",CkMyPe(),chkptPeriod);
//	CcdCallFnAfter(startMlogCheckpoint,NULL,chkptPeriod);
	if(CkMyPe() == 0){
//		CcdCallFnAfter(checkpointAlarm,NULL,chkptPeriod*1000);
#ifdef 	BUFFERED_LOCAL
		if(CmiMyPe() == 0){
			printf("Local messages being buffered _maxBufferedMessages %d BUFFER_TIME %d ms \n",_maxBufferedMessages,BUFFER_TIME);
		}
#endif
	}
#ifdef 	BUFFERED_REMOTE
	if(CmiMyPe() == 0){
		printf("[%d] Remote messages being buffered _maxBufferedTicketRequests %d BUFFER_TIME %d ms %p \n",CkMyPe(),_maxBufferedTicketRequests,BUFFER_TIME,CpvAccess(_bufferTicketReply));
	}
#endif

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
//	CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,checkBufferedLocalMessageCopy,NULL);
	CcdCallFnAfter( checkBufferedLocalMessageCopy ,NULL , BUFFER_TIME);
}

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

void killLocal(void *_dummy,double curWallTime){
	printf("[%d] KillLocal called at %.6lf \n",CkMyPe(),CmiWallTimer());
	if(CmiWallTimer()<killTime-1){
		CcdCallFnAfter(killLocal,NULL,(killTime-CmiWallTimer())*1000);	
	}else{	
		kill(getpid(),SIGKILL);
	}
}



/************************ Message logging methods ****************/

// send a ticket request to a group on a processor
void sendTicketGroupRequest(envelope *env,int destPE,int _infoIdx){
	if(destPE == CLD_BROADCAST || destPE == CLD_BROADCAST_ALL){
		DEBUG(printf("[%d] Group Broadcast \n",CkMyPe()));
		void *origMsg = EnvToUsr(env);
		for(int i=0;i<CmiNumPes();i++){
			if(!(destPE == CLD_BROADCAST && i == CmiMyPe())){
				void *copyMsg = CkCopyMsg(&origMsg);
				envelope *copyEnv = UsrToEnv(copyMsg);
				copyEnv->SN=0;
				copyEnv->TN=0;
				copyEnv->sender.type = TypeInvalid;
				DEBUG(printf("[%d] Sending group broadcast message to proc %d \n",CkMyPe(),i));
				sendTicketGroupRequest(copyEnv,i,_infoIdx);
			}
		}
		return;
	}
	CkObjID recver;
	recver.type = TypeGroup;
	recver.data.group.id = env->getGroupNum();
	recver.data.group.onPE = destPE;
/*	if(recver.data.group.id.idx == 11 && recver.data.group.onPE == 1){
		CmiPrintStackTrace(0);
	}*/
	generateCommonTicketRequest(recver,env,destPE,_infoIdx);
}

//send a ticket request to a nodegroup
void sendTicketNodeGroupRequest(envelope *env,int destNode,int _infoIdx){
	if(destNode == CLD_BROADCAST || destNode == CLD_BROADCAST_ALL){
		DEBUG(printf("[%d] NodeGroup Broadcast \n",CkMyPe()));
		void *origMsg = EnvToUsr(env);
		for(int i=0;i<CmiNumNodes();i++){
			if(!(destNode == CLD_BROADCAST && i == CmiMyNode())){
				void *copyMsg = CkCopyMsg(&origMsg);
				envelope *copyEnv = UsrToEnv(copyMsg);
				copyEnv->SN=0;
				copyEnv->TN=0;
				copyEnv->sender.type = TypeInvalid;
				sendTicketNodeGroupRequest(copyEnv,i,_infoIdx);
			}
		}
		return;
	}
	CkObjID recver;
	recver.type = TypeNodeGroup;
	recver.data.group.id = env->getGroupNum();
	recver.data.group.onPE = destNode;
	generateCommonTicketRequest(recver,env,destNode,_infoIdx);
}

//send a ticket request to an array element
void sendTicketArrayRequest(envelope *env,int destPE,int _infoIdx){
	CkObjID recver;
	recver.type = TypeArray;
	recver.data.array.id = env->getsetArrayMgr();
	recver.data.array.idx = *(&env->getsetArrayIndex());

	if(CpvAccess(_currentObj)!=NULL &&  CpvAccess(_currentObj)->mlogData->objID.type != TypeArray){
		char recverString[100],senderString[100];
		
		DEBUG(printf("[%d] %s being sent message from non-array %s \n",CkMyPe(),recver.toString(recverString),CpvAccess(_currentObj)->mlogData->objID.toString(senderString)));
	}

	generateCommonTicketRequest(recver,env,destPE,_infoIdx);
};

/**
 * A method to generate the actual ticket requests for groups, nodegroups or arrays.
 */
void generateCommonTicketRequest(CkObjID &recver,envelope *_env,int destPE,int _infoIdx){
	envelope *env = _env;
	MCount ticketNumber = 0;
	int resend=0; //is it a resend
	char recverName[100];
	double _startTime=CkWallTimer();
	
	if(CpvAccess(_currentObj) == NULL){
//		CkAssert(0);
		DEBUG(printf("[%d] !!!!WARNING: _currentObj is NULL while message is being sent\n",CkMyPe());)
		generalCldEnqueue(destPE,env,_infoIdx);
		return;
	}
	
	if(env->sender.type == TypeInvalid){
	 	env->sender = CpvAccess(_currentObj)->mlogData->objID;
		//Set message logging data in the envelope
	}else{
		envelope *copyEnv = copyEnvelope(env);
		env = copyEnv;
		env->sender = CpvAccess(_currentObj)->mlogData->objID;
		env->SN = 0;
	}
	
	CkObjID &sender = env->sender;
	env->recver = recver;

	Chare *obj = (Chare *)env->sender.getObject();
	  
	if(env->SN == 0){
		env->SN = obj->mlogData->nextSN(recver);
	}else{
		resend = 1;
	}
	
	char senderString[100];
//	if(env->SN != 1){
		DEBUG(printf("[%d] Generate Ticket Request to %s from %s PE %d SN %d \n",CkMyPe(),env->recver.toString(recverName),env->sender.toString(senderString),destPE,env->SN));
	//	CmiPrintStackTrace(0);
/*	}else{
		DEBUGRESTART(printf("[%d] Generate Ticket Request to %s from %s PE %d SN %d \n",CkMyPe(),env->recver.toString(recverName),env->sender.toString(senderString),destPE,env->SN));
	}*/
		
	MlogEntry *mEntry = new MlogEntry(env,destPE,_infoIdx);
//	CkPackMessage(&(mEntry->env));
//	traceUserBracketEvent(32,_startTime,CkWallTimer());
	
	_startTime = CkWallTimer();

	// uses the proper ticketing mechanism for local, group and general messages
	if(isLocal(destPE)){
		ticketLogLocalMessage(mEntry);
	}else{
		if((teamSize > 1) && isTeamLocal(destPE)){

			// look to see if this message already has a ticket in the team-table
			Chare *senderObj = (Chare *)sender.getObject();
		 	SNToTicket *ticketRow = senderObj->mlogData->teamTable.get(recver);
			if(ticketRow != NULL){
				Ticket ticket = ticketRow->get(env->SN);
				if(ticket.TN != 0){
					ticketNumber = ticket.TN;
					DEBUG(CkPrintf("[%d] Found a team preticketed message\n",CkMyPe()));
				}
			}
		}
		
		// sending the ticket request
		sendTicketRequest(sender,recver,destPE,mEntry,env->SN,ticketNumber,resend);
		
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
 * 1) They belong to the same group in the group-based message logging.
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
void sendTicketRequest(CkObjID &sender,CkObjID &recver,int destPE,MlogEntry *entry,MCount SN,MCount TN,int resend){
	char recverString[100],senderString[100];
	envelope *env = entry->env;
	DEBUG(printf("[%d] Sending ticket Request to %s from %s PE %d SN %d time %.6lf \n",CkMyPe(),env->recver.toString(recverString),env->sender.toString(senderString),destPE,env->SN,CkWallTimer()));
/*	envelope *env = entry->env;
	printf("[%d] Sending ticket Request to %s from %s PE %d SN %d time %.6lf \n",CkMyPe(),env->recver.toString(recverString),env->sender.toString(senderString),destPE,env->SN,CkWallTimer());*/

	Chare *obj = (Chare *)entry->env->sender.getObject();
	if(!resend){
		//TML: only stores message if either it goes to this processor or to a processor in a different group
		if(!isTeamLocal(entry->destPE)){
			obj->mlogData->addLogEntry(entry);
			MLOGFT_totalMessages += 1.0;
			MLOGFT_totalLogSize += entry->env->getTotalsize();
		}else{
			// the message has to be deleted after it has been sent
			entry->env->freeMsg = true;
		}
	}

#ifdef BUFFERED_REMOTE
	//buffer the ticket request 
	if(CpvAccess(_bufferedTicketRequests)[destPE] == NULL){
		//first message to this processor, buffer needs to be created
		int _allocSize = sizeof(TicketRequest)*_maxBufferedTicketRequests + sizeof(BufferedTicketRequestHeader);
		CpvAccess(_bufferedTicketRequests)[destPE] = (char *)CmiAlloc(_allocSize);
		DEBUG(CmiPrintf("[%d] _bufferedTicketRequests[%d] allocated as %p\n",CmiMyPe(),destPE,&((CpvAccess(_bufferedTicketRequests))[destPE][0])));
	}
	//CpvAccess(_bufferedTicketRequests)[destPE]->enq(ticketRequest);
	//Buffer the ticketrequests
	TicketRequest *ticketRequest = (TicketRequest *)&(CpvAccess(_bufferedTicketRequests)[destPE][sizeof(BufferedTicketRequestHeader)+CpvAccess(_numBufferedTicketRequests)[destPE]*sizeof(TicketRequest)]);
	ticketRequest->sender = sender;
	ticketRequest->recver = recver;
	ticketRequest->logEntry = entry;
	ticketRequest->SN = SN;
	ticketRequest->TN = TN;
	ticketRequest->senderPE = CkMyPe();

	CpvAccess(_numBufferedTicketRequests)[destPE]++;
	
	
	if(CpvAccess(_numBufferedTicketRequests)[destPE] >= _maxBufferedTicketRequests){
		sendBufferedTicketRequests(destPE);
	}else{
		if(CpvAccess(_numBufferedTicketRequests)[destPE] == 1){
			int *checkPE = new int;
			*checkPE = destPE;
			CcdCallFnAfter( checkBufferedTicketRequests ,checkPE , BUFFER_TIME);		
		}
	}
#else

	TicketRequest ticketRequest;
	ticketRequest.sender = sender;
	ticketRequest.recver = recver;
	ticketRequest.logEntry = entry;
	ticketRequest.SN = SN;
	ticketRequest.TN = TN;
	ticketRequest.senderPE = CkMyPe();
	
	CmiSetHandler((void *)&ticketRequest,_ticketRequestHandlerIdx);
//	CmiBecomeImmediate(&ticketRequest);
	CmiSyncSend(destPE,sizeof(TicketRequest),(char *)&ticketRequest);
#endif
	DEBUG_MEM(CmiMemoryCheck());
};

/**
 * Send the ticket requests buffered for processor PE
 **/
void sendBufferedTicketRequests(int destPE){
	DEBUG_MEM(CmiMemoryCheck());
	int numberRequests = CpvAccess(_numBufferedTicketRequests)[destPE];
	if(numberRequests == 0){
		return;
	}
	DEBUG(printf("[%d] Send Buffered Ticket Requests to %d number %d\n",CkMyPe(),destPE,numberRequests));
	int totalSize = sizeof(BufferedTicketRequestHeader )+numberRequests*(sizeof(TicketRequest));
	void *buf = &(CpvAccess(_bufferedTicketRequests)[destPE][0]);
	BufferedTicketRequestHeader *header = (BufferedTicketRequestHeader *)buf;
	header->numberLogs = numberRequests;
	
	CmiSetHandler(buf,_bufferedTicketRequestHandlerIdx);
	CmiSyncSend(destPE,totalSize,(char *)buf);
	
	CpvAccess(_numBufferedTicketRequests)[destPE]=0;
	DEBUG_MEM(CmiMemoryCheck());
};

void checkBufferedTicketRequests(void *_destPE,double curWallTime){
	int destPE = *(int *)_destPE;
  if(CpvAccess(_numBufferedTicketRequests)[destPE] > 0){
		sendBufferedTicketRequests(destPE);
//		traceUserEvent(35);
	}
	delete (int *)_destPE;
	DEBUG_MEM(CmiMemoryCheck());
};

/**
 * Gets a ticket for a local message and then sends a copy to the buddy.
 * This method is always in the main thread(not interrupt).. so it should 
 * never find itself locked out of a newTicket.
 */
void ticketLogLocalMessage(MlogEntry *entry){
	double _startTime=CkWallTimer();
	DEBUG_MEM(CmiMemoryCheck());

	Chare *recverObj = (Chare *)entry->env->recver.getObject();
	DEBUG(Chare *senderObj = (Chare *)entry->env->sender.getObject();)
	if(recverObj){
		//Consider the case, after a restart when this message has already been allotted a ticket number
		// and should get the same one as the old one.
		Ticket ticket;
		if(recverObj->mlogData->mapTable.numObjects() > 0){
			ticket.TN = recverObj->mlogData->searchRestoredLocalQ(entry->env->sender,entry->env->recver,entry->env->SN);
		}else{
			ticket.TN = 0;
		}
		
		char senderString[100], recverString[100] ;
		
		if(ticket.TN == 0){
			ticket = recverObj->mlogData->next_ticket(entry->env->sender,entry->env->SN);
	
			if(ticket.TN == 0){
				CpvAccess(_delayedLocalTicketRequests)->enq(entry);
				DEBUG(printf("[%d] Local Message request enqueued for SN %d sender %s recver %s \n",CmiMyPe(),entry->env->SN,entry->env->sender.toString(senderString),entry->env->recver.toString(recverString)));
				
	//		_lockNewTicket = 0;
//				traceUserBracketEvent(33,_startTime,CkWallTimer());
				return;
			}
		}	
		//TODO: check for the case when an invalid ticket is returned
		//TODO: check for OLD or RECEIVED TICKETS
		entry->env->TN = ticket.TN;
		CkAssert(entry->env->TN > 0);
		DEBUG(printf("[%d] Local Message gets TN %d for SN %d sender %s recver %s \n",CmiMyPe(),entry->env->TN,entry->env->SN,entry->env->sender.toString(senderString),entry->env->recver.toString(recverString)));
	
		// sends a copy of the metadata to the buddy	
		sendLocalMessageCopy(entry);
		
		DEBUG_MEM(CmiMemoryCheck());

		// sets the unackedLocal flag and stores the message in the log
		entry->unackedLocal = 1;
		CpvAccess(_currentObj)->mlogData->addLogEntry(entry);

		DEBUG_MEM(CmiMemoryCheck());
	}else{
		CkPrintf("[%d] Local message in team-based message logging %d to %d\n",CkMyPe(),CkMyPe(),entry->destPE);
		DEBUG(printf("[%d] Local recver object in NULL \n",CmiMyPe()););
	}
	_lockNewTicket=0;
//	traceUserBracketEvent(33,_startTime,CkWallTimer());
};

/**
 * Sends the metadata of a local message to its buddy.
 */
void sendLocalMessageCopy(MlogEntry *entry){
	LocalMessageLog msgLog;
	msgLog.sender = entry->env->sender;
	msgLog.recver = entry->env->recver;
	msgLog.SN = entry->env->SN;
	msgLog.TN = entry->env->TN;
	msgLog.entry = entry;
	msgLog.senderPE = CkMyPe();
	
	char recvString[100];
	char senderString[100];
	DEBUG(printf("[%d] Sending local message log from %s to %s SN %d TN %d to processor %d handler %d time %.6lf entry %p env %p \n",CkMyPe(),msgLog.sender.toString(senderString),msgLog.recver.toString(recvString),msgLog.SN,	msgLog.TN,getCheckPointPE(),_localMessageCopyHandlerIdx,CkWallTimer(),entry,entry->env));

#ifdef BUFFERED_LOCAL
	countLocal++;
	CpvAccess(_bufferedLocalMessageLogs)->enq(msgLog);
	if(CpvAccess(_bufferedLocalMessageLogs)->length() >= _maxBufferedMessages){
		sendBufferedLocalMessageCopy();
	}else{
		if(countClearBufferedLocalCalls < 10 && CpvAccess(_bufferedLocalMessageLogs)->length() == 1){
			lastBufferedLocalMessageCopyTime = CkWallTimer();
			CcdCallFnAfter( checkBufferedLocalMessageCopy ,NULL , BUFFER_TIME);
			countClearBufferedLocalCalls++;
		}	
	}
#else	
	CmiSetHandler((void *)&msgLog,_localMessageCopyHandlerIdx);
	
	CmiSyncSend(getCheckPointPE(),sizeof(LocalMessageLog),(char *)&msgLog);
#endif
	DEBUG_MEM(CmiMemoryCheck());
};


void sendBufferedLocalMessageCopy(){
	int numberLogs = CpvAccess(_bufferedLocalMessageLogs)->length();
	if(numberLogs == 0){
		return;
	}
	countBuffered++;
	int totalSize = sizeof(BufferedLocalLogHeader)+numberLogs*(sizeof(LocalMessageLog));
	void *buf=CmiAlloc(totalSize);
	BufferedLocalLogHeader *header = (BufferedLocalLogHeader *)buf;
	header->numberLogs=numberLogs;

	DEBUG_MEM(CmiMemoryCheck());
	DEBUG(printf("[%d] numberLogs in sendBufferedCopy = %d buf %p\n",CkMyPe(),numberLogs,buf));
	
	char *ptr = (char *)buf;
	ptr = &ptr[sizeof(BufferedLocalLogHeader)];
	
	for(int i=0;i<numberLogs;i++){
		LocalMessageLog log = CpvAccess(_bufferedLocalMessageLogs)->deq();
		memcpy(ptr,&log,sizeof(LocalMessageLog));
		ptr = &ptr[sizeof(LocalMessageLog)];
	}

	CmiSetHandler(buf,_bufferedLocalMessageCopyHandlerIdx);

	CmiSyncSendAndFree(getCheckPointPE(),totalSize,(char *)buf);
	DEBUG_MEM(CmiMemoryCheck());
};

void checkBufferedLocalMessageCopy(void *_dummy,double curWallTime){
	countClearBufferedLocalCalls--;
	if(countClearBufferedLocalCalls > 10){
		CmiAbort("multiple checkBufferedLocalMessageCopy being called \n");
	}
	DEBUG_MEM(CmiMemoryCheck());
	DEBUG(printf("[%d] checkBufferedLocalMessageCopy \n",CkMyPe()));
	if((curWallTime-lastBufferedLocalMessageCopyTime)*1000 > BUFFER_TIME && CpvAccess(_bufferedLocalMessageLogs)->length() > 0){
		if(CpvAccess(_bufferedLocalMessageLogs)->length() > 0){
			sendBufferedLocalMessageCopy();
//			traceUserEvent(36);
		}
	}
	DEBUG_MEM(CmiMemoryCheck());
}

/****
	The handler functions
*****/


inline bool _processTicketRequest(TicketRequest *ticketRequest,TicketReply *reply=NULL);
/**
 *  If there are any delayed requests, process them first before 
 *  processing this request
 * */
inline void _ticketRequestHandler(TicketRequest *ticketRequest){
	DEBUG(printf("[%d] Ticket Request handler started \n",CkMyPe()));
	double 	_startTime = CkWallTimer();
	if(CpvAccess(_delayedTicketRequests)->length() > 0){
		retryTicketRequest(NULL,_startTime);
	}
	_processTicketRequest(ticketRequest);
	CmiFree(ticketRequest);
//	traceUserBracketEvent(21,_startTime,CkWallTimer());			
}
/** Handler used for dealing with a bunch of ticket requests
 * from one processor. The replies are also bunched together
 * Does not use _ticketRequestHandler
 * */
void _bufferedTicketRequestHandler(BufferedTicketRequestHeader *recvdHeader){
	DEBUG(printf("[%d] Buffered Ticket Request handler started header %p\n",CkMyPe(),recvdHeader));
	DEBUG_MEM(CmiMemoryCheck());
	double _startTime = CkWallTimer();
	if(CpvAccess(_delayedTicketRequests)->length() > 0){
		retryTicketRequest(NULL,_startTime);
	}
	DEBUG_MEM(CmiMemoryCheck());
  int numRequests = recvdHeader->numberLogs;
	char *msg = (char *)recvdHeader;
	msg = &msg[sizeof(BufferedTicketRequestHeader)];
	int senderPE=((TicketRequest *)msg)->senderPE;

	
	int totalSize = sizeof(BufferedTicketRequestHeader)+numRequests*sizeof(TicketReply);
	void *buf = (void *)&(CpvAccess(_bufferTicketReply)[0]);
	
	char *ptr = (char *)buf;
	BufferedTicketRequestHeader *header = (BufferedTicketRequestHeader *)ptr;
	header->numberLogs = 0;

	DEBUG_MEM(CmiMemoryCheck());
	
	ptr = &ptr[sizeof(BufferedTicketRequestHeader)]; //ptr at which the ticket replies will be stored
	
	for(int i=0;i<numRequests;i++){
		TicketRequest *request = (TicketRequest *)msg;
		msg = &msg[sizeof(TicketRequest)];
		bool replied = _processTicketRequest(request,(TicketReply *)ptr);

		if(replied){
			//the ticket request has been processed and 
			//the reply will be stored in the ptr
			header->numberLogs++;
			ptr = &ptr[sizeof(TicketReply)];
		}
	}
/*	if(header->numberLogs == 0){
			printf("[%d] *************** Not sending any replies to previous buffered ticketRequest \n",CkMyPe());
	}*/

	CmiSetHandler(buf,_bufferedTicketHandlerIdx);
	CmiSyncSend(senderPE,totalSize,(char *)buf);
	CmiFree(recvdHeader);
//	traceUserBracketEvent(21,_startTime,CkWallTimer());			
	DEBUG_MEM(CmiMemoryCheck());
};

/**Process the ticket request. 
 * If it is processed and a reply is being sent 
 * by this processor return true
 * else return false.
 * If a reply buffer is specified put the reply into that
 * else send the reply
 * */
inline bool _processTicketRequest(TicketRequest *ticketRequest,TicketReply *reply){

/*	if(_lockNewTicket){
		printf("ddeded %d\n",CkMyPe());
		if(CmiIsImmediate(ticketRequest)){
			CmiSetHandler(ticketRequest, (CmiGetHandler(ticketRequest))^0x8000);
		}
		CmiSyncSend(CkMyPe(),sizeof(TicketRequest),(char *)ticketRequest);
		
	}else{
		_lockNewTicket = 1;
	}*/

	DEBUG_MEM(CmiMemoryCheck());

	// getting information from request
	CkObjID sender = ticketRequest->sender;
	CkObjID recver = ticketRequest->recver;
	MCount SN = ticketRequest->SN;
	MCount TN = ticketRequest->TN;
	Chare *recverObj = (Chare *)recver.getObject();
	
	DEBUG(char recverName[100]);
	DEBUG(recver.toString(recverName);)

	if(recverObj == NULL){
		int estPE = recver.guessPE();
		if(estPE == CkMyPe() || estPE == -1){		
			//try to fulfill the request after some time
			char senderString[100];
			DEBUG(printf("[%d] Ticket request to %s SN %d from %s delayed estPE %d mesg %p\n",CkMyPe(),recverName, SN,sender.toString(senderString),estPE,ticketRequest));
			if(estPE == CkMyPe() && recver.type == TypeArray){
				CkArrayID aid(recver.data.array.id);		
				CkLocMgr *locMgr = aid.ckLocalBranch()->getLocMgr();
				DEBUG(printf("[%d] Object with delayed ticket request has home at %d\n",CkMyPe(),locMgr->homePe(recver.data.array.idx)));
			}
			TicketRequest *delayed = (TicketRequest*)CmiAlloc(sizeof(TicketRequest));
			*delayed = *ticketRequest;
			CpvAccess(_delayedTicketRequests)->enq(delayed);
			
		}else{
			DEBUGRESTART(printf("[%d] Ticket request to %s SN %d needs to be forwarded estPE %d mesg %p\n",CkMyPe(),recver.toString(recverName), SN,estPE,ticketRequest));
			TicketRequest forward = *ticketRequest;
			CmiSetHandler(&forward,_ticketRequestHandlerIdx);
			CmiSyncSend(estPE,sizeof(TicketRequest),(char *)&forward);
		}
	DEBUG_MEM(CmiMemoryCheck());
		return false; // if the receverObj does not exist the ticket request cannot have been 
		              // processed successfully
	}else{
		char senderString[100];
		
		Ticket ticket;

		// checking if the message is team local and if it has a ticket already assigned
		if(teamSize > 1 && TN != 0){
			DEBUG(CkPrintf("[%d] Message has a ticket already assigned\n",CkMyPe()));
			ticket.TN = TN;
			recverObj->mlogData->verifyTicket(sender,SN,TN);
		}

		//check if a ticket for this has been already handed out to an object that used to be local but 
		// is no longer so.. need for parallel restart
		if(recverObj->mlogData->mapTable.numObjects() > 0){
			
			ticket.TN = recverObj->mlogData->searchRestoredLocalQ(ticketRequest->sender,ticketRequest->recver,ticketRequest->SN);
		}
		
		if(ticket.TN == 0){
			ticket = recverObj->mlogData->next_ticket(sender,SN);
		}
		if(ticket.TN > recverObj->mlogData->tProcessed){
			ticket.state = NEW_TICKET;
		}else{
			ticket.state = OLD_TICKET;
		}
		//TODO: check for the case when an invalid ticket is returned
		if(ticket.TN == 0){
			DEBUG(printf("[%d] Ticket request to %s SN %d from %s delayed mesg %p\n",CkMyPe(),recverName, SN,sender.toString(senderString),ticketRequest));
			TicketRequest *delayed = (TicketRequest*)CmiAlloc(sizeof(TicketRequest));
			*delayed = *ticketRequest;
			CpvAccess(_delayedTicketRequests)->enq(delayed);
			return false;
		}
/*		if(ticket.TN < SN){ //error state this really should not happen
			recver.toString(recverName);
		  printf("[%d] TN %d handed out to %s SN %d by %s sent to PE %d mesg %p at %.6lf\n",CkMyPe(),ticket.TN,sender.toString(senderString),SN,recverName,ticketRequest->senderPE,ticketRequest,CmiWallTimer());
		}*/
//		CkAssert(ticket.TN >= SN);
		DEBUG(printf("[%d] TN %d handed out to %s SN %d by %s sent to PE %d mesg %p at %.6lf\n",CkMyPe(),ticket.TN,sender.toString(senderString),SN,recverName,ticketRequest->senderPE,ticketRequest,CmiWallTimer()));
//		TicketReply *ticketReply = (TicketReply *)CmiAlloc(sizeof(TicketReply));
    if(reply == NULL){ 
			//There is no reply buffer and the ticketreply is going to be 
			//sent immediately
			TicketReply ticketReply;
			ticketReply.request = *ticketRequest;
			ticketReply.ticket = ticket;
			ticketReply.recverPE = CkMyPe();
			CmiSetHandler(&ticketReply,_ticketHandlerIdx);
//		CmiBecomeImmediate(&ticketReply);
			CmiSyncSend(ticketRequest->senderPE,sizeof(TicketReply),(char *)&ticketReply);
	 }else{ // Store ticket reply in the buffer provided
		 reply->request = *ticketRequest;
		 reply->ticket = ticket;
		 reply->recverPE = CkMyPe();
		 CmiSetHandler(reply,_ticketHandlerIdx); // not strictly necessary but will do that 
		                                         // in case the ticket needs to be forwarded or something
	 }
		DEBUG_MEM(CmiMemoryCheck());
		return true;
	}
//	_lockNewTicket=0;
};


/**
 * @brief This function handles the ticket received after a request.
 */
inline void _ticketHandler(TicketReply *ticketReply){

	double _startTime = CkWallTimer();
	DEBUG_MEM(CmiMemoryCheck());	
	
	char senderString[100];
	CkObjID sender = ticketReply->request.sender;
	CkObjID recver = ticketReply->request.recver;
	
	if(sender.guessPE() != CkMyPe()){
		DEBUG(CkAssert(sender.guessPE()>= 0));
		DEBUG(printf("[%d] TN %d forwarded to %s on PE %d \n",CkMyPe(),ticketReply->ticket.TN,sender.toString(senderString),sender.guessPE()));
	//	printf("[%d] TN %d forwarded to %s on PE %d \n",CkMyPe(),ticketReply->ticket.TN,sender.toString(senderString),sender.guessPE());
		ticketReply->ticket.state = ticketReply->ticket.state | FORWARDED_TICKET;
		CmiSetHandler(ticketReply,_ticketHandlerIdx);
#ifdef BUFFERED_REMOTE
		//will be freed by the buffered ticket handler most of the time
		//this might lead to a leak just after migration
		//when the ticketHandler is directly used without going through the buffered handler
		CmiSyncSend(sender.guessPE(),sizeof(TicketReply),(char *)ticketReply);
#else
		CmiSyncSendAndFree(sender.guessPE(),sizeof(TicketReply),(char *)ticketReply);
#endif	
	}else{
		char recverName[100];
		DEBUG(printf("[%d] TN %d received for %s SN %d from %s  time %.6lf \n",CkMyPe(),ticketReply->ticket.TN,sender.toString(senderString),ticketReply->request.SN,recver.toString(recverName),CmiWallTimer()));
		MlogEntry *logEntry=NULL;
		if(ticketReply->ticket.state & FORWARDED_TICKET){
			// Handle the case when you receive a forwarded message, We need to search through the message queue since the logEntry pointer is no longer valid
			DEBUG(printf("[%d] TN %d received for %s has been forwarded \n",CkMyPe(),ticketReply->ticket.TN,sender.toString(senderString)));
			Chare *senderObj = (Chare *)sender.getObject();
			if(senderObj){
				CkQ<MlogEntry *> *mlog = senderObj->mlogData->getMlog();
				for(int i=0;i<mlog->length();i++){
					MlogEntry *tempEntry = (*mlog)[i];
					if(tempEntry->env != NULL && ticketReply->request.sender == tempEntry->env->sender && ticketReply->request.recver == tempEntry->env->recver && ticketReply->request.SN == tempEntry->env->SN){
						logEntry = tempEntry;
						break;
					}
				}
				if(logEntry == NULL){
#ifdef BUFFERED_REMOTE
#else
					CmiFree(ticketReply);
#endif					
					return;
				}
			}else{
				CmiAbort("This processor thinks it should have the sender\n");
			}
			ticketReply->ticket.state ^= FORWARDED_TICKET;
		}else{
			logEntry = ticketReply->request.logEntry;
		}
		if(logEntry->env->TN <= 0){
			//This logEntry has not received a TN earlier
			char recverString[100];
			logEntry->env->TN = ticketReply->ticket.TN;
			logEntry->env->setSrcPe(CkMyPe());
			if(ticketReply->ticket.state == NEW_TICKET){

				// if message is group local, we store its metadata in teamTable
				if(isTeamLocal(ticketReply->recverPE)){
					//DEBUG_TEAM(CkPrintf("[%d] Storing meta data for intragroup message %u\n",CkMyPe(),ticketReply->request.SN);)
					Chare *senderObj = (Chare *)sender.getObject();
					SNToTicket *ticketRow = senderObj->mlogData->teamTable.get(recver);
					if(ticketRow == NULL){
						ticketRow = new SNToTicket();
						senderObj->mlogData->teamTable.put(recver) = ticketRow;	
					}
					ticketRow->put(ticketReply->request.SN) = ticketReply->ticket;
				}

				DEBUG(printf("[%d] Message sender %s recver %s SN %d TN %d to processor %d env %p size %d \n",CkMyPe(),sender.toString(senderString),recver.toString(recverString), ticketReply->request.SN,ticketReply->ticket.TN,ticketReply->recverPE,logEntry->env,logEntry->env->getTotalsize()));
				if(ticketReply->recverPE != CkMyPe()){
					generalCldEnqueue(ticketReply->recverPE,logEntry->env,logEntry->_infoIdx);
				}else{
					//It is now a local message use the local message protocol
					sendLocalMessageCopy(logEntry);
				}	
			}
		}else{
			DEBUG(printf("[%d] Message sender %s recver %s SN %d already had TN %d received TN %d\n",CkMyPe(),sender.toString(senderString),recver.toString(recverName),ticketReply->request.SN,logEntry->env->TN,ticketReply->ticket.TN));
		}
		recver.updatePosition(ticketReply->recverPE);
#ifdef BUFFERED_REMOTE
#else
		CmiFree(ticketReply);
#endif
	}
	CmiMemoryCheck();

//	traceUserBracketEvent(22,_startTime,CkWallTimer());	
};

/**
 * Message to handle the bunch of tickets 
 * that we get from one processor. We send 
 * the tickets to be handled one at a time
 * */

void _bufferedTicketHandler(BufferedTicketRequestHeader *recvdHeader){
	double _startTime=CmiWallTimer();
	int numTickets = recvdHeader->numberLogs;
	char *msg = (char *)recvdHeader;
	msg = &msg[sizeof(BufferedTicketRequestHeader)];
	DEBUG_MEM(CmiMemoryCheck());
	
	TicketReply *_reply = (TicketReply *)msg;

	
	for(int i=0;i<numTickets;i++){
		TicketReply *reply = (TicketReply *)msg;
		_ticketHandler(reply);
		
		msg = &msg[sizeof(TicketReply)];
	}
	
	CmiFree(recvdHeader);
//	traceUserBracketEvent(22,_startTime,CkWallTimer());
	DEBUG_MEM(CmiMemoryCheck());
};

/**
 * Stores the metadata of a local message from its buddy.
 */
void _localMessageCopyHandler(LocalMessageLog *msgLog){
	double _startTime = CkWallTimer();
	
	char senderString[100],recverString[100];
	DEBUG(printf("[%d] Local Message log from processor %d sender %s recver %s TN %d handler %d time %.6lf \n",CkMyPe(),msgLog->PE,msgLog->sender.toString(senderString),msgLog->recver.toString(recverString),msgLog->TN,_localMessageAckHandlerIdx,CmiWallTimer()));
/*	if(!fault_aware(msgLog->recver)){
		CmiAbort("localMessageCopyHandler with non fault aware local message copy");
	}*/
	CpvAccess(_localMessageLog)->enq(*msgLog);
	
	LocalMessageLogAck ack;
	ack.entry = msgLog->entry;
	DEBUG(printf("[%d] About to send back ack \n",CkMyPe()));
	CmiSetHandler(&ack,_localMessageAckHandlerIdx);
	CmiSyncSend(msgLog->senderPE,sizeof(LocalMessageLogAck),(char *)&ack);
	
//	traceUserBracketEvent(23,_startTime,CkWallTimer());
};

void _bufferedLocalMessageCopyHandler(BufferedLocalLogHeader *recvdHeader,int freeHeader){
	double _startTime = CkWallTimer();
	DEBUG_MEM(CmiMemoryCheck());
	
	int numLogs = recvdHeader->numberLogs;
	char *msg = (char *)recvdHeader;

	//piggy back the logs already stored on this processor
	int numPiggyLogs = CpvAccess(_bufferedLocalMessageLogs)->length();
	numPiggyLogs=0; //uncomment to turn off piggy backing of acks
/*	if(numPiggyLogs > 0){
		if((*CpvAccess(_bufferedLocalMessageLogs))[0].PE != getCheckPointPE()){
			CmiAssert(0);
		}
	}*/
	DEBUG(printf("[%d] _bufferedLocalMessageCopyHandler numLogs %d numPiggyLogs %d\n",CmiMyPe(),numLogs,numPiggyLogs));
	
	int totalSize = sizeof(BufferedLocalLogHeader)+numLogs*sizeof(LocalMessageLogAck)+sizeof(BufferedLocalLogHeader)+numPiggyLogs*sizeof(LocalMessageLog);
	void *buf = CmiAlloc(totalSize);
	char *ptr = (char *)buf;
	memcpy(ptr,msg,sizeof(BufferedLocalLogHeader));
	
	msg = &msg[sizeof(BufferedLocalLogHeader)];
	ptr = &ptr[sizeof(BufferedLocalLogHeader)];

	DEBUG_MEM(CmiMemoryCheck());
	int PE;
	for(int i=0;i<numLogs;i++){
		LocalMessageLog *msgLog = (LocalMessageLog *)msg;
		CpvAccess(_localMessageLog)->enq(*msgLog);
		PE = msgLog->senderPE;
		DEBUG(CmiAssert( PE == getCheckPointPE()));

		LocalMessageLogAck *ack = (LocalMessageLogAck *)ptr;
		ack->entry = msgLog->entry;
		
		msg = &msg[sizeof(LocalMessageLog)];
		ptr = &ptr[sizeof(LocalMessageLogAck)];
	}
	DEBUG_MEM(CmiMemoryCheck());

	BufferedLocalLogHeader *piggyHeader = (BufferedLocalLogHeader *)ptr;
	piggyHeader->numberLogs = numPiggyLogs;
	ptr = &ptr[sizeof(BufferedLocalLogHeader)];
	if(numPiggyLogs > 0){
		countPiggy++;
	}

	for(int i=0;i<numPiggyLogs;i++){
		LocalMessageLog log = CpvAccess(_bufferedLocalMessageLogs)->deq();
		memcpy(ptr,&log,sizeof(LocalMessageLog));
		ptr = &ptr[sizeof(LocalMessageLog)];
	}
	DEBUG_MEM(CmiMemoryCheck());
	
	CmiSetHandler(buf,_bufferedLocalMessageAckHandlerIdx);
	CmiSyncSendAndFree(PE,totalSize,(char *)buf);
		
/*	for(int i=0;i<CpvAccess(_localMessageLog)->length();i++){
			LocalMessageLog localLogEntry = (*CpvAccess(_localMessageLog))[i];
			if(!fault_aware(localLogEntry.recver)){
				CmiAbort("Non fault aware logEntry recver found while clearing old local logs");
			}
	}*/
	if(freeHeader){
		CmiFree(recvdHeader);
	}
	DEBUG_MEM(CmiMemoryCheck());
//	traceUserBracketEvent(23,_startTime,CkWallTimer());
}


void _localMessageAckHandler(LocalMessageLogAck *ack){
	
	double _startTime = CkWallTimer();
	
	MlogEntry *entry = ack->entry;
	if(entry == NULL){
		CkExit();
	}
	entry->unackedLocal = 0;
	envelope *env = entry->env;
	char recverName[100];
	char senderString[100];
	DEBUG_MEM(CmiMemoryCheck());
	
	DEBUG(printf("[%d] at start of local message ack handler for entry %p env %p\n",CkMyPe(),entry,env));
	if(env == NULL)
		return;
	CkAssert(env->SN > 0);
	CkAssert(env->TN > 0);
	env->sender.toString(senderString);
	DEBUG(printf("[%d] local message ack handler verified sender \n",CkMyPe()));
	env->recver.toString(recverName);

	DEBUG(printf("[%d] Local Message log ack received for message from %s to %s TN %d time %.6lf \n",CkMyPe(),env->sender.toString(senderString),env->recver.toString(recverName),env->TN,CkWallTimer()));
	
/*	
	void *origMsg = EnvToUsr(env);
	void *copyMsg = CkCopyMsg(&origMsg);
	envelope *copyEnv = UsrToEnv(copyMsg);
	entry->env = UsrToEnv(origMsg);*/

//	envelope *copyEnv = copyEnvelope(env);

	envelope *copyEnv = env;
	copyEnv->localMlogEntry = entry;

	DEBUG(printf("[%d] Local message copied response to ack \n",CkMyPe()));
	if(CmiMyPe() != entry->destPE){
		DEBUG(printf("[%d] Formerly remote message to PE %d converted to local\n",CmiMyPe(),entry->destPE));
	}
//	generalCldEnqueue(entry->destPE,copyEnv,entry->_infoIdx)
	_skipCldEnqueue(CmiMyPe(),copyEnv,entry->_infoIdx);	
	
	
#ifdef BUFFERED_LOCAL
#else
	CmiFree(ack);
//	traceUserBracketEvent(24,_startTime,CkWallTimer());
#endif
	
	DEBUG_MEM(CmiMemoryCheck());
	DEBUG(printf("[%d] Local message log ack handled \n",CkMyPe()));
}


void _bufferedLocalMessageAckHandler(BufferedLocalLogHeader *recvdHeader){

	double _startTime=CkWallTimer();
	DEBUG_MEM(CmiMemoryCheck());

	int numLogs = recvdHeader->numberLogs;
	char *msg = (char *)recvdHeader;
	msg = &msg[sizeof(BufferedLocalLogHeader)];

	DEBUG(printf("[%d] _bufferedLocalMessageAckHandler numLogs %d \n",CmiMyPe(),numLogs));
	
	for(int i=0;i<numLogs;i++){
		LocalMessageLogAck *ack = (LocalMessageLogAck *)msg;
		_localMessageAckHandler(ack);
		
		msg = &msg[sizeof(LocalMessageLogAck)];	
	}

	//deal with piggy backed local logs
	BufferedLocalLogHeader *piggyHeader = (BufferedLocalLogHeader *)msg;
	//printf("[%d] number of local logs piggied with ack %d \n",CkMyPe(),piggyHeader->numberLogs);
	if(piggyHeader->numberLogs > 0){
		_bufferedLocalMessageCopyHandler(piggyHeader,0);
	}
	
	CmiFree(recvdHeader);
	DEBUG_MEM(CmiMemoryCheck());
//	traceUserBracketEvent(24,_startTime,CkWallTimer());
}

bool fault_aware(CkObjID &recver){
	switch(recver.type){
		case TypeChare:
			return false;
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

int preProcessReceivedMessage(envelope *env,Chare **objPointer,MlogEntry **logEntryPointer){
	char recverString[100];
	char senderString[100];
	
	DEBUG_MEM(CmiMemoryCheck());
	CkObjID recver = env->recver;
	if(!fault_aware(recver))
		return 1;


	Chare *obj = (Chare *)recver.getObject();
	*objPointer = obj;
	if(obj == NULL){
		int possiblePE = recver.guessPE();
		if(possiblePE != CkMyPe()){
			int totalSize = env->getTotalsize();			
			CmiSyncSend(possiblePE,totalSize,(char *)env);
		}
		return 0;
	}


	double _startTime = CkWallTimer();
//env->sender.updatePosition(env->getSrcPe());
	if(env->TN == obj->mlogData->tProcessed+1){
		//the message that needs to be processed now
		DEBUG(printf("[%d] Message SN %d TN %d sender %s recver %s being processed recvPointer %p\n",CkMyPe(),env->SN,env->TN,env->sender.toString(senderString), recver.toString(recverString),obj));
		// once we find a message that we can process we put back all the messages in the out of order queue
		// back into the main scheduler queue. 
		if(env->sender.guessPE() == CkMyPe()){
			*logEntryPointer = env->localMlogEntry;
		}
	DEBUG_MEM(CmiMemoryCheck());
		while(!CqsEmpty(CpvAccess(_outOfOrderMessageQueue))){
			void *qMsgPtr;
			CqsDequeue(CpvAccess(_outOfOrderMessageQueue),&qMsgPtr);
			envelope *qEnv = (envelope *)qMsgPtr;
			CqsEnqueueGeneral((Queue)CpvAccess(CsdSchedQueue),qEnv,CQS_QUEUEING_FIFO,qEnv->getPriobits(),(unsigned int *)qEnv->getPrioPtr());			
	DEBUG_MEM(CmiMemoryCheck());
		}
//		traceUserBracketEvent(25,_startTime,CkWallTimer());
		//TODO: this might be a problem.. change made for leanMD
//		CpvAccess(_currentObj) = obj;
	DEBUG_MEM(CmiMemoryCheck());
		return 1;
	}
	if(env->TN <= obj->mlogData->tProcessed){
		//message already processed
		DEBUG(printf("[%d] Message SN %d TN %d for recver %s being ignored tProcessed %d \n",CkMyPe(),env->SN,env->TN,recver.toString(recverString),obj->mlogData->tProcessed));
//		traceUserBracketEvent(26,_startTime,CkWallTimer());
	DEBUG_MEM(CmiMemoryCheck());
		return 0;
	}
	//message that needs to be processed in the future

//	DEBUG(printf("[%d] Early Message SN %d TN %d tProcessed %d for recver %s stored for future time %.6lf \n",CkMyPe(),env->SN,env->TN,obj->mlogData->tProcessed, recver.toString(recverString),CkWallTimer()));
	//the message cant be processed now put it back in the out of order message Q, 
	//It will be transferred to the main queue later
	CqsEnqueue(CpvAccess(_outOfOrderMessageQueue),env);
//		traceUserBracketEvent(27,_startTime,CkWallTimer());
	DEBUG_MEM(CmiMemoryCheck());
	
	return 0;
}

/**
 * @brief Updates a few variables once a message has been processed.
 */
void postProcessReceivedMessage(Chare *obj,CkObjID &sender,MCount SN,MlogEntry *entry){
	DEBUG(char senderString[100]);
	if(obj){
		if(sender.guessPE() == CkMyPe()){
			if(entry != NULL){
				entry->env = NULL;
			}
		}
		obj->mlogData->tProcessed++;
/*		DEBUG(int qLength = CqsLength((Queue )CpvAccess(CsdSchedQueue)));		
		DEBUG(printf("[%d] Message SN %d %s has been processed  tProcessed %d scheduler queue length %d\n",CkMyPe(),SN,obj->mlogData->objID.toString(senderString),obj->mlogData->tProcessed,qLength));		*/
//		CpvAccess(_currentObj)= NULL;
	}
	DEBUG_MEM(CmiMemoryCheck());
}

/***
	Helpers for the handlers and message logging methods
***/

void generalCldEnqueue(int destPE,envelope *env,int _infoIdx){
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
/** This method is used to retry the ticket requests
 * that had been queued up earlier
 * */

int calledRetryTicketRequest=0;

void retryTicketRequestTimer(void *_dummy,double _time){
		calledRetryTicketRequest=0;
		retryTicketRequest(_dummy,_time);
}

void retryTicketRequest(void *_dummy,double curWallTime){	
	double start = CkWallTimer();
	DEBUG_MEM(CmiMemoryCheck());
	int length = CpvAccess(_delayedTicketRequests)->length();
	for(int i=0;i<length;i++){
		TicketRequest *ticketRequest = CpvAccess(_delayedTicketRequests)->deq();
		if(ticketRequest){
			char senderString[100],recverString[100];
			DEBUGRESTART(printf("[%d] RetryTicketRequest for ticket %p sender %s recver %s SN %d at %.6lf \n",CkMyPe(),ticketRequest,ticketRequest->sender.toString(senderString),ticketRequest->recver.toString(recverString), ticketRequest->SN, CmiWallTimer()));
			DEBUG_MEM(CmiMemoryCheck());
			_processTicketRequest(ticketRequest);
		  CmiFree(ticketRequest);
			DEBUG_MEM(CmiMemoryCheck());
		}	
	}	
	for(int i=0;i<CpvAccess(_delayedLocalTicketRequests)->length();i++){
		MlogEntry *entry = CpvAccess(_delayedLocalTicketRequests)->deq();
		ticketLogLocalMessage(entry);
	}
	int qLength = CqsLength((Queue )CpvAccess(CsdSchedQueue));
//	int converse_qLength = CmiGetNonLocalLength();
	
//	DEBUG(printf("[%d] Total RetryTicketRequest took %.6lf scheduler queue length %d converse queue length %d \n",CkMyPe(),CkWallTimer()-start,qLength,converse_qLength));

/*	PingMsg pingMsg;
	pingMsg.PE = CkMyPe();
	CmiSetHandler(&pingMsg,_pingHandlerIdx);
	if(CkMyPe() == 0 || CkMyPe() == CkNumPes() -1){
		for(int i=0;i<CkNumPes();i++){
			if(i != CkMyPe()){
				CmiSyncSend(i,sizeof(PingMsg),(char *)&pingMsg);
			}
		}
	}*/	
	//TODO: change this back to 100
	if(calledRetryTicketRequest == 0){
		CcdCallFnAfter(retryTicketRequestTimer,NULL,500);	
		calledRetryTicketRequest =1;
	}
	DEBUG_MEM(CmiMemoryCheck());
}

void _pingHandler(CkPingMsg *msg){
	printf("[%d] Received Ping from %d\n",CkMyPe(),msg->PE);
	CmiFree(msg);
}


/*****************************************************************************
	Checkpointing methods..
		Pack all the data on a processor and send it to the buddy periodically
		Also used to throw away message logs
*****************************************************************************/
CkVec<TProcessedLog> processedTicketLog;
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

void startMlogCheckpoint(void *_dummy,double curWallTime){
	double _startTime = CkWallTimer();
	checkpointCount++;
/*	if(checkpointCount == 3 && CmiMyPe() == 4 && restarted == 0){
		kill(getpid(),SIGKILL);
	}*/
	if(CmiNumPes() < 256 || CmiMyPe() == 0){
		printf("[%d] starting checkpoint at %.6lf CmiTimer %.6lf \n",CkMyPe(),CmiWallTimer(),CmiTimer());
	}
	PUP::sizer psizer;
	DEBUG_MEM(CmiMemoryCheck());

	psizer | checkpointCount;
	
	CkPupROData(psizer);
	DEBUG_MEM(CmiMemoryCheck());
	CkPupGroupData(psizer,CmiTrue);
	DEBUG_MEM(CmiMemoryCheck());
	CkPupNodeGroupData(psizer,CmiTrue);
	DEBUG_MEM(CmiMemoryCheck());
	pupArrayElementsSkip(psizer,CmiTrue,NULL);
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
	
	CkPupROData(pBuf);
	CkPupGroupData(pBuf,CmiTrue);
	CkPupNodeGroupData(pBuf,CmiTrue);
	pupArrayElementsSkip(pBuf,CmiTrue,NULL);

	unAckedCheckpoint=1;
	CmiSetHandler(msg,_storeCheckpointHandlerIdx);
	CmiSyncSendAndFree(getCheckPointPE(),totalSize,msg);
	
	/*
		Store the highest Ticket number processed for each chare on this processor
	*/
	processedTicketLog.removeAll();
	forAllCharesDo(buildProcessedTicketLog,(void *)&processedTicketLog);
	if(CmiNumPes() < 256 || CmiMyPe() == 0){
		printf("[%d] finishing checkpoint at %.6lf CmiTimer %.6lf with dataSize %d\n",CkMyPe(),CmiWallTimer(),CmiTimer(),dataSize);
	}

	if(CkMyPe() ==  0 && onGoingLoadBalancing==0 ){
		lastCompletedAlarm = curWallTime;
		CcdCallFnAfter(checkpointAlarm,NULL,chkptPeriod);
	}
	traceUserBracketEvent(28,_startTime,CkWallTimer());
};

void buildProcessedTicketLog(void *data,ChareMlogData *mlogData){
	CkVec<TProcessedLog> *log = (	CkVec<TProcessedLog> *)data;
	TProcessedLog logEntry;
	logEntry.recver = mlogData->objID;
	logEntry.tProcessed = mlogData->tProcessed;
	log->push_back(logEntry);
	char objString[100];
	DEBUG(printf("[%d] Tickets lower than %d to be thrown away for %s \n",CkMyPe(),logEntry.tProcessed,logEntry.recver.toString(objString)));
}

class ElementPacker : public CkLocIterator {
private:
	CkLocMgr *locMgr;
	PUP::er &p;
public:
		ElementPacker(CkLocMgr* mgr_, PUP::er &p_):locMgr(mgr_),p(p_){};
		void addLocation(CkLocation &loc) {
			CkArrayIndex idx=loc.getIndex();
			CkGroupID gID = locMgr->ckGetGroupID();
			p|gID;	    // store loc mgr's GID as well for easier restore
			p|idx;
			p|loc;
    }
};

/**
 * Pups all the array elements in this processor.
 */
void pupArrayElementsSkip(PUP::er &p, CmiBool create, MigrationRecord *listToSkip,int listsize){
	int numElements,i;
	int numGroups = CkpvAccess(_groupIDTable)->size();	
	if(!p.isUnpacking()){
		numElements = CkCountArrayElements();
	}	
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
			CkArrayIndex idx;
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


void writeCheckpointToDisk(int size,char *chkpt){
	char fNameTemp[100];
	sprintf(fNameTemp,"%s/mlogCheckpoint%d_tmp",checkpointDirectory,CkMyPe());
	int fd = creat(fNameTemp,S_IRWXU);
	int ret = write(fd,chkpt,size);
	CkAssert(ret == size);
	close(fd);
	
	char fName[100];
	sprintf(fName,"%s/mlogCheckpoint%d",checkpointDirectory,CkMyPe());
	unlink(fName);

	rename(fNameTemp,fName);
	
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
	//turning off storing checkpoints
	
	int sendingPE = chkMsg->PE;
	
	CpvAccess(_storedCheckpointData)->buf = chkpt;
	CpvAccess(_storedCheckpointData)->bufSize = chkMsg->dataSize;
	CpvAccess(_storedCheckpointData)->PE = sendingPE;

#ifdef CHECKPOINT_DISK
	//store the checkpoint on disk
	writeCheckpointToDisk(chkMsg->dataSize,chkpt);
	CpvAccess(_storedCheckpointData)->buf = NULL;
	CmiFree(msg);
#endif

	int count=0;
	for(int j=migratedNoticeList.size()-1;j>=0;j--){
		if(migratedNoticeList[j].fromPE == sendingPE){
			migratedNoticeList[j].ackFrom = 1;
		}else{
			CmiAssert("migratedNoticeList entry for processor other than buddy");
		}
		if(migratedNoticeList[j].ackFrom == 1 && migratedNoticeList[j].ackTo == 1){
			migratedNoticeList.remove(j);
			count++;
		}
		
	}
	DEBUG(printf("[%d] For proc %d from number of migratedNoticeList cleared %d checkpointAckHandler %d\n",CmiMyPe(),sendingPE,count,_checkpointAckHandlerIdx));
	
	CheckPointAck ackMsg;
	ackMsg.PE = CkMyPe();
	ackMsg.dataSize = CpvAccess(_storedCheckpointData)->bufSize;
	CmiSetHandler(&ackMsg,_checkpointAckHandlerIdx);
	CmiSyncSend(sendingPE,sizeof(CheckPointAck),(char *)&ackMsg);
	
	
	
	traceUserBracketEvent(29,_startTime,CkWallTimer());
};


void sendRemoveLogRequests(){
	double _startTime = CkWallTimer();	
	//send out the messages asking senders to throw away message logs below a certain ticket number
	/*
		The remove log request message looks like
		|RemoveLogRequest||List of TProcessedLog|
	*/
	int totalSize = sizeof(RemoveLogRequest)+processedTicketLog.size()*sizeof(TProcessedLog);
	char *requestMsg = (char *)CmiAlloc(totalSize);
	RemoveLogRequest *request = (RemoveLogRequest *)requestMsg;
	request->PE = CkMyPe();
	request->numberObjects = processedTicketLog.size();
	char *listProcessedLogs = &requestMsg[sizeof(RemoveLogRequest)];
	memcpy(listProcessedLogs,(char *)processedTicketLog.getVec(),processedTicketLog.size()*sizeof(TProcessedLog));
	CmiSetHandler(requestMsg,_removeProcessedLogHandlerIdx);
	
	DEBUG_MEM(CmiMemoryCheck());
	for(int i=0;i<CkNumPes();i++){
		CmiSyncSend(i,totalSize,requestMsg);
	}
	CmiFree(requestMsg);

	clearUpMigratedRetainedLists(CmiMyPe());
	//TODO: clear ticketTable
	
	traceUserBracketEvent(30,_startTime,CkWallTimer());
	DEBUG_MEM(CmiMemoryCheck());
}


void _checkpointAckHandler(CheckPointAck *ackMsg){
	DEBUG_MEM(CmiMemoryCheck());
	unAckedCheckpoint=0;
	DEBUG(printf("[%d] CheckPoint Acked from PE %d with size %d onGoingLoadBalancing %d \n",CkMyPe(),ackMsg->PE,ackMsg->dataSize,onGoingLoadBalancing));
	DEBUGLB(CkPrintf("[%d] ACK HANDLER with %d\n",CkMyPe(),onGoingLoadBalancing));	
	if(onGoingLoadBalancing){
		onGoingLoadBalancing = 0;
		finishedCheckpointLoadBalancing();
	}else{
		sendRemoveLogRequests();
	}
	CmiFree(ackMsg);
	
};

void removeProcessedLogs(void *_data,ChareMlogData *mlogData){
	DEBUG_MEM(CmiMemoryCheck());
	CmiMemoryCheck();
	char *data = (char *)_data;
	RemoveLogRequest *request = (RemoveLogRequest *)data;
	TProcessedLog *list = (TProcessedLog *)(&data[sizeof(RemoveLogRequest)]);
	CkQ<MlogEntry *> *mlog = mlogData->getMlog();

	int count=0;
	for(int i=0;i<mlog->length();i++){
		MlogEntry *logEntry = mlog->deq();
		int match=0;
		for(int j=0;j<request->numberObjects;j++){
			if(logEntry->env == NULL || (logEntry->env->recver == list[j].recver && logEntry->env->TN > 0 && logEntry->env->TN < list[j].tProcessed && logEntry->unackedLocal != 1)){
				//this log Entry should be removed
				match = 1;
				break;
			}
		}
		char senderString[100],recverString[100];
//		DEBUG(CkPrintf("[%d] Message sender %s recver %s TN %d removed %d PE %d\n",CkMyPe(),logEntry->env->sender.toString(senderString),logEntry->env->recver.toString(recverString),logEntry->env->TN,match,request->PE));
		if(match){
			count++;
			delete logEntry;
		}else{
			mlog->enq(logEntry);
		}
	}
	if(count > 0){
		char nameString[100];
		DEBUG(printf("[%d] Removed %d processed Logs for %s\n",CkMyPe(),count,mlogData->objID.toString(nameString)));
	}
	DEBUG_MEM(CmiMemoryCheck());
	CmiMemoryCheck();
}

void _removeProcessedLogHandler(char *requestMsg){
	double start = CkWallTimer();
	forAllCharesDo(removeProcessedLogs,requestMsg);
	// printf("[%d] Removing Processed logs took %.6lf \n",CkMyPe(),CkWallTimer()-start);
	RemoveLogRequest *request = (RemoveLogRequest *)requestMsg;
	DEBUG(printf("[%d] Removing Processed logs for proc %d took %.6lf \n",CkMyPe(),request->PE,CkWallTimer()-start));
	//this assumes the buddy relationship between processors is symmetric. TODO:remove this assumption later
	if(request->PE == getCheckPointPE()){
		TProcessedLog *list = (TProcessedLog *)(&requestMsg[sizeof(RemoveLogRequest)]);
		CkQ<LocalMessageLog> *localQ = CpvAccess(_localMessageLog);
		CkQ<LocalMessageLog> *tempQ = new CkQ<LocalMessageLog>;
		int count=0;
/*		DEBUG(for(int j=0;j<request->numberObjects;j++){)
		DEBUG(char nameString[100];)
			DEBUG(printf("[%d] Remove local message logs for %s with TN less than %d\n",CkMyPe(),list[j].recver.toString(nameString),list[j].tProcessed));
		DEBUG(})*/
		for(int i=0;i<localQ->length();i++){
			LocalMessageLog localLogEntry = (*localQ)[i];
			if(!fault_aware(localLogEntry.recver)){
				CmiAbort("Non fault aware logEntry recver found while clearing old local logs");
			}
			bool keep = true;
			for(int j=0;j<request->numberObjects;j++){				
				if(localLogEntry.recver == list[j].recver && localLogEntry.TN > 0 && localLogEntry.TN < list[j].tProcessed){
					keep = false;
					break;
				}
			}	
			if(keep){
				tempQ->enq(localLogEntry);
			}else{
				count++;
			}
		}
		delete localQ;
		CpvAccess(_localMessageLog) = tempQ;
		DEBUG(printf("[%d] %d Local logs for proc %d deleted on buddy \n",CkMyPe(),count,request->PE));
	}

	/*
		Clear up the retainedObjectList and the migratedNoticeList that were created during load balancing
	*/
	CmiMemoryCheck();
	clearUpMigratedRetainedLists(request->PE);
	
	traceUserBracketEvent(20,start,CkWallTimer());
	CmiFree(requestMsg);	
};


void clearUpMigratedRetainedLists(int PE){
	int count=0;
	CmiMemoryCheck();
	
	for(int j=migratedNoticeList.size()-1;j>=0;j--){
		if(migratedNoticeList[j].toPE == PE){
			migratedNoticeList[j].ackTo = 1;
		}
		if(migratedNoticeList[j].ackFrom == 1 && migratedNoticeList[j].ackTo == 1){
			migratedNoticeList.remove(j);
			count++;
		}
	}
	DEBUG(printf("[%d] For proc %d to number of migratedNoticeList cleared %d \n",CmiMyPe(),PE,count));
	
	for(int j=retainedObjectList.size()-1;j>=0;j--){
		if(retainedObjectList[j]->migRecord.toPE == PE){
			RetainedMigratedObject *obj = retainedObjectList[j];
			DEBUG(printf("[%d] Clearing retainedObjectList %d to PE %d obj %p msg %p\n",CmiMyPe(),j,PE,obj,obj->msg));
			retainedObjectList.remove(j);
			if(obj->msg != NULL){
				CmiMemoryCheck();
				CmiFree(obj->msg);
			}
			delete obj;
		}
	}
}

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

	// if we are using team-based message logging, all members of the group have to be restarted
	if(teamSize > 1){
		for(int i=(CkMyPe()/teamSize)*teamSize; i<((CkMyPe()/teamSize)+1)*teamSize; i++){
			if(i != CkMyPe() && i < CkNumPes()){
				// sending a message to the team member
				msg.PE = CkMyPe();
			    CmiSetHandler(&msg,_restartHandlerIdx);
			    CmiSyncSend(i,sizeof(RestartRequest),(char *)&msg);
			}
		}
	}

	// requesting the latest checkpoint from its buddy
	msg.PE = CkMyPe();
	CmiSetHandler(&msg,_getCheckpointHandlerIdx);
	CmiSyncSend(getCheckPointPE(),sizeof(RestartRequest),(char *)&msg);
};

/**
 * Function to restart this processor.
 * The handler is invoked by a member of its same team in message logging.
 */
void _restartHandler(RestartRequest *restartMsg){
	int i;
	int numGroups = CkpvAccess(_groupIDTable)->size();
	RestartRequest msg;
	
	fprintf(stderr,"[%d] Restart-team started at %.6lf \n",CkMyPe(),CmiWallTimer());

    // setting the restart flag
	_restartFlag = 1;

	// flushing all buffers
	//TEST END
/*	CkPrintf("[%d] HERE numGroups = %d\n",CkMyPe(),numGroups);
	CKLOCMGR_LOOP(mgr->flushAllRecs(););	
	for(int i=0;i<numGroups;i++){
    	CkGroupID gID = (*CkpvAccess(_groupIDTable))[i];
		IrrGroup *obj = CkpvAccess(_groupTable)->find(gID).getObj();
		obj->flushStates();
		obj->ckJustMigrated();
	}*/

    // requesting the latest checkpoint from its buddy
	msg.PE = CkMyPe();
	CmiSetHandler(&msg,_getRestartCheckpointHandlerIdx);
	CmiSyncSend(getCheckPointPE(),sizeof(RestartRequest),(char *)&msg);
}


/**
 * Gets the stored checkpoint but calls another function in the sender.
 */
void _getRestartCheckpointHandler(RestartRequest *restartMsg){

	// retrieving the stored checkpoint
	StoredCheckpoint *storedChkpt =	CpvAccess(_storedCheckpointData);

	// making sure it is its buddy who is requesting the checkpoint
	CkAssert(restartMsg->PE == storedChkpt->PE);

	storedRequest = restartMsg;
	verifyAckTotal = 0;

	for(int i=0;i<migratedNoticeList.size();i++){
		if(migratedNoticeList[i].fromPE == restartMsg->PE){
//			if(migratedNoticeList[i].ackFrom == 0 && migratedNoticeList[i].ackTo == 0){
			if(migratedNoticeList[i].ackFrom == 0){
				//need to verify if the object actually exists .. it might not
				//have been acked but it might exist on it
				VerifyAckMsg msg;
				msg.migRecord = migratedNoticeList[i];
				msg.index = i;
				msg.fromPE = CmiMyPe();
				CmiPrintf("[%d] Verify  gid %d idx %s from proc %d\n",CmiMyPe(),migratedNoticeList[i].gID.idx,idx2str(migratedNoticeList[i].idx),migratedNoticeList[i].toPE);
				CmiSetHandler(&msg,_verifyAckRequestHandlerIdx);
				CmiSyncSend(migratedNoticeList[i].toPE,sizeof(VerifyAckMsg),(char *)&msg);
				verifyAckTotal++;
			}
		}
	}

	// sending the checkpoint back to its buddy	
	if(verifyAckTotal == 0){
		sendCheckpointData(MLOG_RESTARTED);
	}
	verifyAckCount = 0;
}

/**
 * Receives the checkpoint coming from its buddy. This is the case of restart for one team member that did not crash.
 */
void _recvRestartCheckpointHandler(char *_restartData){
	RestartProcessorData *restartData = (RestartProcessorData *)_restartData;
	MigrationRecord *migratedAwayElements;

	globalLBID = restartData->lbGroupID;
	
	restartData->restartWallTime *= 1000;
	adjustChkptPeriod = restartData->restartWallTime/(double) chkptPeriod - floor(restartData->restartWallTime/(double) chkptPeriod);
	adjustChkptPeriod = (double )chkptPeriod*(adjustChkptPeriod);
	if(adjustChkptPeriod < 0) adjustChkptPeriod = 0;

	
	printf("[%d] Team Restart Checkpointdata received from PE %d at %.6lf with checkpointSize %d\n",CkMyPe(),restartData->PE,CmiWallTimer(),restartData->checkPointSize);
	char *buf = &_restartData[sizeof(RestartProcessorData)];
	
	if(restartData->numMigratedAwayElements != 0){
		migratedAwayElements = new MigrationRecord[restartData->numMigratedAwayElements];
		memcpy(migratedAwayElements,buf,restartData->numMigratedAwayElements*sizeof(MigrationRecord));
		printf("[%d] Number of migratedaway elements %d\n",CmiMyPe(),restartData->numMigratedAwayElements);
		buf = &buf[restartData->numMigratedAwayElements*sizeof(MigrationRecord)];
	}

	// turning on the team recovery flag
	forAllCharesDo(setTeamRecovery,NULL);
	
	PUP::fromMem pBuf(buf);
	pBuf | checkpointCount;
	CkPupROData(pBuf);
	CkPupGroupData(pBuf,CmiFalse);
	CkPupNodeGroupData(pBuf,CmiFalse);
	pupArrayElementsSkip(pBuf,CmiFalse,NULL);
	CkAssert(pBuf.size() == restartData->checkPointSize);
	printf("[%d] Restart Objects created from CheckPointData at %.6lf \n",CkMyPe(),CmiWallTimer());
	
	// turning off the team recovery flag
	forAllCharesDo(unsetTeamRecovery,NULL);

	// initializing a few variables for handling local messages
	forAllCharesDo(initializeRestart,NULL);
	
	//store the restored local message log in a vector
	buf = &buf[restartData->checkPointSize];	
	for(int i=0;i<restartData->numLocalMessages;i++){
		LocalMessageLog logEntry;
		memcpy(&logEntry,buf,sizeof(LocalMessageLog));
		
		Chare *recverObj = (Chare *)logEntry.recver.getObject();
		if(recverObj!=NULL){
			recverObj->mlogData->addToRestoredLocalQ(&logEntry);
			recverObj->mlogData->receivedTNs->push_back(logEntry.TN);
			char senderString[100];
			char recverString[100];
			DEBUGRESTART(printf("[%d] Received local message log sender %s recver %s SN %d  TN %d\n",CkMyPe(),logEntry.sender.toString(senderString),logEntry.recver.toString(recverString),logEntry.SN,logEntry.TN));
		}else{
//			DEBUGRESTART(printf("Object receiving local message doesnt exist on restarted processor .. ignoring it"));
		}
		buf = &buf[sizeof(LocalMessageLog)];
	}

	forAllCharesDo(sortRestoredLocalMsgLog,NULL);
	CmiFree(_restartData);	

	/*HERE _initDone();

	getGlobalStep(globalLBID);
	
	countUpdateHomeAcks = 0;
	RestartRequest updateHomeRequest;
	updateHomeRequest.PE = CmiMyPe();
	CmiSetHandler (&updateHomeRequest,_updateHomeRequestHandlerIdx);
	for(int i=0;i<CmiNumPes();i++){
		if(i != CmiMyPe()){
			CmiSyncSend(i,sizeof(RestartRequest),(char *)&updateHomeRequest);
		}
	}
*/


	// Send out the request to resend logged messages to all other processors
	CkVec<TProcessedLog> objectVec;
	forAllCharesDo(createObjIDList, (void *)&objectVec);
	int numberObjects = objectVec.size();
	
	/*
		resendMsg layout |ResendRequest|Array of TProcessedLog|
	*/
	int totalSize = sizeof(ResendRequest)+numberObjects*sizeof(TProcessedLog);
	char *resendMsg = (char *)CmiAlloc(totalSize);	

	ResendRequest *resendReq = (ResendRequest *)resendMsg;
	resendReq->PE =CkMyPe(); 
	resendReq->numberObjects = numberObjects;
	char *objList = &resendMsg[sizeof(ResendRequest)];
	memcpy(objList,objectVec.getVec(),numberObjects*sizeof(TProcessedLog));
	

	/* test for parallel restart migrate away object**/
//	if(parallelRestart){
//		distributeRestartedObjects();
//		printf("[%d] Redistribution of objects done at %.6lf \n",CkMyPe(),CmiWallTimer());
//	}
	
	/*	To make restart work for load balancing.. should only
	be used when checkpoint happens along with load balancing
	**/
//	forAllCharesDo(resumeFromSyncRestart,NULL);

	CentralLB *lb = (CentralLB *)CkpvAccess(_groupTable)->find(globalLBID).getObj();
	CpvAccess(_currentObj) = lb;
	lb->ReceiveDummyMigration(restartDecisionNumber);

	sleep(10);
	
	CmiSetHandler(resendMsg,_resendMessagesHandlerIdx);
	for(int i=0;i<CkNumPes();i++){
		if(i != CkMyPe()){
			CmiSyncSend(i,totalSize,resendMsg);
		}	
	}
	_resendMessagesHandler(resendMsg);

}


void CkMlogRestartDouble(void *,double){
	CkMlogRestart(NULL,NULL);
};

//TML: restarting from local (group) failure
void CkMlogRestartLocal(){
    CkMlogRestart(NULL,NULL);
};


void readCheckpointFromDisk(int size,char *buf){
	char fName[100];
	sprintf(fName,"%s/mlogCheckpoint%d",checkpointDirectory,CkMyPe());

	int fd = open(fName,O_RDONLY);
	int count=0;
	while(count < size){
		count += read(fd,&buf[count],size-count);
	}
	close(fd);
	
};



/**
 * Gets the stored checkpoint for its buddy processor.
 */
void _getCheckpointHandler(RestartRequest *restartMsg){

	// retrieving the stored checkpoint
	StoredCheckpoint *storedChkpt =	CpvAccess(_storedCheckpointData);

	// making sure it is its buddy who is requesting the checkpoint
	CkAssert(restartMsg->PE == storedChkpt->PE);

	storedRequest = restartMsg;
	verifyAckTotal = 0;

	for(int i=0;i<migratedNoticeList.size();i++){
		if(migratedNoticeList[i].fromPE == restartMsg->PE){
//			if(migratedNoticeList[i].ackFrom == 0 && migratedNoticeList[i].ackTo == 0){
			if(migratedNoticeList[i].ackFrom == 0){
				//need to verify if the object actually exists .. it might not
				//have been acked but it might exist on it
				VerifyAckMsg msg;
				msg.migRecord = migratedNoticeList[i];
				msg.index = i;
				msg.fromPE = CmiMyPe();
				CmiPrintf("[%d] Verify  gid %d idx %s from proc %d\n",CmiMyPe(),migratedNoticeList[i].gID.idx,idx2str(migratedNoticeList[i].idx),migratedNoticeList[i].toPE);
				CmiSetHandler(&msg,_verifyAckRequestHandlerIdx);
				CmiSyncSend(migratedNoticeList[i].toPE,sizeof(VerifyAckMsg),(char *)&msg);
				verifyAckTotal++;
			}
		}
	}

	// sending the checkpoint back to its buddy	
	if(verifyAckTotal == 0){
		sendCheckpointData(MLOG_CRASHED);
	}
	verifyAckCount = 0;
}


void _verifyAckRequestHandler(VerifyAckMsg *verifyRequest){
	CkLocMgr *locMgr =  (CkLocMgr*)CkpvAccess(_groupTable)->find(verifyRequest->migRecord.gID).getObj();
	CkLocRec *rec = locMgr->elementNrec(verifyRequest->migRecord.idx);
	if(rec != NULL && rec->type() == CkLocRec::local){
			//this location exists on this processor
			//and needs to be removed	
			CkLocRec_local *localRec = (CkLocRec_local *) rec;
			CmiPrintf("[%d] Found element gid %d idx %s that needs to be removed\n",CmiMyPe(),verifyRequest->migRecord.gID.idx,idx2str(verifyRequest->migRecord.idx));
			
			int localIdx = localRec->getLocalIndex();
			LBDatabase *lbdb = localRec->getLBDB();
			LDObjHandle ldHandle = localRec->getLdHandle();
				
			locMgr->setDuringMigration(true);
			
			locMgr->reclaim(verifyRequest->migRecord.idx,localIdx);
			lbdb->UnregisterObj(ldHandle);
			
			locMgr->setDuringMigration(false);
			
			verifyAckedRequests++;

	}
	CmiSetHandler(verifyRequest, _verifyAckHandlerIdx);
	CmiSyncSendAndFree(verifyRequest->fromPE,sizeof(VerifyAckMsg),(char *)verifyRequest);
};


void _verifyAckHandler(VerifyAckMsg *verifyReply){
	int index = 	verifyReply->index;
	migratedNoticeList[index] = verifyReply->migRecord;
	verifyAckCount++;
	CmiPrintf("[%d] VerifyReply received %d for  gid %d idx %s from proc %d\n",CmiMyPe(),migratedNoticeList[index].ackTo, migratedNoticeList[index].gID,idx2str(migratedNoticeList[index].idx),migratedNoticeList[index].toPE);
	if(verifyAckCount == verifyAckTotal){
		sendCheckpointData(MLOG_CRASHED);
	}
}


/**
 * Sends the checkpoint to its buddy. The mode distinguishes between the two cases:
 * MLOG_RESTARTED: sending the checkpoint to a team member that did not crash but is restarting.
 * MLOG_CRASHED: sending the checkpoint to the processor that crashed.
 */
void sendCheckpointData(int mode){	
	RestartRequest *restartMsg = storedRequest;
	StoredCheckpoint *storedChkpt = 	CpvAccess(_storedCheckpointData);
	int numMigratedAwayElements = migratedNoticeList.size();
	if(migratedNoticeList.size() != 0){
			printf("[%d] size of migratedNoticeList %d\n",CmiMyPe(),migratedNoticeList.size());
//			CkAssert(migratedNoticeList.size() == 0);
	}
	
	
	int totalSize = sizeof(RestartProcessorData)+storedChkpt->bufSize;
	
	DEBUGRESTART(CkPrintf("[%d] Sending out checkpoint for processor %d size %d \n",CkMyPe(),restartMsg->PE,totalSize);)
	CkPrintf("[%d] Sending out checkpoint for processor %d size %d \n",CkMyPe(),restartMsg->PE,totalSize);
	
	CkQ<LocalMessageLog > *localMsgQ = CpvAccess(_localMessageLog);
	totalSize += localMsgQ->length()*sizeof(LocalMessageLog);
	totalSize += numMigratedAwayElements*sizeof(MigrationRecord);
	
	char *msg = (char *)CmiAlloc(totalSize);
	
	RestartProcessorData *dataMsg = (RestartProcessorData *)msg;
	dataMsg->PE = CkMyPe();
	dataMsg->restartWallTime = CmiTimer();
	dataMsg->checkPointSize = storedChkpt->bufSize;
	
	dataMsg->numMigratedAwayElements = numMigratedAwayElements;
//	dataMsg->numMigratedAwayElements = 0;
	
	dataMsg->numMigratedInElements = 0;
	dataMsg->migratedElementSize = 0;
	dataMsg->lbGroupID = globalLBID;
	/*msg layout 
		|RestartProcessorData|List of Migrated Away ObjIDs|CheckpointData|CheckPointData for objects migrated in|
		Local MessageLog|
	*/
	//store checkpoint data
	char *buf = &msg[sizeof(RestartProcessorData)];

	if(dataMsg->numMigratedAwayElements != 0){
		memcpy(buf,migratedNoticeList.getVec(),migratedNoticeList.size()*sizeof(MigrationRecord));
		buf = &buf[migratedNoticeList.size()*sizeof(MigrationRecord)];
	}
	

#ifdef CHECKPOINT_DISK
	readCheckpointFromDisk(storedChkpt->bufSize,buf);
#else	
	memcpy(buf,storedChkpt->buf,storedChkpt->bufSize);
#endif
	buf = &buf[storedChkpt->bufSize];


	//store localmessage Log
	dataMsg->numLocalMessages = localMsgQ->length();
	for(int i=0;i<localMsgQ->length();i++){
		if(!fault_aware(((*localMsgQ)[i]).recver )){
			CmiAbort("Non fault aware localMsgQ");
		}
		memcpy(buf,&(*localMsgQ)[i],sizeof(LocalMessageLog));
		buf = &buf[sizeof(LocalMessageLog)];
	}
	
	if(mode == MLOG_RESTARTED){
		CmiSetHandler(msg,_recvRestartCheckpointHandlerIdx);
		CmiSyncSendAndFree(restartMsg->PE,totalSize,msg);
		CmiFree(restartMsg);
	}else{
		CmiSetHandler(msg,_recvCheckpointHandlerIdx);
		CmiSyncSendAndFree(restartMsg->PE,totalSize,msg);
		CmiFree(restartMsg);
	}
};


// this list is used to create a vector of the object ids of all
//the chares on this processor currently and the highest TN processed by them 
//the first argument is actually a CkVec<TProcessedLog> *
void createObjIDList(void *data,ChareMlogData *mlogData){
	CkVec<TProcessedLog> *list = (CkVec<TProcessedLog> *)data;
	TProcessedLog entry;
	entry.recver = mlogData->objID;
	entry.tProcessed = mlogData->tProcessed;
	list->push_back(entry);
	DEBUG_TEAM(char objString[100]);
	DEBUG_TEAM(CkPrintf("[%d] %s restored with tProcessed set to %d \n",CkMyPe(),mlogData->objID.toString(objString),mlogData->tProcessed));
}


/**
 * Receives the checkpoint data from its buddy, restores the state of all the objects
 * and asks everyone else to update its home.
 */
void _recvCheckpointHandler(char *_restartData){
	RestartProcessorData *restartData = (RestartProcessorData *)_restartData;
	MigrationRecord *migratedAwayElements;

	globalLBID = restartData->lbGroupID;
	
	restartData->restartWallTime *= 1000;
	adjustChkptPeriod = restartData->restartWallTime/(double) chkptPeriod - floor(restartData->restartWallTime/(double) chkptPeriod);
	adjustChkptPeriod = (double )chkptPeriod*(adjustChkptPeriod);
	if(adjustChkptPeriod < 0) adjustChkptPeriod = 0;

	
	printf("[%d] Restart Checkpointdata received from PE %d at %.6lf with checkpointSize %d\n",CkMyPe(),restartData->PE,CmiWallTimer(),restartData->checkPointSize);
	char *buf = &_restartData[sizeof(RestartProcessorData)];
	
	if(restartData->numMigratedAwayElements != 0){
		migratedAwayElements = new MigrationRecord[restartData->numMigratedAwayElements];
		memcpy(migratedAwayElements,buf,restartData->numMigratedAwayElements*sizeof(MigrationRecord));
		printf("[%d] Number of migratedaway elements %d\n",CmiMyPe(),restartData->numMigratedAwayElements);
		buf = &buf[restartData->numMigratedAwayElements*sizeof(MigrationRecord)];
	}
	
	PUP::fromMem pBuf(buf);

	pBuf | checkpointCount;

	CkPupROData(pBuf);
	CkPupGroupData(pBuf,CmiTrue);
	CkPupNodeGroupData(pBuf,CmiTrue);
	pupArrayElementsSkip(pBuf,CmiTrue,NULL);
	CkAssert(pBuf.size() == restartData->checkPointSize);
	printf("[%d] Restart Objects created from CheckPointData at %.6lf \n",CkMyPe(),CmiWallTimer());
	
	forAllCharesDo(initializeRestart,NULL);
	
	//store the restored local message log in a vector
	buf = &buf[restartData->checkPointSize];	
	for(int i=0;i<restartData->numLocalMessages;i++){
		LocalMessageLog logEntry;
		memcpy(&logEntry,buf,sizeof(LocalMessageLog));
		
		Chare *recverObj = (Chare *)logEntry.recver.getObject();
		if(recverObj!=NULL){
			recverObj->mlogData->addToRestoredLocalQ(&logEntry);
			recverObj->mlogData->receivedTNs->push_back(logEntry.TN);
			char senderString[100];
			char recverString[100];
			DEBUGRESTART(printf("[%d] Received local message log sender %s recver %s SN %d  TN %d\n",CkMyPe(),logEntry.sender.toString(senderString),logEntry.recver.toString(recverString),logEntry.SN,logEntry.TN));
		}else{
//			DEBUGRESTART(printf("Object receiving local message doesnt exist on restarted processor .. ignoring it"));
		}
		buf = &buf[sizeof(LocalMessageLog)];
	}

	forAllCharesDo(sortRestoredLocalMsgLog,NULL);

	CmiFree(_restartData);
	
	
	_initDone();

	getGlobalStep(globalLBID);
	
	countUpdateHomeAcks = 0;
	RestartRequest updateHomeRequest;
	updateHomeRequest.PE = CmiMyPe();
	CmiSetHandler (&updateHomeRequest,_updateHomeRequestHandlerIdx);
	for(int i=0;i<CmiNumPes();i++){
		if(i != CmiMyPe()){
			CmiSyncSend(i,sizeof(RestartRequest),(char *)&updateHomeRequest);
		}
	}

}

/**
 * Receives the updateHome ACKs from all other processors. Once everybody
 * has replied, it sends a request to resend the logged messages.
 */
void _updateHomeAckHandler(RestartRequest *updateHomeAck){
	countUpdateHomeAcks++;
	CmiFree(updateHomeAck);
	// one is from the recvglobal step handler .. it is a dummy updatehomeackhandler
	if(countUpdateHomeAcks != CmiNumPes()){
		return;
	}

	// Send out the request to resend logged messages to all other processors
	CkVec<TProcessedLog> objectVec;
	forAllCharesDo(createObjIDList, (void *)&objectVec);
	int numberObjects = objectVec.size();
	
	//	resendMsg layout: |ResendRequest|Array of TProcessedLog|
	int totalSize = sizeof(ResendRequest)+numberObjects*sizeof(TProcessedLog);
	char *resendMsg = (char *)CmiAlloc(totalSize);	

	ResendRequest *resendReq = (ResendRequest *)resendMsg;
	resendReq->PE =CkMyPe(); 
	resendReq->numberObjects = numberObjects;
	char *objList = &resendMsg[sizeof(ResendRequest)];
	memcpy(objList,objectVec.getVec(),numberObjects*sizeof(TProcessedLog));	

	/* test for parallel restart migrate away object**/
	if(parallelRestart){
		distributeRestartedObjects();
		printf("[%d] Redistribution of objects done at %.6lf \n",CkMyPe(),CmiWallTimer());
	}
	
	/*	To make restart work for load balancing.. should only
	be used when checkpoint happens along with load balancing
	**/
//	forAllCharesDo(resumeFromSyncRestart,NULL);

	CentralLB *lb = (CentralLB *)CkpvAccess(_groupTable)->find(globalLBID).getObj();
	CpvAccess(_currentObj) = lb;
	lb->ReceiveDummyMigration(restartDecisionNumber);

	sleep(10);
	
	CmiSetHandler(resendMsg,_resendMessagesHandlerIdx);
	for(int i=0;i<CkNumPes();i++){
		if(i != CkMyPe()){
			CmiSyncSend(i,totalSize,resendMsg);
		}	
	}
	_resendMessagesHandler(resendMsg);
	CmiFree(resendMsg);
};

/**
 * @brief Initializes variables and flags for restarting procedure.
 */
void initializeRestart(void *data, ChareMlogData *mlogData){
	mlogData->resendReplyRecvd = 0;
	mlogData->receivedTNs = new CkVec<MCount>;
	mlogData->restartFlag = 1;
	mlogData->restoredLocalMsgLog.removeAll();
	mlogData->mapTable.empty();
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
		CkArrayIndex myIdx =  mlogData->objID.data.array.idx;
		CkArrayID aid(mlogData->objID.data.array.id);		
		//check if the restarted processor is the home processor for this object
		CkLocMgr *locMgr = aid.ckLocalBranch()->getLocMgr();
		if(locMgr->homePe(myIdx) == PE){
			DEBUGRESTART(printf("[%d] Tell %d of current location of array element",CkMyPe(),PE));
			DEBUGRESTART(myIdx.print());
			informLocationHome(locMgr->getGroupID(),myIdx,PE,CkMyPe());
		}
	}
};


/**
 * Updates the homePe for all chares in this processor.
 */
void _updateHomeRequestHandler(RestartRequest *updateRequest){
	int sender = updateRequest->PE;
	
	forAllCharesDo(updateHomePE,updateRequest);
	
	updateRequest->PE = CmiMyPe();
	CmiSetHandler(updateRequest,_updateHomeAckHandlerIdx);
	CmiSyncSendAndFree(sender,sizeof(RestartRequest),(char *)updateRequest);
	if(sender == getCheckPointPE() && unAckedCheckpoint==1){
		CmiPrintf("[%d] Crashed processor did not ack so need to checkpoint again\n",CmiMyPe());
		checkpointCount--;
		startMlogCheckpoint(NULL,0);
	}
	if(sender == getCheckPointPE()){
		for(int i=0;i<retainedObjectList.size();i++){
			if(retainedObjectList[i]->acked == 0){
				MigrationNotice migMsg;
				migMsg.migRecord = retainedObjectList[i]->migRecord;
				migMsg.record = retainedObjectList[i];
				CmiSetHandler((void *)&migMsg,_receiveMigrationNoticeHandlerIdx);
				CmiSyncSend(getCheckPointPE(),sizeof(migMsg),(char *)&migMsg);
			}
		}
	}
}

/**
 * @brief Fills up the ticket vector for each chare.
 */
void fillTicketForChare(void *data, ChareMlogData *mlogData){
	ResendData *resendData = (ResendData *)data;
	int PE = resendData->PE; //restarted PE
	int count=0;
	CkHashtableIterator *iterator;
	void *objp;
	void *objkey;
	CkObjID *objID;
	SNToTicket *snToTicket;
	Ticket ticket;
	
	// traversing the team table looking up for the maximum TN received	
	iterator = mlogData->teamTable.iterator();
	while( (objp = iterator->next(&objkey)) != NULL ){
		objID = (CkObjID *)objkey;
	
		// traversing the resendData structure to add ticket numbers
		for(int j=0;j<resendData->numberObjects;j++){
			if((*objID) == (resendData->listObjects)[j].recver){
char name[100];
				snToTicket = *(SNToTicket **)objp;
//CkPrintf("[%d] ---> Traversing the resendData for %s start=%u finish=%u \n",CkMyPe(),objID->toString(name),snToTicket->getStartSN(),snToTicket->getFinishSN());
				for(MCount snIndex=snToTicket->getStartSN(); snIndex<=snToTicket->getFinishSN(); snIndex++){
					ticket = snToTicket->get(snIndex);	
					if(ticket.TN > resendData->maxTickets[j]){
						resendData->maxTickets[j] = ticket.TN;
					}
					if(ticket.TN >= (resendData->listObjects)[j].tProcessed){
						//store the TNs that have been since the recver last checkpointed
						resendData->ticketVecs[j].push_back(ticket.TN);
					}
				}
			}
		}
	}

	//releasing the memory for the iterator
	delete iterator;
}


/**
 * @brief Turns on the flag for team recovery that selectively restores
 * particular metadata information.
 */
void setTeamRecovery(void *data, ChareMlogData *mlogData){
	char name[100];
	mlogData->teamRecoveryFlag = 1;	
}

/**
 * @brief Turns off the flag for team recovery.
 */
void unsetTeamRecovery(void *data, ChareMlogData *mlogData){
	mlogData->teamRecoveryFlag = 0;
}

//the data argument is of type ResendData which contains the 
//array of objects on  the restartedProcessor
//this method resends the messages stored in this chare's message log 
//to the restarted processor. It also accumulates the maximum TN
//for all the objects on the restarted processor
void resendMessageForChare(void *data,ChareMlogData *mlogData){
	char nameString[100];
	ResendData *resendData = (ResendData *)data;
	int PE = resendData->PE; //restarted PE
	DEBUGRESTART(printf("[%d] Resend message from %s to processor %d \n",CkMyPe(),mlogData->objID.toString(nameString),PE);)
	int count=0;
	int ticketRequests=0;
	CkQ<MlogEntry *> *log = mlogData->getMlog();
	
	for(int i=0;i<log->length();i++){
		MlogEntry *logEntry = (*log)[i];
		
		// if we sent out the logs of a local message to buddy and he crashed
		//before acking
		envelope *env = logEntry->env;
		if(env == NULL){
			continue;
		}
		if(logEntry->unackedLocal){
			char recverString[100];
			DEBUGRESTART(printf("[%d] Resend Local unacked message from %s to %s SN %d TN %d \n",CkMyPe(),env->sender.toString(nameString),env->recver.toString(recverString),env->SN,env->TN);)
			sendLocalMessageCopy(logEntry);
		}
		//looks like near a crash messages between uninvolved processors can also get lost. Resend ticket requests as a result
		if(env->TN <= 0){
			//ticket not yet replied send it out again
			sendTicketRequest(env->sender,env->recver,logEntry->destPE,logEntry,env->SN,0,1);
		}
		
		if(env->recver.type != TypeInvalid){
			int flag = 0;//marks if any of the restarted objects matched this log entry
			for(int j=0;j<resendData->numberObjects;j++){
				if(env->recver == (resendData->listObjects)[j].recver){
					flag = 1;
					//message has a valid TN
					if(env->TN > 0){
						//store maxTicket
						if(env->TN > resendData->maxTickets[j]){
							resendData->maxTickets[j] = env->TN;
						}
						//if the TN for this entry is more than the TN processed, send the message out
						if(env->TN >= (resendData->listObjects)[j].tProcessed){
							//store the TNs that have been since the recver last checkpointed
							resendData->ticketVecs[j].push_back(env->TN);
							
							if(PE != CkMyPe()){
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
							char senderString[100];
							DEBUGRESTART(printf("[%d] Resent message sender %s recver %s SN %d TN %d \n",CkMyPe(),env->sender.toString(senderString),env->recver.toString(nameString),env->SN,env->TN));
							count++;
						}	
					}else{
/*					//the message didnt get a ticket the last time and needs to start with a ticket request
						DEBUGRESTART(printf("[%d] Resent ticket request SN %d to %s needs ticket at %d in logQ \n",CkMyPe(),env->SN,env->recver.toString(nameString),i));
						//generateCommonTicketRequest(env->recver,env,PE,logEntry->_infoIdx);						
						CkAssert(logEntry->destPE != CkMyPe());
						
						sendTicketRequest(env->sender,env->recver,PE,logEntry,env->SN,1);
						
						ticketRequests++;*/
					}
				}
			}//end of for loop of objects
			
		}	
	}
	DEBUGRESTART(printf("[%d] Resent  %d/%d (%d) messages  from %s to processor %d \n",CkMyPe(),count,log->length(),ticketRequests,mlogData->objID.toString(nameString),PE);)	
}

/**
 * Resends the messages since the last checkpoint to the list of objects included in the 
 * request.
 */
void _resendMessagesHandler(char *msg){
	ResendData d;
	ResendRequest *resendReq = (ResendRequest *)msg;

	// building the reply message
	char *listObjects = &msg[sizeof(ResendRequest)];
	d.numberObjects = resendReq->numberObjects;
	d.PE = resendReq->PE;
	d.listObjects = (TProcessedLog *)listObjects;
	d.maxTickets = new MCount[d.numberObjects];
	d.ticketVecs = new CkVec<MCount>[d.numberObjects];
	for(int i=0;i<d.numberObjects;i++){
		d.maxTickets[i] = 0;
	}

	//Check if any of the retained objects need to be recreated
	//If they have not been recreated on the restarted processor
	//they need to be recreated on this processor
	int count=0;
	for(int i=0;i<retainedObjectList.size();i++){
		if(retainedObjectList[i]->migRecord.toPE == d.PE){
			count++;
			int recreate=1;
			for(int j=0;j<d.numberObjects;j++){
				if(d.listObjects[j].recver.type != TypeArray ){
					continue;
				}
				CkArrayID aid(d.listObjects[j].recver.data.array.id);		
				CkLocMgr *locMgr = aid.ckLocalBranch()->getLocMgr();
				if(retainedObjectList[i]->migRecord.gID == locMgr->getGroupID()){
					if(retainedObjectList[i]->migRecord.idx == d.listObjects[j].recver.data.array.idx){
						recreate = 0;
						break;
					}
				}
			}
			CmiPrintf("[%d] Object migrated away but did not checkpoint recreate %d locmgrid %d idx %s\n",CmiMyPe(),recreate,retainedObjectList[i]->migRecord.gID.idx,idx2str(retainedObjectList[i]->migRecord.idx));
			if(recreate){
				donotCountMigration=1;
				_receiveMlogLocationHandler(retainedObjectList[i]->msg);
				donotCountMigration=0;
				CkLocMgr *locMgr =  (CkLocMgr*)CkpvAccess(_groupTable)->find(retainedObjectList[i]->migRecord.gID).getObj();
				int homePE = locMgr->homePe(retainedObjectList[i]->migRecord.idx);
				informLocationHome(retainedObjectList[i]->migRecord.gID,retainedObjectList[i]->migRecord.idx,homePE,CmiMyPe());
				sendDummyMigration(d.PE,globalLBID,retainedObjectList[i]->migRecord.gID,retainedObjectList[i]->migRecord.idx,CmiMyPe());
				CkLocRec *rec = locMgr->elementRec(retainedObjectList[i]->migRecord.idx);
				CmiAssert(rec->type() == CkLocRec::local);
				CkVec<CkMigratable *> eltList;
				locMgr->migratableList((CkLocRec_local *)rec,eltList);
				for(int j=0;j<eltList.size();j++){
					if(eltList[j]->mlogData->toResumeOrNot == 1 && eltList[j]->mlogData->resumeCount < globalResumeCount){
						CpvAccess(_currentObj) = eltList[j];
						eltList[j]->ResumeFromSync();
					}
				}
				retainedObjectList[i]->msg=NULL;	
			}
		}
	}
	
	if(count > 0){
//		CmiAbort("retainedObjectList for restarted processor not empty");
	}
	
	DEBUG(printf("[%d] Received request to Resend Messages to processor %d numberObjects %d at %.6lf\n",CkMyPe(),resendReq->PE,resendReq->numberObjects,CmiWallTimer()));


	//TML: examines the origin processor to determine if it belongs to the same group.
	// In that case, it only returns the maximum ticket received for each object in the list.
	if(isTeamLocal(resendReq->PE) && CkMyPe() != resendReq->PE)
		forAllCharesDo(fillTicketForChare,&d);
	else
		forAllCharesDo(resendMessageForChare,&d);

	//send back the maximum ticket number for a message sent to each object on the 
	//restarted processor
	//Message: |ResendRequest|List of CkObjIDs|List<#number of objects in vec,TN of tickets seen>|
	
	int totalTNStored=0;
	for(int i=0;i<d.numberObjects;i++){
		totalTNStored += d.ticketVecs[i].size();
	}
	
	int totalSize = sizeof(ResendRequest)+d.numberObjects*(sizeof(CkObjID)+sizeof(int)) + totalTNStored*sizeof(MCount);
	char *resendReplyMsg = (char *)CmiAlloc(totalSize);
	
	ResendRequest *resendReply = (ResendRequest *)resendReplyMsg;
	resendReply->PE = CkMyPe();
	resendReply->numberObjects = d.numberObjects;
	
	char *replyListObjects = &resendReplyMsg[sizeof(ResendRequest)];
	CkObjID *replyObjects = (CkObjID *)replyListObjects;
	for(int i=0;i<d.numberObjects;i++){
		replyObjects[i] = d.listObjects[i].recver;
	}
	
	char *ticketList = &replyListObjects[sizeof(CkObjID)*d.numberObjects];
	for(int i=0;i<d.numberObjects;i++){
		int vecsize = d.ticketVecs[i].size();
		memcpy(ticketList,&vecsize,sizeof(int));
		ticketList = &ticketList[sizeof(int)];
		memcpy(ticketList,d.ticketVecs[i].getVec(),sizeof(MCount)*vecsize);
		ticketList = &ticketList[sizeof(MCount)*vecsize];
	}	

	CmiSetHandler(resendReplyMsg,_resendReplyHandlerIdx);
	CmiSyncSendAndFree(d.PE,totalSize,(char *)resendReplyMsg);
	
/*	
	if(verifyAckRequestsUnacked){
		CmiPrintf("[%d] verifyAckRequestsUnacked %d call dummy migrates\n",CmiMyPe(),verifyAckRequestsUnacked);
		for(int i=0;i<verifyAckRequestsUnacked;i++){
			CentralLB *lb = (CentralLB *)CkpvAccess(_groupTable)->find(globalLBID).getObj();
			LDObjHandle h;
			lb->Migrated(h,1);
		}
	}
	
	verifyAckRequestsUnacked=0;*/
	
	delete [] d.maxTickets;
	delete [] d.ticketVecs;
	if(resendReq->PE != CkMyPe()){
		CmiFree(msg);
	}	
//	CmiPrintf("[%d] End of resend Request \n",CmiMyPe());
	lastRestart = CmiWallTimer();
}

void sortVec(CkVec<MCount> *TNvec);
int searchVec(CkVec<MCount> *TNVec,MCount searchTN);

/**
 * @brief Receives the tickets assigned to message to other objects.
 */
void _resendReplyHandler(char *msg){	
	/**
		need to rewrite this method to deal with parallel restart
	*/
	ResendRequest *resendReply = (ResendRequest *)msg;
	CkObjID *listObjects = (CkObjID *)( &msg[sizeof(ResendRequest)]);

	char *listTickets = (char *)(&listObjects[resendReply->numberObjects]);
	
//	DEBUGRESTART(printf("[%d] _resendReply from %d \n",CmiMyPe(),resendReply->PE));
	DEBUG_TEAM(printf("[%d] _resendReply from %d \n",CmiMyPe(),resendReply->PE));
	for(int i =0; i< resendReply->numberObjects;i++){	
		Chare *obj = (Chare *)listObjects[i].getObject();
		
		int vecsize;
		memcpy(&vecsize,listTickets,sizeof(int));
		listTickets = &listTickets[sizeof(int)];
		MCount *listTNs = (MCount *)listTickets;	
		listTickets = &listTickets[vecsize*sizeof(MCount)];
		
		if(obj != NULL){
			//the object was restarted on the processor on which it existed
			processReceivedTN(obj,vecsize,listTNs);
		}else{
		//pack up objID vecsize and listTNs and send it to the correct processor
			int totalSize = sizeof(ReceivedTNData)+vecsize*sizeof(MCount);
			char *TNMsg = (char *)CmiAlloc(totalSize);
			ReceivedTNData *receivedTNData = (ReceivedTNData *)TNMsg;
			receivedTNData->recver = listObjects[i];
			receivedTNData->numTNs = vecsize;
			char *tnList = &TNMsg[sizeof(ReceivedTNData)];
			memcpy(tnList,listTNs,sizeof(MCount)*vecsize);

			CmiSetHandler(TNMsg,_receivedTNDataHandlerIdx);
			CmiSyncSendAndFree(listObjects[i].guessPE(),totalSize,TNMsg);
		}	
	}
};

void _receivedTNDataHandler(ReceivedTNData *msg){
	char objName[100];
	Chare *obj = (Chare *) msg->recver.getObject();
	if(obj){		
		char *_msg = (char *)msg;
		DEBUGRESTART(printf("[%d] receivedTNDataHandler for %s\n",CmiMyPe(),obj->mlogData->objID.toString(objName)));
		MCount *listTNs = (MCount *)(&_msg[sizeof(ReceivedTNData)]);
		processReceivedTN(obj,msg->numTNs,listTNs);
	}else{
		int totalSize = sizeof(ReceivedTNData)+sizeof(MCount)*msg->numTNs;
		CmiSyncSendAndFree(msg->recver.guessPE(),totalSize,(char *)msg);
	}
};

/**
 * @brief Processes the received list of tickets from a particular PE.
 */
void processReceivedTN(Chare *obj, int listSize, MCount *listTNs){
	// increases the number of resendReply received
	obj->mlogData->resendReplyRecvd++;

	DEBUG(char objName[100]);
	DEBUG(CkPrintf("[%d] processReceivedTN obj->mlogData->resendReplyRecvd=%d CkNumPes()=%d\n",CkMyPe(),obj->mlogData->resendReplyRecvd,CkNumPes()));
	//CkPrintf("[%d] processReceivedTN with %d listSize by %s\n",CkMyPe(),listSize,obj->mlogData->objID.toString(objName));
	//if(obj->mlogData->receivedTNs == NULL)
	//	CkPrintf("NULL\n");	
	//CkPrintf("using %d entries\n",obj->mlogData->receivedTNs->length());	

	// includes the tickets into the receivedTN structure
	for(int j=0;j<listSize;j++){
		obj->mlogData->receivedTNs->push_back(listTNs[j]);
	}
	
	//if this object has received all the replies find the ticket numbers
	//that senders know about. Those less than the ticket number processed 
	//by the receiver can be thrown away. The rest need not be consecutive
	// ie there can be holes in the list of ticket numbers seen by senders
	if(obj->mlogData->resendReplyRecvd == CkNumPes()){
		obj->mlogData->resendReplyRecvd = 0;
		//sort the received TNS
		sortVec(obj->mlogData->receivedTNs);
	
		//after all the received tickets are in we need to sort them and then 
		// calculate the holes	
		if(obj->mlogData->receivedTNs->size() > 0){
			int tProcessedIndex = searchVec(obj->mlogData->receivedTNs,obj->mlogData->tProcessed);
			int vecsize = obj->mlogData->receivedTNs->size();
			int numberHoles = ((*obj->mlogData->receivedTNs)[vecsize-1] - obj->mlogData->tProcessed)-(vecsize -1 - tProcessedIndex);
			
			// updating tCount with the highest ticket handed out
			if(teamSize > 1){
				if(obj->mlogData->tCount < (*obj->mlogData->receivedTNs)[vecsize-1])
					obj->mlogData->tCount = (*obj->mlogData->receivedTNs)[vecsize-1];
			}else{
				obj->mlogData->tCount = (*obj->mlogData->receivedTNs)[vecsize-1];
			}
			
			if(numberHoles == 0){
			}else{
				char objName[100];					
				printf("[%d] Holes detected in the TNs for %s number %d \n",CkMyPe(),obj->mlogData->objID.toString(objName),numberHoles);
				obj->mlogData->numberHoles = numberHoles;
				obj->mlogData->ticketHoles = new MCount[numberHoles];
				int countHoles=0;
				for(int k=tProcessedIndex+1;k<vecsize;k++){
					if((*obj->mlogData->receivedTNs)[k] != (*obj->mlogData->receivedTNs)[k-1]+1){
						//the TNs are not consecutive at this point
						for(MCount newTN=(*obj->mlogData->receivedTNs)[k-1]+1;newTN<(*obj->mlogData->receivedTNs)[k];newTN++){
							DEBUG(CKPrintf("hole no %d at %d next available ticket %d \n",countHoles,newTN,(*obj->mlogData->receivedTNs)[k]));
							obj->mlogData->ticketHoles[countHoles] = newTN;
							countHoles++;
						}	
					}
				}
				//Holes have been given new TN
				if(countHoles != numberHoles){
					char str[100];
					printf("[%d] Obj %s countHoles %d numberHoles %d\n",CmiMyPe(),obj->mlogData->objID.toString(str),countHoles,numberHoles);
				}
				CkAssert(countHoles == numberHoles);					
				obj->mlogData->currentHoles = numberHoles;
			}
		}
	
		// cleaning up structures and getting ready to continue execution	
		delete obj->mlogData->receivedTNs;
		DEBUG(CkPrintf("[%d] Resetting receivedTNs\n",CkMyPe()));
		obj->mlogData->receivedTNs = NULL;
		obj->mlogData->restartFlag = 0;

		DEBUGRESTART(char objString[100]);
		DEBUGRESTART(CkPrintf("[%d] Can restart handing out tickets again at %.6lf for %s\n",CkMyPe(),CmiWallTimer(),obj->mlogData->objID.toString(objString)));
	}

}


void sortVec(CkVec<MCount> *TNvec){
	//sort it ->its bloddy bubble sort
	//TODO: use quicksort
	for(int i=0;i<TNvec->size();i++){
		for(int j=i+1;j<TNvec->size();j++){
			if((*TNvec)[j] < (*TNvec)[i]){
				MCount temp;
				temp = (*TNvec)[i];
				(*TNvec)[i] = (*TNvec)[j];
				(*TNvec)[j] = temp;
			}
		}
	}
	//make it unique .. since its sorted all equal units will be consecutive
	MCount *tempArray = new MCount[TNvec->size()];
	int	uniqueCount=-1;
	for(int i=0;i<TNvec->size();i++){
		tempArray[i] = 0;
		if(uniqueCount == -1 || tempArray[uniqueCount] != (*TNvec)[i]){
			uniqueCount++;
			tempArray[uniqueCount] = (*TNvec)[i];
		}
	}
	uniqueCount++;
	TNvec->removeAll();
	for(int i=0;i<uniqueCount;i++){
		TNvec->push_back(tempArray[i]);
	}
	delete [] tempArray;
}	

int searchVec(CkVec<MCount> *TNVec,MCount searchTN){
	if(TNVec->size() == 0){
		return -1; //not found in an empty vec
	}
	//binary search to find 
	int left=0;
	int right = TNVec->size();
	int mid = (left +right)/2;
	while(searchTN != (*TNVec)[mid] && left < right){
		if((*TNVec)[mid] > searchTN){
			right = mid-1;
		}else{
			left = mid+1;
		}
		mid = (left + right)/2;
	}
	if(left < right){
		//mid is the element to be returned
		return mid;
	}else{
		if(mid < TNVec->size() && mid >=0){
			if((*TNVec)[mid] == searchTN){
				return mid;
			}else{
				return -1;
			}
		}else{
			return -1;
		}
	}
};


/*
	Method to do parallel restart. Distribute some of the array elements to other processors.
	The problem is that we cant use to charm entry methods to do migration as it will get
	stuck in the protocol that is going to restart
*/

class ElementDistributor: public CkLocIterator{
	CkLocMgr *locMgr;
	int *targetPE;
	void pupLocation(CkLocation &loc,PUP::er &p){
		CkArrayIndex idx=loc.getIndex();
		CkGroupID gID = locMgr->ckGetGroupID();
		p|gID;	    // store loc mgr's GID as well for easier restore
		p|idx;
		p|loc;
	};
	public:
		ElementDistributor(CkLocMgr *mgr_,int *toPE_):locMgr(mgr_),targetPE(toPE_){};
		void addLocation(CkLocation &loc){
			if(*targetPE == CkMyPe()){
				*targetPE = (*targetPE +1)%CkNumPes();				
				return;
			}
			
			CkArrayIndex idx=loc.getIndex();
			CkLocRec_local *rec = loc.getLocalRecord();
			
			CkPrintf("[%d] Distributing objects to Processor %d: ",CkMyPe(),*targetPE);
			idx.print();
			

			//TODO: an element that is being moved should leave some trace behind so that
			// the arraybroadcaster can forward messages to it
			
			//pack up this location and send it across
			PUP::sizer psizer;
			pupLocation(loc,psizer);
			int totalSize = psizer.size()+CmiMsgHeaderSizeBytes;
			char *msg = (char *)CmiAlloc(totalSize);
			char *buf = &msg[CmiMsgHeaderSizeBytes];
			PUP::toMem pmem(buf);
			pmem.becomeDeleting();
			pupLocation(loc,pmem);
			
			locMgr->setDuringMigration(CmiTrue);			
			delete rec;
			locMgr->setDuringMigration(CmiFalse);			
			locMgr->inform(idx,*targetPE);

			CmiSetHandler(msg,_distributedLocationHandlerIdx);
			CmiSyncSendAndFree(*targetPE,totalSize,msg);

			CmiAssert(locMgr->lastKnown(idx) == *targetPE);
			//decide on the target processor for the next object
			*targetPE = (*targetPE +1)%CkNumPes();
		}
		
};

void distributeRestartedObjects(){
	int numGroups = CkpvAccess(_groupIDTable)->size();	
	int i;
	int targetPE=CkMyPe();
	CKLOCMGR_LOOP(ElementDistributor distributor(mgr,&targetPE);mgr->iterate(distributor););
};

void _distributedLocationHandler(char *receivedMsg){
	printf("Array element received at processor %d after distribution at restart\n",CkMyPe());
	char *buf = &receivedMsg[CmiMsgHeaderSizeBytes];
	PUP::fromMem pmem(buf);
	CkGroupID gID;
	CkArrayIndex idx;
	pmem |gID;
	pmem |idx;
	CkLocMgr *mgr = (CkLocMgr*)CkpvAccess(_groupTable)->find(gID).getObj();
	donotCountMigration=1;
	mgr->resume(idx,pmem,CmiTrue);
	donotCountMigration=0;
	informLocationHome(gID,idx,mgr->homePe(idx),CkMyPe());
	printf("Array element inserted at processor %d after distribution at restart ",CkMyPe());
	idx.print();

	CkLocRec *rec = mgr->elementRec(idx);
	CmiAssert(rec->type() == CkLocRec::local);
	
	CkVec<CkMigratable *> eltList;
	mgr->migratableList((CkLocRec_local *)rec,eltList);
	for(int i=0;i<eltList.size();i++){
		if(eltList[i]->mlogData->toResumeOrNot == 1 && eltList[i]->mlogData->resumeCount < globalResumeCount){
			CpvAccess(_currentObj) = eltList[i];
			eltList[i]->ResumeFromSync();
		}
	}
	
	
}


/** this method is used to send messages to a restarted processor to tell
 * it that a particular expected object is not going to get to it */
void sendDummyMigration(int restartPE,CkGroupID lbID,CkGroupID locMgrID,CkArrayIndex &idx,int locationPE){
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
		DEBUGRESTART(CmiPrintf("[%d] dummy Migration received from pe %d for %d:%s \n",CmiMyPe(),msg->locationPE,msg->mgrID.idx,idx2str(msg->idx)));
		LDObjHandle h;
		lb->Migrated(h,1);
	}
	if(msg->flag == MLOG_COUNT){
		DEBUGRESTART(CmiPrintf("[%d] dummyMigration count %d received from restarted processor\n",CmiMyPe(),msg->count));
		msg->count -= verifyAckedRequests;
		for(int i=0;i<msg->count;i++){
			LDObjHandle h;
			lb->Migrated(h,1);
		}
	}
	verifyAckedRequests=0;
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
		CkLocRec_local *local = loc.getLocalRecord();
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
void forAllCharesDo(MlogFn fnPointer,void *data){
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
	CheckpointBarrierMsg msg;
	msg.fromPE = CmiMyPe();
	msg.checkpointCount = checkpointCount;

	CmiSetHandler(&msg,_checkpointBarrierHandlerIdx);
	CmiSyncSend(0,sizeof(CheckpointBarrierMsg),(char *)&msg);
	
};


void sendMlogLocation(int targetPE,envelope *env){
	void *_msg = EnvToUsr(env);
	CkArrayElementMigrateMessage *msg = (CkArrayElementMigrateMessage *)_msg;


	int existing = 0;
	//if this object is already in the retainedobjectlust destined for this
	//processor it should not be sent
	
	for(int i=0;i<retainedObjectList.size();i++){
		MigrationRecord &migRecord = retainedObjectList[i]->migRecord;
		if(migRecord.gID == msg->gid && migRecord.idx == msg->idx){
			DEBUG(CmiPrintf("[%d] gid %d idx %s being sent to %d exists in retainedObjectList with toPE %d\n",CmiMyPe(),msg->gid.idx,idx2str(msg->idx),targetPE,migRecord.toPE));
			existing = 1;
			break;
		}
	}

	if(existing){
		return;
	}
	
	
	countLBToMigrate++;
	
	MigrationNotice migMsg;
	migMsg.migRecord.gID = msg->gid;
	migMsg.migRecord.idx = msg->idx;
	migMsg.migRecord.fromPE = CkMyPe();
	migMsg.migRecord.toPE =  targetPE;
	
	DEBUGLB(printf("[%d] Sending array to proc %d gid %d idx %s\n",CmiMyPe(),targetPE,msg->gid.idx,idx2str(msg->idx)));
	
	RetainedMigratedObject	*retainedObject = new RetainedMigratedObject;
	retainedObject->migRecord = migMsg.migRecord;
	retainedObject->acked  = 0;
	
	CkPackMessage(&env);
	
	migMsg.record = retainedObject;
	retainedObject->msg = env;
	int size = retainedObject->size = env->getTotalsize();
	
	retainedObjectList.push_back(retainedObject);
	
	CmiSetHandler((void *)&migMsg,_receiveMigrationNoticeHandlerIdx);
	CmiSyncSend(getCheckPointPE(),sizeof(migMsg),(char *)&migMsg);
	
	DEBUGLB(printf("[%d] Location in message of size %d being sent to PE %d\n",CkMyPe(),size,targetPE));

}

void _receiveMigrationNoticeHandler(MigrationNotice *msg){
	msg->migRecord.ackFrom = msg->migRecord.ackTo = 0;
	migratedNoticeList.push_back(msg->migRecord);

	MigrationNoticeAck buf;
	buf.record = msg->record;
	CmiSetHandler((void *)&buf,_receiveMigrationNoticeAckHandlerIdx);
	CmiSyncSend(getCheckPointPE(),sizeof(MigrationNoticeAck),(char *)&buf);
}

void _receiveMigrationNoticeAckHandler(MigrationNoticeAck *msg){
	
	RetainedMigratedObject *retainedObject = (RetainedMigratedObject *)(msg->record);
	retainedObject->acked = 1;

	CmiSetHandler(retainedObject->msg,_receiveMlogLocationHandlerIdx);
	CmiSyncSend(retainedObject->migRecord.toPE,retainedObject->size,(char *)retainedObject->msg);

	//inform home about the new location of this object
	CkGroupID gID = retainedObject->migRecord.gID ;
	CkLocMgr *mgr = (CkLocMgr*)CkpvAccess(_groupTable)->find(gID).getObj();
	informLocationHome(gID,retainedObject->migRecord.idx, mgr->homePe(retainedObject->migRecord.idx),retainedObject->migRecord.toPE);
	
	countLBMigratedAway++;
	if(countLBMigratedAway == countLBToMigrate && migrationDoneCalled == 1){
		DEBUGLB(printf("[%d] calling startMlogCheckpoint in _receiveMigrationNoticeAckHandler countLBToMigrate %d countLBMigratedAway %d \n",CmiMyPe(),countLBToMigrate,countLBMigratedAway));
		startMlogCheckpoint(NULL,CmiWallTimer());
	}
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


void resumeFromSyncRestart(void *data,ChareMlogData *mlogData){
/*	if(mlogData->objID.type == TypeArray){
		CkMigratable *elt = (CkMigratable *)mlogData->objID.getObject();
	//	TODO: make sure later that atSync has been called and it needs 
	//	to be resumed from sync
	//
		CpvAccess(_currentObj) = elt;
		elt->ResumeFromSync();
	}*/
}

inline void checkAndSendCheckpointBarrierAcks(CheckpointBarrierMsg *msg){
	if(checkpointBarrierCount == CmiNumPes()){
		CmiSetHandler(msg,_checkpointBarrierAckHandlerIdx);
		for(int i=0;i<CmiNumPes();i++){
			CmiSyncSend(i,sizeof(CheckpointBarrierMsg),(char *)msg);
		}
	}
}

void _checkpointBarrierHandler(CheckpointBarrierMsg *msg){
	DEBUG(CmiPrintf("[%d] msg->checkpointCount %d pe %d checkpointCount %d checkpointBarrierCount %d \n",CmiMyPe(),msg->checkpointCount,msg->fromPE,checkpointCount,checkpointBarrierCount));
	if(msg->checkpointCount == checkpointCount){
		checkpointBarrierCount++;
		checkAndSendCheckpointBarrierAcks(msg);
	}else{
		if(msg->checkpointCount-1 == checkpointCount){
			checkpointBarrierCount++;
			checkAndSendCheckpointBarrierAcks(msg);
		}else{
			printf("[%d] msg->checkpointCount %d checkpointCount %d\n",CmiMyPe(),msg->checkpointCount,checkpointCount);
			CmiAbort("msg->checkpointCount and checkpointCount differ by more than 1");
		}
	}
	CmiFree(msg);
}

void _checkpointBarrierAckHandler(CheckpointBarrierMsg *msg){
	DEBUG(CmiPrintf("[%d] _checkpointBarrierAckHandler \n",CmiMyPe()));
	DEBUGLB(CkPrintf("[%d] Reaching this point\n",CkMyPe()));
	sendRemoveLogRequests();
	(*resumeLbFnPtr)(centralLb);
	CmiFree(msg);
}

/**
	method that informs an array elements home processor of its current location
	It is a converse method to bypass the charm++ message logging framework
*/

void informLocationHome(CkGroupID locMgrID,CkArrayIndex idx,int homePE,int currentPE){
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
		if(mgr->lastKnown(data->idx) == CmiMyPe() && data->locationPE != CmiMyPe() && rec->type() == CkLocRec::local){
			if(data->fromPE == data->locationPE){
				CmiAbort("Another processor has the same object");
			}
		}
	}
	if(rec!= NULL && rec->type() == CkLocRec::local && data->fromPE != CmiMyPe()){
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
	CentralLB *lb = (CentralLB *)CkpvAccess(_groupTable)->find(msg->lbID).getObj();
	msg->step = lb->step();
	CmiAssert(msg->fromPE != CmiMyPe());
	CmiPrintf("[%d] getGlobalStep called from %d step %d gid %d \n",CmiMyPe(),msg->fromPE,lb->step(),msg->lbID.idx);
	CmiSetHandler(msg,_recvGlobalStepHandlerIdx);
	CmiSyncSend(msg->fromPE,sizeof(LBStepMsg),(char *)msg);
};

void _recvGlobalStepHandler(LBStepMsg *msg){
	
	restartDecisionNumber=msg->step;
	RestartRequest *dummyAck = (RestartRequest *)CmiAlloc(sizeof(RestartRequest));
	_updateHomeAckHandler(dummyAck);
};

/**
 * @brief Function to wrap up performance information.
 */
void _messageLoggingExit(){
/*	if(CkMyPe() == 0){
		if(countBuffered != 0){
			printf("[%d] countLocal %d countBuffered %d countPiggy %d Effeciency blocking %.2lf \n",CkMyPe(),countLocal,countBuffered,countPiggy,countLocal/(double )(countBuffered*_maxBufferedMessages));
		}

//		printf("[%d] totalSearchRestoredTime = %.6lf totalSearchRestoredCount %.1lf \n",CkMyPe(),totalSearchRestoredTime,totalSearchRestoredCount);	
	}
	printf("[%d] countHashCollisions %d countHashRefs %d \n",CkMyPe(),countHashCollisions,countHashRefs);*/
	printf("[%d] _messageLoggingExit \n",CmiMyPe());

	//TML: printing some statistics for group approach
	//if(teamSize > 1)
		CkPrintf("[%d] Logged messages = %.0f, log size =  %.2f MB\n",CkMyPe(),MLOGFT_totalMessages,MLOGFT_totalLogSize/(float)MEGABYTE);

}

/**
	The method for returning the actual object pointed to by an id
	If the object doesnot exist on the processor it returns NULL
**/

void* CkObjID::getObject(){
	
		switch(type){
			case TypeChare:	
				return CkLocalChare(&data.chare.id);
			case TypeMainChare:
				return CkLocalChare(&data.chare.id);
			case TypeGroup:
	
				CkAssert(data.group.onPE == CkMyPe());
				return CkLocalBranch(data.group.id);
			case TypeNodeGroup:
				CkAssert(data.group.onPE == CkMyNode());
				//CkLocalNodeBranch(data.group.id);
				{
					CmiImmediateLock(CksvAccess(_nodeGroupTableImmLock));
				  void *retval = CksvAccess(_nodeGroupTable)->find(data.group.id).getObj();
				  CmiImmediateUnlock(CksvAccess(_nodeGroupTableImmLock));					
	
					return retval;
				}	
			case TypeArray:
				{
	
	
					CkArrayID aid(data.array.id);
	
					if(aid.ckLocalBranch() == NULL){ return NULL;}
	
					CProxyElement_ArrayBase aProxy(aid,data.array.idx);
	
					return aProxy.ckLocal();
				}
			default:
				CkAssert(0);
		}
}


int CkObjID::guessPE(){
		switch(type){
			case TypeChare:
			case TypeMainChare:
				return data.chare.id.onPE;
			case TypeGroup:
			case TypeNodeGroup:
				return data.group.onPE;
			case TypeArray:
				{
					CkArrayID aid(data.array.id);
					if(aid.ckLocalBranch() == NULL){
						return -1;
					}
					return aid.ckLocalBranch()->lastKnown(data.array.idx);
				}
			default:
				CkAssert(0);
		}
};

char *CkObjID::toString(char *buf) const {
	
	switch(type){
		case TypeChare:
			sprintf(buf,"Chare %p PE %d \0",data.chare.id.objPtr,data.chare.id.onPE);
			break;
		case TypeMainChare:
			sprintf(buf,"Chare %p PE %d \0",data.chare.id.objPtr,data.chare.id.onPE);	
			break;
		case TypeGroup:
			sprintf(buf,"Group %d	PE %d \0",data.group.id.idx,data.group.onPE);
			break;
		case TypeNodeGroup:
			sprintf(buf,"NodeGroup %d	Node %d \0",data.group.id.idx,data.group.onPE);
			break;
		case TypeArray:
			{
				const CkArrayIndex &idx = data.array.idx;
				const int *indexData = idx.data();
				sprintf(buf,"Array |%d %d %d| id %d \0",indexData[0],indexData[1],indexData[2],data.array.id.idx);
				break;
			}
		default:
			CkAssert(0);
	}
	
	return buf;
};

void CkObjID::updatePosition(int PE){
	if(guessPE() == PE){
		return;
	}
	switch(type){
		case TypeArray:
			{
					CkArrayID aid(data.array.id);
					if(aid.ckLocalBranch() == NULL){
						
					}else{
						char str[100];
						CkLocMgr *mgr = aid.ckLocalBranch()->getLocMgr();
//						CmiPrintf("[%d] location for object %s is %d\n",CmiMyPe(),toString(str),PE);
						CkLocRec *rec = mgr->elementNrec(data.array.idx);
						if(rec != NULL){
							if(rec->type() == CkLocRec::local){
								CmiPrintf("[%d] local object %s can not exist on another processor %d\n",CmiMyPe(),str,PE);
								return;
							}
						}
						mgr->inform(data.array.idx,PE);
					}	
				}

			break;
		case TypeChare:
		case TypeMainChare:
			CkAssert(data.chare.id.onPE == PE);
			break;
		case TypeGroup:
		case TypeNodeGroup:
			CkAssert(data.group.onPE == PE);
			break;
		default:
			CkAssert(0);
	}
}



void MlogEntry::pup(PUP::er &p){
	p | destPE;
	p | _infoIdx;
	p | unackedLocal;
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
	
		if(p.isUnpacking()){
			env->localMlogEntry = NULL;
		}
	}
};

void RestoredLocalMap::pup(PUP::er &p){
	p | minSN;
	p | maxSN;
	p | count;
	if(p.isUnpacking()){
		TNArray = new MCount[count];
	}
	p(TNArray,count);
};




/**********************************
	* The methods of the message logging
	* data structure stored in each chare
	********************************/

MCount ChareMlogData::nextSN(const CkObjID &recver){
/*	MCount SN = snTable.get(recver);
	snTable.put(recver) = SN+1;
	return SN+1;*/
	double _startTime = CmiWallTimer();
	MCount *SN = snTable.getPointer(recver);
	if(SN==NULL){
		snTable.put(recver) = 1;
		return 1;
	}else{
		(*SN)++;
		return *SN;
	}
//	traceUserBracketEvent(34,_startTime,CkWallTimer());
};


MCount ChareMlogData::newTN(){
	MCount TN;
	if(currentHoles > 0){
		int holeidx = numberHoles-currentHoles;
		TN = ticketHoles[holeidx];
		currentHoles--;
		if(currentHoles == 0){
			delete []ticketHoles;
			numberHoles = 0;
		}
	}else{
		TN = ++tCount;
	}	
	return TN;
};

/**
 * Inserts a ticket in the ticketTable if it is not already there.
 */
inline void ChareMlogData::verifyTicket(CkObjID &sender, MCount SN, MCount TN){
	Ticket ticket;

	SNToTicket *ticketRow = ticketTable.get(sender);
	if(ticketRow != NULL){
		Ticket earlierTicket = ticketRow->get(SN);
		if(earlierTicket.TN != 0){
			CkAssert(earlierTicket.TN == TN);
			return;
		}
	}else{
		ticketRow = new SNToTicket();
		ticketTable.put(sender) = ticketRow;
	}
	ticket.TN = TN;
	ticketRow->put(SN) = ticket;
}

/**
 * Generates the next ticket for a request.
 */
inline Ticket ChareMlogData::next_ticket(CkObjID &sender,MCount SN){
	DEBUG(char senderName[100];)
	DEBUG(char recverName[100];)
	double _startTime =CmiWallTimer();
	Ticket ticket;

	// if a ticket is requested during restart, 0 is returned to make the requester to ask for it later.
	if(restartFlag){
		ticket.TN = 0;
		return ticket;
	}
/*	SNToTicket &ticketRow = ticketTable.put(sender);
	Ticket earlierTicket = ticketRow.get(SN);
	if(earlierTicket.TN == 0){
		//This SN has not been ever alloted a ticket
		ticket.TN = newTN();
		ticketRow.put(SN)=ticket;
	}else{
		ticket.TN = earlierTicket.TN;
	}*/
	

	SNToTicket *ticketRow = ticketTable.get(sender);
	if(ticketRow != NULL){
		Ticket earlierTicket = ticketRow->get(SN);
		if(earlierTicket.TN == 0){
			ticket.TN = newTN();
			ticketRow->put(SN) = ticket;
			DEBUG(CkAssert((ticketRow->get(SN)).TN == ticket.TN));
		}else{
			ticket.TN = earlierTicket.TN;
			if(ticket.TN > tCount){
				DEBUG(CmiPrintf("[%d] next_ticket old row ticket sender %s recver %s SN %d TN %d tCount %d\n",CkMyPe(),sender.toString(senderName),objID.toString(recverName),SN,ticket.TN,tCount));
			}
				CmiAssert(ticket.TN <= tCount);
		}
		DEBUG(CmiPrintf("[%d] next_ticket old row ticket sender %s recver %s SN %d TN %d tCount %d\n",CkMyPe(),sender.toString(senderName),objID.toString(recverName),SN,ticket.TN,tCount));
	}else{
		SNToTicket *newRow = new SNToTicket;		
		ticket.TN = newTN();
		newRow->put(SN) = ticket;
		ticketTable.put(sender) = newRow;
		DEBUG(printf("[%d] next_ticket new row ticket sender %s recver %s SN %d TN %d\n",CkMyPe(),sender.toString(senderName),objID.toString(recverName),SN,ticket.TN));
	}
/*TODO: check if the message for this SN has already been received
	in the table of received SNs 
	If it was received before the last checkpoint mark it as old
	other wise received
	*/
	ticket.state = NEW_TICKET;
//	traceUserBracketEvent(34,_startTime,CkWallTimer());
	return ticket;	
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

double totalSearchRestoredTime=0;
double totalSearchRestoredCount=0;

/**
 * Searches the restoredlocal map to see if the combination of sender and sequence number
 * shows up in the map. Returns the ticket if found, or 0 otherwise.
 */
MCount ChareMlogData::searchRestoredLocalQ(CkObjID &sender,CkObjID &recver,MCount SN){
	double start= CkWallTimer();
	MCount TN=0;	
	if(mapTable.numObjects() > 0){
		RestoredLocalMap *map = mapTable.get(sender);
		if(map){
			int index = SN - map->minSN;
			if(index < map->count){
				TN = map->TNArray[index];
			}
		}
	}
	
	DEBUG(char senderName[100]);
	DEBUG(char recverName[100]);
	DEBUG(if(TN != 0){ CmiPrintf("[%d] searchRestoredLocalQ found match sender %s recver %s SN %d TN %d\n",CmiMyPe(),sender.toString(senderName),recver.toString(recverName),SN,TN);});

	totalSearchRestoredTime += CkWallTimer()-start;
	totalSearchRestoredCount++;
	return TN;
}

void ChareMlogData::addToRestoredLocalQ(LocalMessageLog *logEntry){
	restoredLocalMsgLog.push_back(*logEntry);
}

void sortRestoredLocalMsgLog(void *_dummy,ChareMlogData *mlogData){
	mlogData->sortRestoredLocalMsgLog();
}

void ChareMlogData::sortRestoredLocalMsgLog(){
	//sort it ->its bloddy bubble sort
	
	for(int i=0;i<restoredLocalMsgLog.size();i++){
		LocalMessageLog &logEntry = restoredLocalMsgLog[i];
		RestoredLocalMap *map = mapTable.get(logEntry.sender);
		if(map == NULL){
			map = new RestoredLocalMap;
			mapTable.put(logEntry.sender)=map;
		}
		map->count++;
		if(map->minSN == 0){
			map->minSN = logEntry.SN;
		}else{
			if(logEntry.SN < map->minSN){
				map->minSN = logEntry.SN;
			}
		}
		if(logEntry.SN > map->maxSN){
			map->maxSN = logEntry.SN;
		}

	}
	for(int i=0;i< restoredLocalMsgLog.size();i++){
		LocalMessageLog &logEntry = restoredLocalMsgLog[i];
		RestoredLocalMap *map = mapTable.get(logEntry.sender);
		CkAssert(map != NULL);
		if(map->TNArray == NULL){
			map->TNArray = new MCount[map->maxSN-map->minSN+1];			
			CkAssert(map->count == map->maxSN-map->minSN+1);
			map->count = 0;
		}
		map->TNArray[map->count] = logEntry.TN;
		map->count++;
	}
	restoredLocalMsgLog.free();
}

/**
 * Pup method for the metadata.
 * We are preventing the whole message log to be stored (as proposed by Sayantan for dealing with multiple failures).
 * Then, we only support one failure at a time. Read Sayantan's thesis, sections 4.2 and 4.3 for more details.
 */
void ChareMlogData::pup(PUP::er &p){
	int tCountAux;
	int startSize=0;
	char nameStr[100];
	if(p.isSizing()){
		PUP::sizer *sizep = (PUP::sizer *)&p;
		startSize = sizep->size();
	}
	double _startTime = CkWallTimer();
	
	p | objID;
	if(teamRecoveryFlag)
		p | tCountAux;
	else
		p | tCount;
	p | tProcessed;
	if(p.isUnpacking()){
		DEBUG(CmiPrintf("[%d] Obj %s being unpacked with tCount %d tProcessed %d \n",CmiMyPe(),objID.toString(nameStr),tCount,tProcessed));
	}
	p | toResumeOrNot;
	p | resumeCount;
	DEBUG(CmiPrintf("[%d] Obj %s toResumeOrNot %d resumeCount %d \n",CmiMyPe(),objID.toString(nameStr),toResumeOrNot,resumeCount));
	

	/*pack the receivedTN vector*/
	int lengthReceivedTNs;
	if(!p.isUnpacking()){
		if(receivedTNs == NULL){
			lengthReceivedTNs = -1;
		}else{
			lengthReceivedTNs = receivedTNs->size();		
		}
	}
	p | lengthReceivedTNs;
	if(p.isUnpacking()){
		if(lengthReceivedTNs == -1){
			receivedTNs = NULL;
		}else{
			receivedTNs = new CkVec<MCount>;
			for(int i=0;i<lengthReceivedTNs;i++){
				MCount tempTicket;
				p | tempTicket;
				CkAssert(tempTicket > 0);
				receivedTNs->push_back(tempTicket);
			}
		}
	}else{
		for(int i=0;i<lengthReceivedTNs;i++){
			p | (*receivedTNs)[i];
		}
	}
	
	
	p | currentHoles;
	p | numberHoles;
	if(p.isUnpacking()){
		if(numberHoles > 0){
			ticketHoles = new MCount[numberHoles];			
		}else{
			ticketHoles = NULL;
		}
	}
	if(numberHoles > 0){
		p(ticketHoles,numberHoles);
	}
	
	snTable.pup(p);

	// pupping only the unacked local messages in the message log
	int length = 0;
	MlogEntry *entry;
	if(!p.isUnpacking()){
		for(int i=0; i<mlog.length(); i++){
			entry = mlog[i];
			if(entry->unackedLocal)
				length++;
		}
	}
	p | length;
	if(p.isUnpacking()){
		for(int i=0; i<length; i++){
			entry = new MlogEntry();
			mlog.enq(entry);
			entry->pup(p);
		}
	}else{
		for(int i=0; i<mlog.length(); i++){
			entry = mlog[i];
			if(entry->unackedLocal){
				entry->pup(p);
			}
		}
	}

/*	int length;
	if(!p.isUnpacking()){		
		length = mlog.length();	
		if(length > 0)
			DEBUG(printf("[%d] Mlog length %d \n",CkMyPe(),length));
	}
	p | length;
	for(int i=0;i<length;i++){
		MlogEntry *entry;
		if(p.isUnpacking()){
			entry = new MlogEntry();
			mlog.enq(entry);
		}else{
			entry = mlog[i];
		}
		entry->pup(p);
	}*/
	
	p | restoredLocalMsgLog;
	p | resendReplyRecvd;
	p | restartFlag;

	// pup the mapTable
	int tableSize;
	if(!p.isUnpacking()){
		tableSize = mapTable.numObjects();
	}
	p | tableSize;
	if(!p.isUnpacking()){
		CkHashtableIterator *iter = mapTable.iterator();
		while(iter->hasNext()){
			CkObjID *objID;
			RestoredLocalMap **map = (RestoredLocalMap **) iter->next((void **)&objID);
			p | (*objID);
			(*map)->pup(p);
		}
		// releasing memory for iterator
		delete iter;
	}else{
		for(int i=0;i<tableSize;i++){
			CkObjID objID;
			p | objID;
			RestoredLocalMap *map = new RestoredLocalMap;
			map->pup(p);
			mapTable.put(objID) = map;
		}
	}

	//pup the ticketTable
	{
		int ticketTableSize;
		if(!p.isUnpacking()){
			ticketTableSize = ticketTable.numObjects();
		}
		p | ticketTableSize;
		if(!p.isUnpacking()){
			CkHashtableIterator *iter = ticketTable.iterator();
			while(iter->hasNext()){
				CkObjID *objID;
				SNToTicket **ticketRow = (SNToTicket **)iter->next((void **)&objID);
				p | (*objID);
				(*ticketRow)->pup(p);
			}
			//releasing memory for iterator
			delete iter;
		}else{
			for(int i=0;i<ticketTableSize;i++){
				CkObjID objID;
				p | objID;
				SNToTicket *ticketRow = new SNToTicket;
				ticketRow->pup(p);
				if(!teamRecoveryFlag)
					ticketTable.put(objID) = ticketRow;
				else
					delete ticketRow;
			}
		}
	}	
	
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
 */
int getCheckPointPE(){
	//TML: assigning a team-based buddy
	if(teamSize != 1){
		return (CmiMyPe() + teamSize) % CmiNumPes();
	}
	return (CmiNumPes() -1 - CmiMyPe());
}

//assume it is a packed envelope
envelope *copyEnvelope(envelope *env){
	envelope *newEnv = (envelope *)CmiAlloc(env->getTotalsize());
	memcpy(newEnv,env,env->getTotalsize());
	return newEnv;
}

#endif
