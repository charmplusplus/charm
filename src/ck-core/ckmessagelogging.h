#ifndef _CKMESSAGELOGGING_H_
#define _CKMESSAGELOGGING_H_

#include "ckobjid.h"

CpvExtern(Chare *,_currentObj);
CpvExtern(int, _numImmigrantRecObjs);

//states of a ticket sent as a reply to a request
#define NEW_TICKET 1
#define OLD_TICKET 2
#define FORWARDED_TICKET 0x8000

//TML: global variable for the size of the team
#define MLOG_RESTARTED 0
#define MLOG_CRASHED 1
#define MEGABYTE 1048576

//array on which we print the formatted string representing an object id
extern char objString[100];

// defines the initial size of _bufferedDets
#define INITIAL_BUFFERED_DETERMINANTS 1024

// constant to define the type of checkpoint used (synchronized or not)
#define SYNCHRONIZED_CHECKPOINT 1

#define DEBUGGING(x) // x
#define DEBUGGING_NOW(x)  x

class MlogEntry;

class RestoredLocalMap;

#define RSSN_INITIAL_SIZE 16

/**
 * @brief Class that stores all received-sender-sequence-numbers (rssn) from another object.
 */
class RSSN{
private:
	MCount *data;
	int currentSize, start, end;
public:

	// Constructor
	RSSN(){
		currentSize = RSSN_INITIAL_SIZE;
		start = 0;
		end = 0;
		data = new MCount[RSSN_INITIAL_SIZE];
		memset(data,0,sizeof(MCount)*currentSize);
	}

        ~RSSN()
        {
          if(data != NULL)
          {
            delete []data;
            data = NULL;
          }
        }

	// Checks if a particular SSN is already in the data; if not, stores it		
	// return value: 0 (sucess, value stored), 1 (value already there)
	int checkAndStore(MCount ssn){
		int index, oldCS, num, i;
		MCount *old;

		// checking if ssn can be inserted, most common case
		if((start == end && ssn == (data[start] + 1)) || data[start] == 0){
			data[start] = ssn;
			return 0;
		}

		// checking if ssn was already received
		if(ssn <= data[start]){
			DEBUGGING(CkPrintf("[%d] Repeated ssn=%d start=%d\n",CkMyPe(),ssn,data[start]));
			return 1;
		}

		// checking if data needs to be extended
		if(ssn-data[start] >= currentSize){
			DEBUGGING(CkPrintf("[%d] Extending Data %d %d %d\n",CkMyPe(),ssn,data[start],currentSize));

			// HACK for migration
			data[0] = ssn;
			start = end = 0;
			return 0;		//HACK

			old = data;
			oldCS = currentSize;
			currentSize *= 2;
			data = new MCount[currentSize];
			memset(data,0,sizeof(MCount)*currentSize);
			for(i=start, num=0; i!=end; i=(i+1)%oldCS,num++){
				data[num] = old[i];
			}
			start = 0;
			end = num-1;
			delete[] old;
		}

		DEBUGGING(CkPrintf("[%d] Ahead ssn=%d start=%d\n",CkMyPe(),ssn,data[start]));

		// adding ssn into data
		num = end - start;
		if(num < 0) num += currentSize;
		num++;
		index = (start+ssn-data[start])%currentSize;
		data[index] = ssn;
		if((ssn-data[start]) >= num) end = index;

		// compressing ssn
		index = start + 1;
		while(data[index]){
			data[start] = 0;
			start = index;
			index = (index + 1)%currentSize;
			if(index == end) break;
		}
		return 0;
	}

	// PUP method
	inline void pup(PUP::er &p){
		p | start;
		p | end;
		p | currentSize;
		if(p.isUnpacking()){
			if(currentSize > RSSN_INITIAL_SIZE){
				delete[] data;
				data = new MCount[currentSize];
			}
		}
		for(int i=0;i<currentSize;i++){
			p | data[i];
		}
	}

};


/**
 * This file includes the definition of the class for storing the meta data
 * associdated with the message logging protocol.
 */


/**
 * @brief This class stores all the message logging related data for a chare.
 */
class ChareMlogData{
public:
	// Object unique ID.
	CkObjID objID;
	// variable that keeps a count of the processors that have replied to a requests to resend messages. 
	int resendReplyRecvd;
	// 0 -> Normal state .. 1-> just after restart. tickets should not be handed out at this time 
	int restartFlag;
	// 0 -> normal state .. 1 -> recovery of a team member 
    int teamRecoveryFlag; 	
	int toResumeOrNot;
	int resumeCount;
	int immigrantRecFlag;
	int immigrantSourcePE;

private:
	// ssnTable, stores the number of messages sent (sequence numbers) to other objects.
	CkHashtableT<CkHashtableAdaptorT<CkObjID>, MCount> ssnTable;
	// receivedSsnTable, stores the list of ssn received from other objects.
	CkHashtableT<CkHashtableAdaptorT<CkObjID>, RSSN *> receivedSsnTable;
	// Log of messages sent.
	CkQ<MlogEntry *> mlog;

public:
	/**
 	 * Default constructor.
 	 */ 
	ChareMlogData():ssnTable(100,0.4),receivedSsnTable(100,0.4){
		restartFlag=0;
		teamRecoveryFlag=0;
		resendReplyRecvd=0;
		toResumeOrNot=0;
		resumeCount=0;
		immigrantRecFlag = 0;
	};
	inline MCount nextSN(const CkObjID &recver);
	int checkAndStoreSsn(const CkObjID &sender, MCount ssn);
	void addLogEntry(MlogEntry *entry);
	virtual void pup(PUP::er &p);
	CkQ<MlogEntry *> *getMlog(){ return &mlog;};
};

/**
 * @brief Entry in a message log. It also includes the index of the buffered
 * determinants array and the number of appended determinants.
 * @note: this message appended numBufDets counting downwards from indexBufDets.
 * In other words, if indexBufDets == 5 and numBufDets = 3, it means that
 * determinants bufDets[2], bufDets[3] and bufDets[4] were piggybacked.
 */
class MlogEntry{
public:
	envelope *env;
	int destPE;
	int _infoIdx;
	
	MlogEntry(envelope *_env,int _destPE,int __infoIdx){
		env = _env;
		destPE = _destPE;
		_infoIdx = __infoIdx;
	}
	MlogEntry(){
		env = 0;
		destPE = -1;
		_infoIdx = 0;
	}
	~MlogEntry(){
		if(env){
			CmiFree(env);
		}
	}
	virtual void pup(PUP::er &p);
};

/**
 * @brief
 */
class StoredCheckpoint{
public:
	char *buf;
	int bufSize;
	int PE;
	StoredCheckpoint(){
		buf = NULL;
		bufSize = 0;
		PE = -1;
	};
};

typedef struct{
	char header[CmiMsgHeaderSizeBytes];
	int PE;
	int dataSize;
} CheckPointDataMsg;

typedef struct{
    char header[CmiMsgHeaderSizeBytes];
    int PE;
} DistributeObjectMsg;


/*typedef struct{
	char header[CmiMsgHeaderSizeBytes];
	int PE;
	int dataSize;
} CheckPointAck;*/

typedef CheckPointDataMsg CheckPointAck;


/**
 * Struct to request a particular action during restart.
 */
typedef struct{
	char header[CmiMsgHeaderSizeBytes];
	int PE;
} RestartRequest;

typedef RestartRequest CkPingMsg;
typedef RestartRequest CheckpointRequest;

typedef struct{
	char header[CmiMsgHeaderSizeBytes];
	int PE;
	double restartWallTime;
	int checkPointSize;
	int numMigratedAwayElements;
	int numMigratedInElements;
	int migratedElementSize;
	int numLocalMessages;	
	CkGroupID lbGroupID;
} RestartProcessorData;

typedef struct{
	char header[CmiMsgHeaderSizeBytes];
	int PE;
	int numberObjects;
} ResendRequest;

typedef ResendRequest RemoveLogRequest;

typedef struct {
	char header[CmiMsgHeaderSizeBytes];
	CkObjID recver;
	int numTNs;
} ReceivedTNData;

// Structure to forward determinants in parallel restart
typedef struct {
	char header[CmiMsgHeaderSizeBytes];
	CkObjID recver;
	int numDets;
} ReceivedDetData;

typedef struct{
	int PE;
	int numberObjects;
	CkObjID *listObjects;
} ResendData;

typedef struct {
	CkGroupID gID;
	CkArrayIndexMax idx;
	int fromPE,toPE;
	char ackFrom,ackTo;
} MigrationRecord;

typedef struct {
	char header[CmiMsgHeaderSizeBytes];
	MigrationRecord migRecord;
	void *record;
} MigrationNotice;

typedef struct {
	char header[CmiMsgHeaderSizeBytes];
	void *record;
} MigrationNoticeAck;

typedef struct {
	MigrationRecord migRecord;
	void *msg;
	int size;
	char acked;
} RetainedMigratedObject;

typedef struct {
	char header[CmiMsgHeaderSizeBytes];
	MigrationRecord migRecord;
	int index;
	int fromPE;
} VerifyAckMsg;

typedef struct {
	char header[CmiMsgHeaderSizeBytes];
	int checkpointCount;
	int fromPE;
} CheckpointBarrierMsg;


//message used to inform a locmgr of an object's current location
typedef struct {
	char header[CmiMsgHeaderSizeBytes];
	CkGroupID mgrID;
	CkArrayIndexMax idx;
	int locationPE;
	int fromPE;
} CurrentLocationMsg;

typedef struct {
	char header[CmiMsgHeaderSizeBytes];
	CkGroupID lbID;
	int fromPE;
	int step;
} LBStepMsg;


#define MLOG_OBJECT 1
#define MLOG_COUNT 2

typedef struct {
	char header[CmiMsgHeaderSizeBytes];
	int flag;// specific object(1) or count(2)
	CkGroupID lbID;
	int count;// if just count
	/**if object **/
	CkGroupID mgrID;
	CkArrayIndexMax idx;
	int locationPE;
} DummyMigrationMsg;


//function pointer passed to the forAllCharesDo method.
//It takes a void *data and a ChareMlogData pointer 
//It gets called for each chare
typedef void (*MlogFn)(void *,ChareMlogData *);

void _messageLoggingInit();

//Methods for sending ticket requests
void sendGroupMsg(envelope *env,int destPE,int _infoIdx);
void sendArrayMsg(envelope *env,int destPE,int _infoIdx);
void sendChareMsg(envelope *env,int destPE,int _infoIdx, const CkChareID *pCid);
void sendNodeGroupMsg(envelope *env,int destNode,int _infoIdx);
void sendCommonMsg(CkObjID &recver,envelope *env,int destPE,int _infoIdx);
void sendRemoteMsg(CkObjID &sender,CkObjID &recver,int destPE,MlogEntry *entry,MCount SN,int resend);
void sendLocalMsg(envelope *env, int _infoIdx);

//handler functions
void _pingHandler(CkPingMsg *msg);

//methods for sending messages
extern void _skipCldEnqueue(int pe,envelope *env, int infoFn);
extern void _noCldNodeEnqueue(int node, envelope *env);
void generalCldEnqueue(int destPE,envelope *env,int _infoIdx);

//methods to process received messages with respect to mlog
int preProcessReceivedMessage(envelope *env,Chare **objPointer,MlogEntry **localLogEntry);
void postProcessReceivedMessage(Chare *obj,CkObjID &sender,MCount SN,MlogEntry *entry);


//Checkpoint
CpvExtern(StoredCheckpoint *,_storedCheckpointData);

//methods for checkpointing
void CkStartMlogCheckpoint(CkCallback &cb);
void checkpointAlarm(void *_dummy,double curWallTime);
void startMlogCheckpoint(void *_dummy,double curWallTime);
void pupArrayElementsSkip(PUP::er &p, bool create, MigrationRecord *listToSkip,int listSize=0);

//handler functions for checkpoint
void _checkpointRequestHandler(CheckpointRequest *request);
void _storeCheckpointHandler(char *msg);
void _checkpointAckHandler(CheckPointAck *ackMsg);
void _removeProcessedLogHandler(char *requestMsg);
void garbageCollectMlog();
void _startCheckpointHandler(CheckpointBarrierMsg *msg);
void _endCheckpointHandler(char *msg);

//handler idxs for checkpoint
extern int _checkpointRequestHandlerIdx;
extern int _storeCheckpointHandlerIdx;
extern int _checkpointAckHandlerIdx;
extern int _removeProcessedLogHandlerIdx;

//Restart 


//methods for restart
void CkMlogRestart(const char * dummy, CkArgMsg * dummyMsg);
void CkMlogRestartDouble(void *,double);
void initializeRestart(void *data,ChareMlogData *mlogData);
void distributeRestartedObjects();
void sendDummyMigration(int restartPE,CkGroupID lbID,CkGroupID locMgrID,CkArrayIndexMax &idx,int locationPE);

//TML: function for locally calling the restart
void CkMlogRestartLocal();

//handler functions for restart
void _getCheckpointHandler(RestartRequest *restartMsg);
void _recvCheckpointHandler(char *_restartData);
void _resendMessagesHandler(char *msg);
void _sendDetsHandler(char *msg);
void _sendDetsReplyHandler(char *msg);
void _receivedTNDataHandler(ReceivedTNData *msg);
void _receivedDetDataHandler(ReceivedDetData *msg);
void _distributedLocationHandler(char *receivedMsg);
void _sendBackLocationHandler(char *receivedMsg);
void _updateHomeRequestHandler(RestartRequest *updateRequest);
void _updateHomeAckHandler(RestartRequest *updateHomeAck);
void _verifyAckRequestHandler(VerifyAckMsg *verifyRequest);
void _verifyAckHandler(VerifyAckMsg *verifyReply);
void _dummyMigrationHandler(DummyMigrationMsg *msg);

//TML: new functions for group-based message logging
void _restartHandler(RestartRequest *restartMsg);
void _getRestartCheckpointHandler(RestartRequest *restartMsg);
void _recvRestartCheckpointHandler(char *_restartData);

//handler idxs for restart
extern int _getCheckpointHandlerIdx;
extern int _recvCheckpointHandlerIdx;
extern int _resendMessagesHandlerIdx;
extern int _sendDetsHandlerIdx;
extern int _sendDetsReplyHandlerIdx;
extern int _receivedTNDataHandlerIdx;
extern int _receivedDetDataHandlerIdx;
extern int _distributedLocationHandlerIdx;
extern int _updateHomeRequestHandlerIdx;
extern int _updateHomeAckHandlerIdx;
extern int _verifyAckRequestHandlerIdx;
extern int _verifyAckHandlerIdx;
extern int _dummyMigrationHandlerIdx;

/// Load Balancing

//methods for load balancing
void startLoadBalancingMlog(void (*fnPtr)(void *),void *_centralLb);
void finishedCheckpointLoadBalancing();
void sendMlogLocation(int targetPE,envelope *env);
void resumeFromSyncRestart(void *data,ChareMlogData *mlogData);
void restoreParallelRecovery(void (*fnPtr)(void *),void *_centralLb);

//handlers for Load Balancing
void _receiveMlogLocationHandler(void *buf);
void _receiveMigrationNoticeHandler(MigrationNotice *msg);
void _receiveMigrationNoticeAckHandler(MigrationNoticeAck *msg);
void _getGlobalStepHandler(LBStepMsg *msg);
void _recvGlobalStepHandler(LBStepMsg *msg);
void _checkpointBarrierHandler(CheckpointBarrierMsg *msg);
void _checkpointBarrierAckHandler(CheckpointBarrierMsg *msg);

//globals used for loadBalancing
extern int onGoingLoadBalancing;
extern void *centralLb;
extern void (*resumeLbFnPtr)(void *) ;
extern int _receiveMlogLocationHandlerIdx;
extern int _receiveMigrationNoticeHandlerIdx;
extern int _receiveMigrationNoticeAckHandlerIdx;
extern int _getGlobalStepHandlerIdx;
extern int _recvGlobalStepHandlerIdx;
extern int _checkpointBarrierHandlerIdx;
extern int _checkpointBarrierAckHandlerIdx;

//extern CkHashtableT<CkHashtableAdaptorT<CkObjID>,void *> migratedObjectList;
extern CkVec<MigrationRecord> migratedNoticeList;
extern CkVec<RetainedMigratedObject *> retainedObjectList;

int getCheckPointPE();
void forAllCharesDo(MlogFn fnPointer,void *data);
envelope *copyEnvelope(envelope *env);
extern void _initDone(void);

//TML: needed for group restart
extern void _resetNodeBocInitVec(void);

//methods for updating location
void informLocationHome(CkGroupID mgrID,CkArrayIndexMax idx,int homePE,int currentPE);

//handlers for updating locations
void _receiveLocationHandler(CurrentLocationMsg *data);

//globals for updating locations
extern int _receiveLocationHandlerIdx;


extern "C" void CmiDeliverRemoteMsgHandlerRange(int lowerHandler,int higherHandler);

#endif
