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

/**
 * @brief Struct to store the determinant of a particular message.
 * The determinant remembers all the necessary information for a 
 * message to be replayed in the same order as in the execution prior
 * the failure.
 */
typedef struct {
	// sender ID
	CkObjID sender;
	// receiver ID
	CkObjID receiver;
	// SSN: sender sequence number
	MCount SN;
	// TN: ticket number (RSN: receiver sequence number)
	MCount TN;
} Determinant;

/**
 * @brief Typedef for the hashtable type that maps object IDs to determinants.
 */
typedef CkHashtableT<CkHashtableAdaptorT<CkObjID>, CkVec<Determinant> *> CkDeterminantHashtableT;

/**
 * @brief Struct for the header of the removeDeterminants handler
 */
typedef struct {
	char header[CmiMsgHeaderSizeBytes];
	int phase;
	int index;
} RemoveDeterminantsHeader;

/**
 * @brief Struct for the header of the storeDeterminants handler
 */
typedef struct {
	char header[CmiMsgHeaderSizeBytes];
	int number;
	int index;
	int phase;
	int PE;
} StoreDeterminantsHeader;

/**
 * @brief Structure for a ticket assigned to a particular message.
 */
class Ticket {
public:
	MCount TN;
	int state;
	Ticket(){
		TN = 0;
		state = 0;
	}
	Ticket(int x){
		TN = x;
		state = 0;
	}
};
PUPbytes(Ticket)
class MlogEntry;

class RestoredLocalMap;

#define INITSIZE_SNTOTICKET 100

/**
 * @brief Class that maps SN (sequence numbers) to TN (ticket numbers)
 * for a particular object.
 */
class SNToTicket{
	private:
		Ticket initial[INITSIZE_SNTOTICKET];
		Ticket *ticketVec;
		MCount startSN;
		int currentSize;
		MCount finishSN;
	public:
		SNToTicket(){
			currentSize = INITSIZE_SNTOTICKET;
			ticketVec = &initial[0];
			memset(ticketVec,0,sizeof(Ticket)*currentSize);
			startSN = 0;
			finishSN = 0;
		}
		/**
 		 * Gets the finishSN value.
 		 */ 
		inline MCount getFinishSN(){
			return finishSN;
		}
		/**
 		 * Gets the startSN value.
 		 */	 
		inline MCount getStartSN(){
			return startSN;
		}
		//assume indices start from 1.. true for MCounts
		inline Ticket &put(MCount SN){
			if(SN > finishSN) finishSN = SN;
			if(startSN == 0){
				startSN = SN;				
			}
			int index = SN-startSN;
			if(index >= currentSize){
				int oldSize = currentSize;
				Ticket *old = ticketVec;
				
				currentSize = index*2;
				ticketVec = new Ticket[currentSize];
				memcpy(ticketVec,old,sizeof(Ticket)*oldSize);
				if(old != &initial[0]){					
					delete [] old;
				}
			}
			return ticketVec[index];
		}

		inline Ticket get(MCount SN){
			int index = SN-startSN;
			CmiAssert(index >= 0);
			if(index >= currentSize){
				Ticket tn;
				return tn;
			}else{
				return ticketVec[index];
			}
		}

		inline void pup(PUP::er &p){
			p | startSN;
			p | currentSize;
			if(p.isUnpacking()){
				if(currentSize > INITSIZE_SNTOTICKET){
					ticketVec = new Ticket[currentSize];
				}
			}
			for(int i=0;i<currentSize;i++){
				p | ticketVec[i];
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
	// Counts how many tickets have been handed out.
	MCount tCount; 
	// Stores the highest ticket that has been processed.
	MCount tProcessed;
	
	//TODO: pup receivedTNs
	CkVec<MCount> *receivedTNs; //used to store TNs received by senders during a restart
	MCount *ticketHoles;
	int numberHoles;
	int currentHoles;
	// variable that keeps a count of the processors that have replied to a requests to resend messages. 
	int resendReplyRecvd;
	// 0 -> Normal state .. 1-> just after restart. tickets should not be handed out at this time 
	int restartFlag;
	// 0 -> normal state .. 1 -> recovery of a team member 
    int teamRecoveryFlag; 	
	//TML: teamTable, stores the SN to TN mapping for messages intra team
	CkHashtableT<CkHashtableAdaptorT<CkObjID>, SNToTicket *> teamTable;

	int toResumeOrNot;
	int resumeCount;
	int immigrantRecFlag;
	int immigrantSourcePE;

private:

	// SNTable, stores the number of messages sent (sequence numbers) to other objects.
	CkHashtableT<CkHashtableAdaptorT<CkObjID>,MCount> snTable;
	// TNTable, stores the ticket associated with a particular combination <ObjectID,SN>.
	CkHashtableT<CkHashtableAdaptorT<CkObjID>,SNToTicket *> ticketTable;
	// Log of messages sent.
	CkQ<MlogEntry *> mlog;
	
		
	inline MCount newTN();

public:
	/**
 	 * Default constructor.
 	 */ 
	ChareMlogData():ticketTable(1000,0.3),snTable(100,0.4),teamTable(100,0.4){
		tCount = 0;
		tProcessed = 0;
		numberHoles = 0;
		ticketHoles = NULL;
		currentHoles = 0;
		restartFlag=0;
		teamRecoveryFlag=0;
		receivedTNs = NULL;
		resendReplyRecvd=0;
		toResumeOrNot=0;
		resumeCount=0;
		immigrantRecFlag = 0;
	};
	inline MCount nextSN(const CkObjID &recver);
	inline Ticket next_ticket(CkObjID &sender,MCount SN);
	inline void verifyTicket(CkObjID &sender,MCount SN, MCount TN);
	inline Ticket getTicket(CkObjID &sender, MCount SN);
	void addLogEntry(MlogEntry *entry);
	virtual void pup(PUP::er &p);
	CkQ<MlogEntry *> *getMlog(){ return &mlog;};
	MCount searchRestoredLocalQ(CkObjID &sender,CkObjID &recver,MCount SN);
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
	int indexBufDets;
	int numBufDets;
	
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

/**
 *  @brief Class for storing metadata of local messages.
 *  It maps sequence numbers to ticket numbers.
 *  It is used after a restart to maintain the same ticket numbers.
 */
class RestoredLocalMap {
public:
	MCount minSN,maxSN,count;
	MCount *TNArray;
	RestoredLocalMap(){
		minSN=maxSN=count=0;
		TNArray=NULL;
	};
	RestoredLocalMap(int i){
		minSN=maxSN=count=0;
		TNArray=NULL;
	};

	virtual void pup(PUP::er &p);
};


typedef struct {
	char header[CmiMsgHeaderSizeBytes];
	CkObjID sender;
	CkObjID recver;
	MlogEntry *logEntry;
	MCount SN;
	MCount TN;
	int senderPE;
} TicketRequest;
CpvExtern(CkQ<TicketRequest *> *,_delayedTicketRequests);
CpvExtern(CkQ<MlogEntry *> *,_delayedLocalTicketRequests);

typedef struct{
	TicketRequest request;
	Ticket ticket;
	int recverPE;
} TicketReply;

CpvExtern(char**,_bufferedTicketRequests);
extern int _maxBufferedTicketRequests; //Number of ticket requests to be buffered



typedef struct {
	char header[CmiMsgHeaderSizeBytes];
	int numberLogs;
} BufferedLocalLogHeader;

typedef BufferedLocalLogHeader BufferedTicketRequestHeader;

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

typedef struct{
	CkObjID recver;
	MCount tProcessed;
} TProcessedLog;


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
	TProcessedLog *listObjects;
	CkVec<MCount> *ticketVecs;
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
void sendMsg(CkObjID &sender,CkObjID &recver,int destPE,MlogEntry *entry,MCount SN,MCount TN,int resend);
void sendLocalMsg(envelope *env, int _infoIdx);

//handler functions
void _ticketRequestHandler(TicketRequest *);
void _ticketHandler(TicketReply *);
void _pingHandler(CkPingMsg *msg);
void _bufferedLocalMessageCopyHandler(BufferedLocalLogHeader *recvdHeader,int freeHeader=1);
void _bufferedLocalMessageAckHandler(BufferedLocalLogHeader *recvdHeader);
void _bufferedTicketRequestHandler(BufferedTicketRequestHeader *recvdHeader);
void _bufferedTicketHandler(BufferedTicketRequestHeader *recvdHeader);
void _storeDeterminantsHandler(char *buffer);
void _removeDeterminantsHandler(char *buffer);


//methods for sending messages
extern void _skipCldEnqueue(int pe,envelope *env, int infoFn);
extern void _noCldNodeEnqueue(int node, envelope *env);
void generalCldEnqueue(int destPE,envelope *env,int _infoIdx);
void retryTicketRequest(void *_ticketRequest,double curWallTime);

//methods to process received messages with respect to mlog
int preProcessReceivedMessage(envelope *env,Chare **objPointer,MlogEntry **localLogEntry);
void postProcessReceivedMessage(Chare *obj,CkObjID &sender,MCount SN,MlogEntry *entry);


//Checkpoint
CpvExtern(StoredCheckpoint *,_storedCheckpointData);

//methods for checkpointing
void checkpointAlarm(void *_dummy,double curWallTime);
void startMlogCheckpoint(void *_dummy,double curWallTime);
void pupArrayElementsSkip(PUP::er &p, bool create, MigrationRecord *listToSkip,int listSize=0);

//handler functions for checkpoint
void _checkpointRequestHandler(CheckpointRequest *request);
void _storeCheckpointHandler(char *msg);
void _checkpointAckHandler(CheckPointAck *ackMsg);
void _removeProcessedLogHandler(char *requestMsg);
void garbageCollectMlog();

//handler idxs for checkpoint
extern int _checkpointRequestHandlerIdx;
extern int _storeCheckpointHandlerIdx;
extern int _checkpointAckHandlerIdx;
extern int _removeProcessedLogHandlerIdx;

//Restart 


//methods for restart
void CkMlogRestart(const char * dummy, CkArgMsg * dummyMsg);
void CkMlogRestartDouble(void *,double);
void processReceivedTN(Chare *obj,int vecsize,MCount *listTNs);
void processReceivedDet(Chare *obj,int vecsize, Determinant *listDets);
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
int getReverseCheckPointPE();
inline int isSameDet(Determinant *first, Determinant *second);
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
