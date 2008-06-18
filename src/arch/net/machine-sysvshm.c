/** @file
			size = CmiMsgHeaderGetLength(msg);
 * sysvshm --> sysv shared memory based network layer for communication
 * between processes on the same node
 * This is not going to be the primary mode of communication 
 * but only for messages below a certain size between
 * processes on the same node
 * for non-smp version only
 * * @ingroup NET
 * contains only sysvshm code for 
 * - CmiInitSysvshm()
 * - DeliverViaSysvShm()
 * - CommunicationServerSysvshm()
 * - CmiMachineExitSysvshm()


  created by 
	Eric Shook, eshook2@uiuc.edu , June 10, 2008
*/

/**
 * @addtogroup NET
 * @{
 */


#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#include <sys/shm.h>
#include <sys/sem.h>

#define SYSVSHM_BITS 5 			/* Number of bits to represent number of cpus in a node - currently 32 cpus supported */
#define SYSVSHM_SIZE (1<<SYSVSHM_BITS)	/* Number of bits to represent number of cpus in a node - currently 32 cpus supported */

#define MEMDEBUG(x) /* x */

#define SYSVSHM_STATS 0

#define ACQUIRENW(i) sb.sem_num=i; sb.sem_op=-1; sb.sem_flg=IPC_NOWAIT
#define ACQUIRE(i)   sb.sem_num=i; sb.sem_op=-1; sb.sem_flg=SEM_UNDO
#define RELEASE(i)   sb.sem_num=i; sb.sem_op=1;  sb.sem_flg=SEM_UNDO

#define TESTARRAY 1

#define SNDBITS(i) ((sysvshmContext->nodestart+i)%SYSVSHM_SIZE)
#define RCVBITS(i) ((sysvshmContext->nodestart+sysvshmContext->noderank)%SYSVSHM_SIZE)
#define PIDBITS    Cmi_charmrun_pid%(sizeof(int)-2*SYSVSHM_SIZE)
#define SHMSNDNAME(i)   (PIDBITS<<(SYSVSHM_BITS*2))+(RCVBITS(i)<<SYSVSHM_BITS)+SNDBITS(i)
#define SHMRCVNAME(i)   (PIDBITS<<(SYSVSHM_BITS*2))+(SNDBITS(i)<<SYSVSHM_BITS)+RCVBITS(i)
#define SEMSNDNAME(i)   (PIDBITS<<(SYSVSHM_BITS*2))+SNDBITS(i)
#define SEMRCVNAME(i)   (PIDBITS<<(SYSVSHM_BITS*2))+RCVBITS(i)


/************************
 * 	Implementation currently assumes that
 * 	1) all nodes have the same number of processors
 *  2) in the nodelist all processors in a node are listed in sequence
 *   0 1 2 3      4 5 6 7 
 *   -------      -------
 *    node 1       node 2 
 ************************/

#define SHMBUFLEN 1000000

#define SENDQSTARTSIZE 128


/* This struct is used as the first portion of a shared memory region, followed by data */
typedef struct {
	int count; /* number of messages */
	int bytes; /* number of bytes */

} sharedBufHeader;


typedef struct {
	int semid;
	int shmid;
	sharedBufHeader *header;	
	char *data;
} sharedBufData;

typedef struct {
	int size;  /* total size of data array */
	int begin; /* position of first element */
	int end;   /*	position of next element */
	int numEntries; /* number of entries */

	OutgoingMsg *data;

} SysvshmSendQ;

typedef struct {
	int nodesize;
	int noderank;
	int nodestart,nodeend; /* proc numbers for the start and end of this node */

	ushort *semarray;

        int *sendbufnames;
        int *recvbufnames;

	sharedBufData *recvBufs;
	sharedBufData *sendBufs;

	SysvshmSendQ **sendQs;


#if SYSVSHM_STATS
	int sendCount;
	int validCheckCount;
	int lockRecvCount;
	double validCheckTime;
	double sendTime;
	double commServerTime;
#endif

} SysvshmContext;

SysvshmContext *sysvshmContext=NULL; /* global context */


void calculateNodeSizeAndRank(char **);
void setupSharedBuffers();
void initAllSendQs();

/******************
 * 	Initialization routine
 * 	currently just testing start up
 * ****************/
void CmiInitSysvshm(char **argv){
	MACHSTATE(3,"CminitSysvshm start");
	sysvshmContext = (SysvshmContext *)malloc(sizeof(SysvshmContext));

	if(Cmi_charmrun_pid <= 0){
		CmiAbort("sysvshm must be run with charmrun");
	}
	calculateNodeSizeAndRank(argv);
	if(sysvshmContext->nodesize == 1){
		return;
	}
	MACHSTATE1(3,"CminitSysvshm  %d calculateNodeSizeAndRank",sysvshmContext->nodesize);

	setupSharedBuffers();

	MACHSTATE2(3,"CminitSysvshm %d %d setupSharedBuffers",Cmi_charmrun_pid,sysvshmContext->nodesize);

	initAllSendQs();
	
	MACHSTATE2(3,"CminitSysvshm %d %d initAllSendQs",Cmi_charmrun_pid,sysvshmContext->nodesize);

	MACHSTATE2(3,"CminitSysvshm %d %d done",Cmi_charmrun_pid,sysvshmContext->nodesize);


#if SYSVSHM_STATS
	sysvshmContext->sendCount=0;
	sysvshmContext->sendTime=0.0;
	sysvshmContext->validCheckCount=0;
	sysvshmContext->validCheckTime=0.0;
	sysvshmContext->commServerTime = 0;
	sysvshmContext->lockRecvCount = 0;
#endif

};

/**************
 * shutdown shmem objects and semaphores
 *
 * *******************/
void tearDownSharedBuffers();

void CmiExitSysvshm(){
	int i=0;
	
	if(sysvshmContext->nodesize != 1){
		tearDownSharedBuffers();


		free(sysvshmContext->recvbufnames);
		free(sysvshmContext->sendbufnames);
		free(sysvshmContext->semarray);

		free(sysvshmContext->recvBufs);
		free(sysvshmContext->sendBufs);
	}
#if SYSVSHM_STATS
	CmiPrintf("[%d] sendCount %d sendTime %6lf validCheckCount %d validCheckTime %.6lf commServerTime %6lf lockRecvCount %d \n",
	_Cmi_mynode,sysvshmContext->sendCount,sysvshmContext->sendTime,sysvshmContext->validCheckCount,sysvshmContext->validCheckTime,sysvshmContext->commServerTime,sysvshmContext->lockRecvCount);
#endif
	free(sysvshmContext);
}

/******************
 *Should this message be sent using SysvShm or not ?
 * ***********************/

inline int CmiValidSysvshm(OutgoingMsg ogm, OtherNode node){
#if SYSVSHM_STATS
	sysvshmContext->validCheckCount++;
#endif

	if(ogm->dst >= sysvshmContext->nodestart && ogm->dst <= sysvshmContext->nodeend && ogm->size < SHMBUFLEN ){
		return 1;
	}else{
		return 0;
	}
};


inline int SysvshmRank(int dst){
	return dst - sysvshmContext->nodestart;
}
inline void pushSendQ(SysvshmSendQ *q,OutgoingMsg msg);
inline int sendMessage(OutgoingMsg ogm,sharedBufData *dstBuf,SysvshmSendQ *dstSendQ);
inline int flushSendQ(int dstRank);

/***************
 *
 *Send this message through shared memory
 *if you cannot get lock, put it in the sendQ
 *Before sending messages pick them from sendQ
 *
 * ****************************/

void CmiSendMessageSysvshm(OutgoingMsg ogm,OtherNode node,int rank,unsigned int broot){
	struct sembuf sb;
	
#if SYSVSHM_STATS
	double _startSendTime = CmiWallTimer();
#endif

	
	int dstRank = SysvshmRank(ogm->dst);
	MEMDEBUG(CmiMemoryCheck());
  
	DgramHeaderMake(ogm->data,rank,ogm->src,Cmi_charmrun_pid,1, broot);
	
  
	MACHSTATE4(3,"Send Msg Sysvshm ogm %p size %d dst %d dstRank %d",ogm,ogm->size,ogm->dst,dstRank);

	CmiAssert(dstRank >=0 && dstRank != sysvshmContext->noderank);
	
	sharedBufData *dstBuf = &(sysvshmContext->sendBufs[dstRank]);

	ACQUIRENW(sysvshmContext->noderank);
	if(semop(dstBuf->semid, &sb, 1)<0) {
		/**failed to get the lock 
		insert into q and retain the message*/

		pushSendQ(sysvshmContext->sendQs[dstRank],ogm);
		ogm->refcount++;
		MEMDEBUG(CmiMemoryCheck());
		return;
	}else{
		/***
		 * We got the lock for this buffer
		 * first write all the messages in the sendQ and then write this guy
		 * */
		 if(sysvshmContext->sendQs[dstRank]->numEntries == 0){
				/* send message user event */
				int ret = sendMessage(ogm,dstBuf,sysvshmContext->sendQs[dstRank]);
				MACHSTATE(3,"Sysvshm Send succeeded immediately");
		 }else{
				ogm->refcount+=2;/*this message should not get deleted when the queue is flushed*/
			 	pushSendQ(sysvshmContext->sendQs[dstRank],ogm);
				MACHSTATE3(3,"Sysvshm ogm %p pushed to sendQ length %d refcount %d",ogm,sysvshmContext->sendQs[dstRank]->numEntries,ogm->refcount);
				int sent = flushSendQ(dstRank);
				ogm->refcount--; /*if it has been sent, can be deleted by caller, if not will be deleted when queue is flushed*/
				MACHSTATE1(3,"Sysvshm flushSendQ sent %d messages",sent);
		 }
		 /* unlock the recvbuffer*/
		RELEASE(sysvshmContext->noderank);
		CmiAssert(semop(dstBuf->semid, &sb, 1)>=0);
	}
#if SYSVSHM_STATS
		sysvshmContext->sendCount ++;
		sysvshmContext->sendTime += (CmiWallTimer()-_startSendTime);
#endif
	MEMDEBUG(CmiMemoryCheck());

};

inline void emptyAllRecvBufs();
inline void flushAllSendQs();

/**********
 * Extract all the messages from the recvBuffers you can
 * Flush all sendQs
 * ***/
inline void CommunicationServerSysvshm(){
	
#if SYSVSHM_STATS
	double _startCommServerTime =CmiWallTimer();
#endif	
	
	MEMDEBUG(CmiMemoryCheck());
	emptyAllRecvBufs();
	flushAllSendQs();

#if SYSVSHM_STATS
	sysvshmContext->commServerTime += (CmiWallTimer()-_startCommServerTime);
#endif

	MEMDEBUG(CmiMemoryCheck());
};

static void CmiNotifyStillIdleSysvshm(CmiIdleState *s){
	CommunicationServerSysvshm();
}


static void CmiNotifyBeginIdleSysvshm(CmiIdleState *s)
{
	CmiNotifyStillIdle(s);
}


void calculateNodeSizeAndRank(char **argv){
	sysvshmContext->nodesize=1;
	MACHSTATE(3,"calculateNodeSizeAndRank start");
	CmiGetArgIntDesc(argv, "+nodesize", &(sysvshmContext->nodesize),"Number of cores in this node");
	MACHSTATE1(3,"calculateNodeSizeAndRank argintdesc %d",sysvshmContext->nodesize);

	sysvshmContext->noderank = _Cmi_mynode % (sysvshmContext->nodesize);
	
	MACHSTATE1(3,"calculateNodeSizeAndRank noderank %d",sysvshmContext->noderank);
	
	sysvshmContext->nodestart = _Cmi_mynode -sysvshmContext->noderank;
	
	MACHSTATE(3,"calculateNodeSizeAndRank nodestart ");

	sysvshmContext->nodeend = sysvshmContext->nodestart + sysvshmContext->nodesize -1;

	if(sysvshmContext->nodeend >= _Cmi_numnodes){
		sysvshmContext->nodeend = _Cmi_numnodes-1;
		sysvshmContext->nodesize = (sysvshmContext->nodeend - sysvshmContext->nodestart) +1;
	}
	
	MACHSTATE3(3,"calculateNodeSizeAndRank nodestart %d nodesize %d noderank %d",sysvshmContext->nodestart,sysvshmContext->nodesize,sysvshmContext->noderank);
}

void createShmObjectsAndSems(sharedBufData **bufs, int *bufnames,int issend);
/***************
 * 	calculate the name of the shared objects and semaphores
 * 	
 * 	name scheme
 * 	shared memory: 00..00{rcvbits}{sndbits} as key
 *  	semaphore : 00..00{rcvbits}{sndbits} as key
 * 	open these shared objects and semaphores
 * *********/

void setupSharedBuffers(){
	int i=0;

        CmiAssert((sysvshmContext->recvbufnames=(int *)malloc(sizeof(int)*(sysvshmContext->nodesize)))!=NULL);
        CmiAssert((sysvshmContext->sendbufnames=(int *)malloc(sizeof(int)*(sysvshmContext->nodesize)))!=NULL);
	CmiAssert((sysvshmContext->semarray= (ushort *)malloc(sizeof(int)*(sysvshmContext->nodesize)))!=NULL);

		
	for(i=0;i<sysvshmContext->nodesize;i++){
		sysvshmContext->semarray[i]=1;
		if(i != sysvshmContext->noderank){
			sysvshmContext->sendbufnames[i]=SHMSNDNAME(i);
			sysvshmContext->recvbufnames[i]=SHMRCVNAME(i);
		}
	}
	
	createShmObjectsAndSems(&(sysvshmContext->recvBufs),sysvshmContext->recvbufnames,0);
	createShmObjectsAndSems(&(sysvshmContext->sendBufs),sysvshmContext->sendbufnames,1);

	for(i=0;i<sysvshmContext->nodesize;i++){
		if(i != sysvshmContext->noderank){
			CmiAssert(sysvshmContext->sendBufs[i].header->count == 0);
			sysvshmContext->sendBufs[i].header->count = 0;
			sysvshmContext->sendBufs[i].header->bytes = 0;
		}
	}
}



void createShmObjectsAndSems(sharedBufData **bufs, int *bufnames,int issend) {
	int i;
	int name;

	union semun {
	int val;
	struct semid_ds *buf;
	ushort *array;
	} arg;
	struct semid_ds seminfo;
	arg.array=sysvshmContext->semarray;

	*bufs = (sharedBufData *)malloc(sizeof(sharedBufData)*sysvshmContext->nodesize);
	
	for(i=0;i<sysvshmContext->nodesize;i++){
		if(i != sysvshmContext->noderank){
                        if(issend)
                                name=SEMSNDNAME(i);
                         else
                                name=SEMRCVNAME(i);

			if(((*bufs)[i].semid=semget(name,sysvshmContext->nodesize,0666|IPC_CREAT|IPC_EXCL))>=0) {
				CmiAssert((semctl((*bufs)[i].semid,sysvshmContext->nodesize,SETALL,arg))>=0);
			} else if(errno==EEXIST) {
				CmiAssert((((*bufs)[i].semid)=semget(name,sysvshmContext->nodesize,0666))>=0);
			} else {
				tearDownSharedBuffers();
				CmiPrintf("problem getting sem : %s\n",strerror(errno));
				CmiAbort("sem\n");
			}

			(*bufs)[i].shmid=-1;
			(*bufs)[i].shmid=shmget(bufnames[i],SHMBUFLEN+sizeof(sharedBufHeader),0666|IPC_CREAT|IPC_EXCL); /*Attempt to get shmid*/
			if(errno==EEXIST) 
				(*bufs)[i].shmid=shmget(bufnames[i],SHMBUFLEN+sizeof(sharedBufHeader),0666);
			CmiAssert(((*bufs)[i].shmid)>0);

			CmiAssert(((*bufs)[i].header=shmat((*bufs)[i].shmid, (void *)0, 0))>0);
			(*bufs)[i].data = ((char *)((*bufs)[i].header))+sizeof(sharedBufHeader);
		}else{
			(*bufs)[i].shmid=-1;
			(*bufs)[i].semid=-1;
			(*bufs)[i].header = NULL;
			(*bufs)[i].data = NULL;
		}
	}	
}


void tearDownSharedBuffers(){
	int i,j,ret;
	struct shmid_ds arg;
	for(i= 0;i<sysvshmContext->nodesize;i++){
		if(i != sysvshmContext->noderank){
			/* Shared memory detach */
			shmdt(sysvshmContext->sendBufs[i].header);
			shmdt(sysvshmContext->recvBufs[i].header);

			shmctl(sysvshmContext->sendBufs[i].shmid,IPC_STAT,&arg); /* See if anyone is attached */
			if(arg.shm_nattch==0) { /* No one is attached remove id's */
				shmctl(sysvshmContext->sendBufs[i].shmid,IPC_RMID,NULL);
				shmctl(sysvshmContext->recvBufs[i].shmid,IPC_RMID,NULL);
				semctl(sysvshmContext->sendBufs[i].semid,0, IPC_RMID,0);
				semctl(sysvshmContext->recvBufs[i].semid,0, IPC_RMID,0);
			}
		}
	}
};


void initSendQ(SysvshmSendQ *q,int size);

void initAllSendQs(){
	int i=0;
	sysvshmContext->sendQs = (SysvshmSendQ **) malloc(sizeof(SysvshmSendQ *)*sysvshmContext->nodesize);
	for(i=0;i<sysvshmContext->nodesize;i++){
		if(i != sysvshmContext->noderank){
			(sysvshmContext->sendQs)[i] = (SysvshmSendQ *)malloc(sizeof(SysvshmSendQ));
			initSendQ((sysvshmContext->sendQs)[i],SENDQSTARTSIZE);
		}else{
			(sysvshmContext->sendQs)[i] = NULL;
		}
	}
};


/****************
 *copy this message into the sharedBuf
 If it does not succeed
 *put it into the sendQ 
 *NOTE: This method is called only after obtaining the corresponding mutex
 * ********/
int sendMessage(OutgoingMsg ogm,sharedBufData *dstBuf,SysvshmSendQ *dstSendQ){

	if(dstBuf->header->bytes+ogm->size <= SHMBUFLEN){
		/**copy  this message to sharedBuf **/
		dstBuf->header->count++;
		memcpy(dstBuf->data+dstBuf->header->bytes,ogm->data,ogm->size);
		dstBuf->header->bytes += ogm->size;
		MACHSTATE4(3,"Sysvshm send done ogm %p size %d dstBuf->header->count %d dstBuf->header->bytes %d",ogm,ogm->size,dstBuf->header->count,dstBuf->header->bytes);
		return 1;
	}
	/***
	 * Shared Buffer is too full for this message
	 * **/
	printf("send buffer is too full\n");
	pushSendQ(dstSendQ,ogm);
	ogm->refcount++;
	MACHSTATE3(3,"Sysvshm send ogm %p size %d queued refcount %d",ogm,ogm->size,ogm->refcount);
	return 0;
}

inline OutgoingMsg popSendQ(SysvshmSendQ *q);

/****
 *Try to send all the messages in the sendq to this destination rank
 *NOTE: This method is called only after obtaining the corresponding mutex
 * ************/

inline int flushSendQ(int dstRank){
	sharedBufData *dstBuf = &(sysvshmContext->sendBufs[dstRank]);
	SysvshmSendQ *dstSendQ = sysvshmContext->sendQs[dstRank];
	int count=dstSendQ->numEntries;
	int sent=0;
	while(count > 0){
		OutgoingMsg ogm = popSendQ(dstSendQ);
		ogm->refcount--;
		MACHSTATE4(3,"Sysvshm trysending ogm %p size %d to dstRank %d refcount %d",ogm,ogm->size,dstRank,ogm->refcount);
		int ret = sendMessage(ogm,dstBuf,dstSendQ);
		if(ret==1){
			sent++;
			GarbageCollectMsg(ogm);
		}
		count--;
	}
	return sent;
}

inline void emptyRecvBuf(sharedBufData *recvBuf);

inline void emptyAllRecvBufs(){
	struct sembuf sb;
	int i;
	int j,ret;
	union semun {
	int val;
	struct semid_ds *buf;
	ushort array[1];
	} arg;
	for(i=0;i<sysvshmContext->nodesize;i++){
		if(i != sysvshmContext->noderank){
			sharedBufData *recvBuf = &(sysvshmContext->recvBufs[i]);
			if(recvBuf->header->count > 0){

#if SYSVSHM_STATS
				sysvshmContext->lockRecvCount++;
#endif

				ACQUIRE(i);
				if(semop(recvBuf->semid, &sb, 1)>=0) {
					MACHSTATE1(3,"emptyRecvBuf to be called for rank %d",i);
					emptyRecvBuf(recvBuf);
					RELEASE(i);
					CmiAssert((semop(recvBuf->semid, &sb, 1))>=0);
				}

			}
		}
	}
};

inline void flushAllSendQs(){
	struct sembuf sb;
	int i=0;
	
	for(i=0;i<sysvshmContext->nodesize;i++){
		if(i != sysvshmContext->noderank && sysvshmContext->sendQs[i]->numEntries > 0){
			ACQUIRE(sysvshmContext->noderank);
                        if(semop(sysvshmContext->sendBufs[i].semid, &sb, 1)>=0) {
				MACHSTATE1(3,"flushSendQ %d",i);
				flushSendQ(i);
				RELEASE(sysvshmContext->noderank);
				CmiAssert(semop(sysvshmContext->sendBufs[i].semid, &sb, 1)>=0);
                        }

		}        
	}	
};

void static inline handoverSysvshmMessage(char *newmsg,int total_size,int rank,int broot);

void emptyRecvBuf(sharedBufData *recvBuf){
 	int numMessages = recvBuf->header->count;
	int i=0;

	char *ptr=recvBuf->data;

	for(i=0;i<numMessages;i++){
		int size;
		int rank, srcpe, seqno, magic, i;
		unsigned int broot;
		char *msg = ptr;
		char *newMsg;

		DgramHeaderBreak(msg, rank, srcpe, magic, seqno, broot);
		size = CmiMsgHeaderGetLength(msg);
	
		newMsg = (char *)CmiAlloc(size);
		memcpy(newMsg,msg,size);

		handoverSysvshmMessage(newMsg,size,rank,broot);
		
		ptr += size;

		MACHSTATE3(3,"message of size %d recvd ends at ptr-data %d total bytes %d bytes %d",size,ptr-recvBuf->data,recvBuf->header->bytes);
	}
	CmiAssert(ptr - recvBuf->data == recvBuf->header->bytes);
	recvBuf->header->count=0;
	recvBuf->header->bytes=0;
}


void static inline handoverSysvshmMessage(char *newmsg,int total_size,int rank,int broot){
	CmiAssert(rank == 0);
#if CMK_BROADCAST_SPANNING_TREE
        if (rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || rank == DGRAM_NODEBROADCAST
#endif
         ){
          	SendSpanningChildren(NULL, 0, total_size, newmsg,broot,rank);
					}
#elif CMK_BROADCAST_HYPERCUBE
        if (rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || rank == DGRAM_NODEBROADCAST
#endif
         ){
          		SendHypercube(NULL, 0, total_size, newmsg,broot,rank);
					}
#endif

		switch (rank) {
    	case DGRAM_BROADCAST: {
          CmiPushPE(0, newmsg);
          break;
      }
        default:
				{
					
          CmiPushPE(rank, newmsg);
				}
  	}    /* end of switch */
}


/**************************
 *sendQ helper functions
 * ****************/

void initSendQ(SysvshmSendQ *q,int size){
	q->data = (OutgoingMsg *)malloc(sizeof(OutgoingMsg)*size);

	q->size = size;
	q->numEntries = 0;

	q->begin = 0;
	q->end = 0;
}

void pushSendQ(SysvshmSendQ *q,OutgoingMsg msg){
	if(q->numEntries == q->size){
		/* need to resize */
		OutgoingMsg *oldData = q->data;
		int newSize = q->size<<1;
		q->data = (OutgoingMsg *)malloc(sizeof(OutgoingMsg)*newSize);
		/* copy head to the beginning of the new array */
		
		CmiAssert(q->begin == q->end);

		CmiAssert(q->begin < q->size);
		memcpy(&(q->data[0]),&(oldData[q->begin]),sizeof(OutgoingMsg)*(q->size - q->begin));

		if(q->end != 0){
			memcpy(&(q->data[(q->size - q->begin)]),&(oldData[0]),sizeof(OutgoingMsg)*(q->end));
		}
		free(oldData);
		q->begin = 0;
		q->end = q->size;
		q->size = newSize;
	}
	q->data[q->end] = msg;
	(q->end)++;
	if(q->end >= q->size){
		q->end -= q->size;
	}
	q->numEntries++;
}

OutgoingMsg popSendQ(SysvshmSendQ *q){
	OutgoingMsg ret;
	if(0 == q->numEntries){
		return NULL;
	}

	ret = q->data[q->begin];
	(q->begin)++;
	if(q->begin >= q->size){
		q->begin -= q->size;
	}
	
	q->numEntries--;
	return ret;
}
