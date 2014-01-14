/** @file
 * Support for persistent communication setup
 * @ingroup Machine
 */

/**
 * \addtogroup Machine
*/
/*@{*/

#include "converse.h"
#if CMK_PERSISTENT_COMM
//#define EXTERNAL_COMPRESS 1
//#if EXTERNAL_COMPRESS
//#else
#include "compress.c"
#include "compress-external.c"
//#endif
#include "machine-persistent.h"
#define ENVELOP_SIZE 104
//#define VERIFY 1 
CpvDeclare(PersistentSendsTable *, persistentSendsTableHead);
CpvDeclare(PersistentSendsTable *, persistentSendsTableTail);
CpvDeclare(int, persistentSendsTableCount);
CpvDeclare(PersistentReceivesTable *, persistentReceivesTableHead);
CpvDeclare(PersistentReceivesTable *, persistentReceivesTableTail);
CpvDeclare(int, persistentReceivesTableCount);

/* Converse message type */
typedef struct _PersistentRequestMsg {
  char core[CmiMsgHeaderSizeBytes];
  int requestorPE;
  int maxBytes;
  PersistentHandle sourceHandler;
#if DELTA_COMPRESS
  int   compressStart;
  int   dataType;
#endif
} PersistentRequestMsg;

typedef struct _PersistentReqGrantedMsg {
  char core[CmiMsgHeaderSizeBytes];
/*
  void *msgAddr[PERSIST_BUFFERS_NUM];
  void *slotFlagAddress[PERSIST_BUFFERS_NUM];
*/
  PersistentBuf    buf[PERSIST_BUFFERS_NUM];
  PersistentHandle sourceHandler;
  PersistentHandle destHandler;
  PersistentHandle destDataHandler;
} PersistentReqGrantedMsg;

typedef struct _PersistentDestroyMsg {
  char core[CmiMsgHeaderSizeBytes];
  PersistentHandle destHandlerIndex;
} PersistentDestroyMsg;

/* Converse handler */
int persistentRequestHandlerIdx;
int persistentReqGrantedHandlerIdx;
int persistentDestroyHandlerIdx;
int persistentDecompressHandlerIdx;
int persistentNoDecompressHandlerIdx;

CpvDeclare(PersistentHandle *, phs);
CpvDeclare(int, phsSize);
CpvDeclare(int, curphs);

/******************************************************************************
     Utilities
******************************************************************************/

extern void initRecvSlot(PersistentReceivesTable *slot);
extern void initSendSlot(PersistentSendsTable *slot);

void swapSendSlotBuffers(PersistentSendsTable *slot)
{
  if (PERSIST_BUFFERS_NUM == 2) {
#if 0
  void *tmp = slot->destAddress[0];
  slot->destAddress[0] = slot->destAddress[1];
  slot->destAddress[1] = tmp;
  tmp = slot->destSizeAddress[0];
  slot->destSizeAddress[0] = slot->destSizeAddress[1];
  slot->destSizeAddress[1] = tmp;
#else
  PersistentBuf tmp = slot->destBuf[0];
  slot->destBuf[0] = slot->destBuf[1];
  slot->destBuf[1] = tmp;
#endif
  }
}

void swapRecvSlotBuffers(PersistentReceivesTable *slot)
{
  if (PERSIST_BUFFERS_NUM == 2) {
#if 0
  void *tmp = slot->messagePtr[0];
  slot->messagePtr[0] = slot->messagePtr[1];
  slot->messagePtr[1] = tmp;
  tmp = slot->recvSizePtr[0];
  slot->recvSizePtr[0] = slot->recvSizePtr[1];
  slot->recvSizePtr[1] = tmp;
#else
  PersistentBuf tmp = slot->destBuf[0];
  slot->destBuf[0] = slot->destBuf[1];
  slot->destBuf[1] = tmp;
#endif
  }
}

PersistentHandle getFreeSendSlot()
{
  PersistentSendsTable *slot = (PersistentSendsTable *)malloc(sizeof(PersistentSendsTable));
  initSendSlot(slot);
  if (CpvAccess(persistentSendsTableHead) == NULL) {
    CpvAccess(persistentSendsTableHead) = CpvAccess(persistentSendsTableTail) = slot;
  }
  else {
    CpvAccess(persistentSendsTableTail)->next = slot;
    slot->prev = CpvAccess(persistentSendsTableTail);
    CpvAccess(persistentSendsTableTail) = slot;
  }
  CpvAccess(persistentSendsTableCount)++;
  return slot;
}

PersistentHandle getFreeRecvSlot()
{
  PersistentReceivesTable *slot = (PersistentReceivesTable *)malloc(sizeof(PersistentReceivesTable));
  initRecvSlot(slot);
  if (CpvAccess(persistentReceivesTableHead) == NULL) {
    CpvAccess(persistentReceivesTableHead) = CpvAccess(persistentReceivesTableTail) = slot;
  }
  else {
    CpvAccess(persistentReceivesTableTail)->next = slot;
    slot->prev = CpvAccess(persistentReceivesTableTail);
    CpvAccess(persistentReceivesTableTail) = slot;
  }
  CpvAccess(persistentReceivesTableCount)++;
  return slot;
}

/******************************************************************************
     Create Persistent Comm handler
     When creating a persistent comm with destPE and maxSize
     1. allocate a free PersistentSendsTable entry, send a 
        PersistentRequestMsg message to destPE
        buffer persistent message before  Persistent Comm is setup;
     2. destPE execute Converse handler persistentRequestHandler() on the
        PersistentRequestMsg message:
        allocate a free PersistentReceivesTable entry;
        allocate a message buffer of size maxSize for the communication;
        Send back a PersistentReqGrantedMsg with message address, etc for
        rdma_put;
     3. Converse handler persistentReqGrantedHandler() executed on
        sender for the PersistentReqGrantedMsg. setup finish, send buffered
        message.
******************************************************************************/
PersistentHandle CmiCreateCompressPersistent(int destPE, int maxBytes, int compressStart, int type)
{
  PersistentHandle h;
  PersistentSendsTable *slot;

  if (CmiMyNode() == CmiNodeOf(destPE)) return NULL;

  h = getFreeSendSlot();
  slot = (PersistentSendsTable *)h;

  slot->destPE = destPE;
  slot->sizeMax = ALIGN16(maxBytes);
  slot->addrIndex = 0;
  PersistentRequestMsg *msg = (PersistentRequestMsg *)CmiAlloc(sizeof(PersistentRequestMsg));
  msg->maxBytes = maxBytes;
  msg->sourceHandler = h;
  msg->requestorPE = CmiMyPe();
#if DELTA_COMPRESS
  slot->previousMsg  = NULL; 
  slot->compressStart =  msg->compressStart = compressStart;
  slot->dataType = msg->dataType = type;
  slot->compressSize = 0;
  slot->compressFlag = 1;
#endif
  CmiSetHandler(msg, persistentRequestHandlerIdx);
  CmiSyncSendAndFree(destPE,sizeof(PersistentRequestMsg),msg);

  return h;
}


PersistentHandle CmiCreateCompressPersistentSize(int destPE, int maxBytes, int compressStart, int compressSize, int type)
{
  PersistentHandle h;
  PersistentSendsTable *slot;

  if (CmiMyNode() == CmiNodeOf(destPE)) return NULL;

  h = getFreeSendSlot();
  slot = (PersistentSendsTable *)h;

  slot->destPE = destPE;
  slot->sizeMax = ALIGN16(maxBytes);
  slot->addrIndex = 0;
  PersistentRequestMsg *msg = (PersistentRequestMsg *)CmiAlloc(sizeof(PersistentRequestMsg));
  msg->maxBytes = maxBytes;
  msg->sourceHandler = h;
  msg->requestorPE = CmiMyPe();
#if DELTA_COMPRESS
  slot->previousMsg  = NULL; 
  slot->compressStart =  msg->compressStart = compressStart;
  slot->compressSize = compressSize;
  slot->dataType = msg->dataType = type;
  slot->compressFlag = 1;
#endif
  CmiSetHandler(msg, persistentRequestHandlerIdx);
  CmiSyncSendAndFree(destPE,sizeof(PersistentRequestMsg),msg);

  return h;
}

PersistentHandle CmiCreatePersistent(int destPE, int maxBytes)
{
  PersistentHandle h;
  PersistentSendsTable *slot;

  if (CmiMyNode() == CmiNodeOf(destPE)) return NULL;

  h = getFreeSendSlot();
  slot = (PersistentSendsTable *)h;

  slot->destPE = destPE;
  slot->sizeMax = ALIGN16(maxBytes);
  slot->addrIndex = 0;
  PersistentRequestMsg *msg = (PersistentRequestMsg *)CmiAlloc(sizeof(PersistentRequestMsg));
  msg->maxBytes = maxBytes;
  msg->sourceHandler = h;
  msg->requestorPE = CmiMyPe();

#if DELTA_COMPRESS
  slot->compressFlag = 0;
#endif
  CmiSetHandler(msg, persistentRequestHandlerIdx);
  CmiSyncSendAndFree(destPE,sizeof(PersistentRequestMsg),msg);

  return h;
}

#if DELTA_COMPRESS
static void persistentNoDecompressHandler(void *msg)
{
    //no msg is compressed, just update history
    PersistentReceivesTable *slot = (PersistentReceivesTable *) (((CmiMsgHeaderExt*)msg)-> persistRecvHandler);
    int size = ((CmiMsgHeaderExt*)msg)->size;
    slot->addrIndex = (slot->addrIndex + 1)%PERSIST_BUFFERS_NUM;
    // uncompress data from historyIndex data
#if COPY_HISTORY 
    memcpy(slot->history, msg, size);
#endif
    CldRestoreHandler(msg);
    (((CmiMsgHeaderExt*)msg)->xhdl) =  (((CmiMsgHeaderExt*)msg)->xxhdl);
    CmiHandleMessage(msg);
}

static void persistentDecompressHandler(void *msg)
{
    //  recover message based on previousRecvMsg
    PersistentReceivesTable *slot = (PersistentReceivesTable *) (((CmiMsgHeaderExt*)msg)-> persistRecvHandler);
    int     historyIndex;
    register int i;
    char    *cmsg = (char*)msg;
    int     size = ((CmiMsgHeaderExt*)msg)->size;
    int     compressSize = *(int*)(msg+slot->compressStart);
    int     originalSize = *(int*)(msg+slot->compressStart+sizeof(int));
    
    char    *decompressData =(char*) malloc(originalSize);
#if COPY_HISTORY
    char *history = slot->history;
#else
    historyIndex = (slot->addrIndex + 1)%PERSIST_BUFFERS_NUM;
    slot->addrIndex = (slot->addrIndex + 1)%PERSIST_BUFFERS_NUM;
    char *history = (char*)(slot->destBuf[historyIndex].destAddress);
#endif
    //CmiPrintf("[%d] begin uncompress message is decompressed [%d:%d:%d start:%d]\n ", CmiMyPe(), size, compressSize, originalSize, slot->compressStart);
    int  left_size = size - slot->compressStart - originalSize;
    char *base_dst = cmsg+size-1;
    char *base_src = cmsg+ size - originalSize +compressSize+sizeof(int) -1;
    for(i=0; i<left_size; i++)
    {
       *base_dst = *base_src;
       base_dst--;
       base_src--;
    }

    if(slot->dataType == CMI_FLOATING) 
    	decompressFloatingPoint(msg + slot->compressStart+2*sizeof(int), decompressData, originalSize, compressSize, history+slot->compressStart);
    else if(slot->dataType == CMI_CHAR) 
	decompressChar(msg + slot->compressStart+2*sizeof(int), decompressData, originalSize, compressSize, history+slot->compressStart);
    else if(slot->dataType == CMI_ZLIB) 
	decompressZlib(msg + slot->compressStart+2*sizeof(int), decompressData, originalSize, compressSize, history+slot->compressStart);
    else if(slot->dataType == CMI_LZ4) 
	decompressLz4(msg + slot->compressStart+2*sizeof(int), decompressData, originalSize, compressSize, history+slot->compressStart);
    memcpy(msg+slot->compressStart, decompressData, originalSize);
    free(decompressData);
    CldRestoreHandler(msg);
    (((CmiMsgHeaderExt*)msg)->xhdl) =  (((CmiMsgHeaderExt*)msg)->xxhdl);

#if VERIFY
   
    char    real1 = cmsg[size - originalSize +sizeof(int)+compressSize];
    char    real2 = cmsg[size - originalSize +sizeof(int)+compressSize+1];
    char checksum1 = cmsg[0];
    for(i=1; i< slot->compressStart; i++)
        checksum1 ^= cmsg[i];
    if(memcmp(&checksum1, &real1, 1))
        CmiPrintf("receiver chumsum wrong header \n");
    char  checksum2 = cmsg[slot->compressStart];
    for(i=slot->compressStart+1; i< size; i++)
        checksum2 ^= cmsg[i];
    if(memcmp(&checksum2, &real2, 1))
        CmiPrintf("receiver chumsum wrong data \n");

#endif
#if COPY_HISTORY
    memcpy(slot->history, msg, size);
#endif
    CmiHandleMessage(msg);
}

#if 0
int CompressPersistentMsg(PersistentHandle h, int size, void **m)
{
    void *msg = *m;
    PersistentSendsTable *slot = (PersistentSendsTable *)h;
    int  newSize;
    void *history = slot->previousMsg;
    void *dest=NULL;
    int compressSize=size;
    if(history == NULL)
    {
        newSize = size;
        slot->previousMsg = msg;
        CmiReference(msg);
        (((CmiMsgHeaderExt*)msg)->xxhdl) = (((CmiMsgHeaderExt*)msg)->xhdl);
        CldSwitchHandler(msg, persistentNoDecompressHandlerIdx);
    }else
    {
        if(slot->compressSize == 0)
        {
            slot->compressSize = size - slot->compressStart;
        }
        if(slot->compressSize>100)
        {
            dest = CmiAlloc(size);
            compressChar((char*)msg+slot->compressStart, (char*)dest+slot->compressStart+sizeof(int), slot->compressSize, &compressSize, (char*)history+slot->compressStart);
        }
    
        CmiFree(history);
        history = msg;
        CmiReference(msg);
        if(slot->compressSize-compressSize <= 100) //no compress
        {
            newSize = size;
            (((CmiMsgHeaderExt*)msg)->xxhdl) = (((CmiMsgHeaderExt*)msg)->xhdl);
            CldSwitchHandler(msg, persistentNoDecompressHandlerIdx);
            if(dest != NULL)
                CmiFree(dest);
        }else
        {
            //header
            memcpy(dest, msg, slot->compressStart);
            //compressedSize
            *(int*)(dest+slot->compressStart) = compressSize;
            //tail
            int leftSize = size - slot->compressStart - slot->compressSize;
            if(leftSize > 0)
                memcpy((char*)dest+slot->compressStart+sizeof(int)+compressSize, msg+slot->compressStart+slot->compressSize, leftSize);
            newSize = size-slot->compressSize+compressSize+sizeof(int);
            (((CmiMsgHeaderExt*)dest)->xxhdl) = (((CmiMsgHeaderExt*)dest)->xhdl);
            CldSwitchHandler(dest, persistentDecompressHandlerIdx);
            CmiPrintf(" handler =(%d : %d : %d) (%d:%d:%d)  %d\n", (((CmiMsgHeaderExt*)dest)->hdl), (((CmiMsgHeaderExt*)dest)->xhdl), (((CmiMsgHeaderExt*)dest)->xxhdl), (((CmiMsgHeaderExt*)msg)->hdl),  (((CmiMsgHeaderExt*)msg)->xhdl), (((CmiMsgHeaderExt*)msg)->xxhdl), persistentDecompressHandlerIdx);
            *m=dest;
        }
        //update history
    }
    ((CmiMsgHeaderExt*)*m)-> persistRecvHandler = slot->destDataHandle;
    ((CmiMsgHeaderExt*)*m)->size = size;
    return newSize;
}

#else
int CompressPersistentMsg(PersistentHandle h, int size, void *msg)
{
    PersistentSendsTable *slot = (PersistentSendsTable *)h;
    int  newSize;
    void *history = slot->previousMsg;
    void *dest=NULL;
    int  compressSize=size;
    int  i;
    char *cmsg = (char*)msg;

   
    ((CmiMsgHeaderExt*)msg)-> persistRecvHandler = slot->destDataHandle;
    ((CmiMsgHeaderExt*)msg)->size = size;
    
    if(history == NULL)
    {
        newSize = size;
        slot->previousMsg = msg;
        slot->previousSize = size;
        CmiReference(msg);
        (((CmiMsgHeaderExt*)msg)->xxhdl) = (((CmiMsgHeaderExt*)msg)->xhdl);
        CldSwitchHandler(msg, persistentNoDecompressHandlerIdx);
    }else if(size != slot->previousSize)    //persistent msg size changes
    {
        newSize = size;
        CmiFree(slot->previousMsg);
        slot->previousMsg = msg;
        if(slot->compressSize == slot->previousSize - slot->compressStart)
            slot->compressSize = size - slot->compressStart;
        slot->previousSize = size;
        CmiReference(msg);
        (((CmiMsgHeaderExt*)msg)->xxhdl) = (((CmiMsgHeaderExt*)msg)->xhdl);
        CldSwitchHandler(msg, persistentNoDecompressHandlerIdx);
    }
    else {
        
        if(slot->compressSize == 0) {slot->compressSize = size-slot->compressStart; }
#if VERIFY
        char checksum1;
        char checksum2;
        void *history_save = CmiAlloc(size);
        memcpy(history_save, history, size);
        checksum1 = cmsg[0];
        for(i=1; i< slot->compressStart; i++)
            checksum1 ^= cmsg[i];
        checksum2 = cmsg[slot->compressStart];
        for(i=slot->compressStart+1; i< size; i++)
            checksum2 ^= cmsg[i];
#endif
        //dest = malloc(slot->compressSize);
#if EXTERNAL_COMPRESS
        int maxSize = (slot->compressSize+40)>LZ4_compressBound(slot->compressSize) ? slot->compressSize+40 : LZ4_compressBound(slot->compressSize);
#else
        int maxSize = slot->compressSize;
#endif
        dest = malloc(maxSize);
    if(slot->dataType == CMI_FLOATING) 
        compressFloatingPoint(msg+slot->compressStart, dest, slot->compressSize, &compressSize, history+slot->compressStart);
    else if(slot->dataType == CMI_CHAR)
        compressChar(msg+slot->compressStart, dest, slot->compressSize, &compressSize, history+slot->compressStart);
    else if(slot->dataType == CMI_ZLIB)
        compressZlib(msg+slot->compressStart, dest, slot->compressSize, &compressSize, history+slot->compressStart);
    else if(slot->dataType == CMI_LZ4) 
        compressLz4(msg+slot->compressStart, dest, slot->compressSize, &compressSize, history+slot->compressStart);

#if VERIFY
        void *recover = malloc(slot->compressSize);
        decompressChar(dest, recover, slot->compressSize, compressSize, history_save+slot->compressStart);
        if(memcmp(msg+slot->compressStart, recover, slot->compressSize))
            CmiPrintf("sth wrong with compression\n");
#endif
        if(slot->compressSize - compressSize <= 100) //not compress
        {
            newSize = size;
            (((CmiMsgHeaderExt*)msg)->xxhdl) = (((CmiMsgHeaderExt*)msg)->xhdl);
            CldSwitchHandler(msg, persistentNoDecompressHandlerIdx);
            CmiFree(slot->previousMsg);
            slot->previousMsg = msg;
            CmiReference(msg);
        }else
        {
            memcpy(history+slot->compressStart, msg+slot->compressStart, slot->compressSize);
            *(int*)(msg+slot->compressStart) = compressSize;
            *(int*)(msg+slot->compressStart+sizeof(int)) = slot->compressSize;
            memcpy(msg+slot->compressStart+2*sizeof(int), dest, compressSize);
            int leftSize = size-slot->compressStart-slot->compressSize;
            //depending on memcpy implementation, this might not be safe
            if(leftSize > 0)
                memcpy(msg+slot->compressStart+compressSize+2*sizeof(int), msg+slot->compressStart+slot->compressSize, leftSize);
            newSize = slot->compressStart + compressSize + 2*sizeof(int) +leftSize;
            (((CmiMsgHeaderExt*)msg)->xxhdl) = (((CmiMsgHeaderExt*)msg)->xhdl);
            CldSwitchHandler(msg, persistentDecompressHandlerIdx);
#if VERIFY
            memcpy(msg+newSize, &checksum1, 1); 
            memcpy(msg+newSize+1, &checksum2, 1); 
            char *orig = CmiAlloc(size);
            memcpy(orig, msg, newSize);
             
            char    *decompressData =(char*) malloc(slot->compressSize);
            int left_size = size - slot->compressStart - slot->compressSize;
            char *base_dst = orig+size-1;
            char *base_src = orig + size - slot->compressSize +compressSize+2*sizeof(int) -1;
            for(i=0; i<left_size; i++)
            {
                *base_dst = *base_src;
                base_dst--;
                base_src--;
            }
    
            decompressChar(orig+slot->compressStart+2*sizeof(int), decompressData, slot->compressSize, compressSize, history_save+slot->compressStart);
            memcpy(orig+slot->compressStart, decompressData, slot->compressSize);
            free(decompressData);
            CldRestoreHandler(orig);
            (((CmiMsgHeaderExt*)orig)->xhdl) =  (((CmiMsgHeaderExt*)orig)->xxhdl);
            if(memcmp(orig, history, slot->compressStart))
                CmiPrintf("sth wrong header all \n");
            if(memcmp(orig+slot->compressStart, history+slot->compressStart, slot->compressSize))
                CmiPrintf("sth wrong data \n");
            newSize += 2;
#endif
        }
        free(dest);
    }
 
    return newSize;
}

#endif
#else
#endif

/* for SMP */
PersistentHandle CmiCreateNodePersistent(int destNode, int maxBytes)
{
    /* randomly pick one rank on the destination node is fine for setup.
       actual message will be handled by comm thread anyway */
  int pe = CmiNodeFirst(destNode) + rand()/RAND_MAX * CmiMyNodeSize();
  return CmiCreatePersistent(pe, maxBytes);
}
PersistentHandle CmiCreateCompressNodePersistent(int destNode, int maxBytes, int start, int type)
{
    /* randomly pick one rank on the destination node is fine for setup.
       actual message will be handled by comm thread anyway */
  int pe = CmiNodeFirst(destNode) + rand()/RAND_MAX * CmiMyNodeSize();
  return CmiCreateCompressPersistent(pe, maxBytes, start, type);
}

PersistentHandle CmiCreateCompressNodePersistentSize(int destNode, int maxBytes, int start, int compressSize, int type)
{
    /* randomly pick one rank on the destination node is fine for setup.
       actual message will be handled by comm thread anyway */
  int pe = CmiNodeFirst(destNode) + rand()/RAND_MAX * CmiMyNodeSize();
  return CmiCreateCompressPersistentSize(pe, maxBytes, start, compressSize, type);
}


static void persistentRequestHandler(void *env)
{             
  PersistentRequestMsg *msg = (PersistentRequestMsg *)env;
  char *buf;
  int i;

  PersistentHandle h = getFreeRecvSlot();
  PersistentReceivesTable *slot = (PersistentReceivesTable *)h;

  /* build reply message */
  PersistentReqGrantedMsg *gmsg = CmiAlloc(sizeof(PersistentReqGrantedMsg));

#if DELTA_COMPRESS
  slot->compressStart = msg->compressStart;
  slot->dataType = msg->dataType;
#if COPY_HISTORY
  slot->history = malloc(msg->maxBytes);
#endif
#endif
  setupRecvSlot(slot, msg->maxBytes);

  for (i=0; i<PERSIST_BUFFERS_NUM; i++) {
#if 0
    gmsg->msgAddr[i] = slot->messagePtr[i];
    gmsg->slotFlagAddress[i] = slot->recvSizePtr[i];
#else
    gmsg->buf[i] = slot->destBuf[i];
#endif
  }

  gmsg->sourceHandler = msg->sourceHandler;
  gmsg->destHandler = getPersistentHandle(h, 1);
#if  DELTA_COMPRESS
  gmsg->destDataHandler = h;
  //CmiPrintf("[%d] receiver slot=%p, current=%d, h=%p  =%p \n", CmiMyPe(), slot, slot->addrIndex, h, gmsg->destDataHandler);
#endif
  CmiSetHandler(gmsg, persistentReqGrantedHandlerIdx);
  CmiSyncSendAndFree(msg->requestorPE,sizeof(PersistentReqGrantedMsg),gmsg);

  CmiFree(msg);
}

static void persistentReqGrantedHandler(void *env)
{
  int i;

  PersistentReqGrantedMsg *msg = (PersistentReqGrantedMsg *)env;
  PersistentHandle h = msg->sourceHandler;
  PersistentSendsTable *slot = (PersistentSendsTable *)h;

  /* CmiPrintf("[%d] Persistent handler granted  h:%p\n", CmiMyPe(), h); */

  for (i=0; i<PERSIST_BUFFERS_NUM; i++) {
#if 0
    slot->destAddress[i] = msg->msgAddr[i];
    slot->destSizeAddress[i] = msg->slotFlagAddress[i];
#else
    slot->destBuf[i] = msg->buf[i];
#endif
  }
  slot->destHandle = msg->destHandler;
#if DELTA_COMPRESS
  slot->destDataHandle = msg->destDataHandler;
  //CmiPrintf("+++[%d] req grant %p\n", CmiMyPe(), slot->destDataHandle);
#endif
  if (slot->messageBuf) {
    LrtsSendPersistentMsg(h, CmiGetNodeGlobal(CmiNodeOf(slot->destPE),CmiMyPartition()), slot->messageSize, slot->messageBuf);
    slot->messageBuf = NULL;
  }
  CmiFree(msg);
}

/*
  Another API:
  receiver initiate the persistent communication
*/
PersistentReq CmiCreateReceiverPersistent(int maxBytes)
{
    PersistentReq ret;
    int i;

  PersistentHandle h = getFreeRecvSlot();
  PersistentReceivesTable *slot = (PersistentReceivesTable *)h;

  setupRecvSlot(slot, maxBytes);

  ret.pe = CmiMyPe();
  ret.maxBytes = maxBytes;
  ret.myHand = h;
  ret.bufPtr = (void **)malloc(PERSIST_BUFFERS_NUM*sizeof(void*));
  for (i=0; i<PERSIST_BUFFERS_NUM; i++) {
#if 0
    ret.messagePtr[i] = slot->messagePtr[i];
    ret.recvSizePtr[i] = slot->recvSizePtr[i];
#else
    ret.bufPtr[i] = malloc(sizeof(PersistentBuf));
    memcpy(&ret.bufPtr[i], &slot->destBuf[i], sizeof(PersistentBuf));
#endif
  }

  return ret;
}

PersistentHandle CmiRegisterReceivePersistent(PersistentReq recvHand)
{
  int i;
  PersistentHandle h = getFreeSendSlot();

  PersistentSendsTable *slot = (PersistentSendsTable *)h;
  slot->destPE = recvHand.pe;
  slot->sizeMax = recvHand.maxBytes;

#if 0
  for (i=0; i<PERSIST_BUFFERS_NUM; i++) {
    slot->destAddress[i] = recvHand.messagePtr[i];
    slot->destSizeAddress[i] = recvHand.recvSizePtr[i];
  }
#else
  memcpy(slot->destBuf, recvHand.bufPtr, PERSIST_BUFFERS_NUM*sizeof(PersistentBuf));
#endif
  slot->destHandle = recvHand.myHand;
  return h;
}

/******************************************************************************
     destroy Persistent Comm handler
******************************************************************************/

/* Converse Handler */
void persistentDestroyHandler(void *env)
{             
  int i;
  PersistentDestroyMsg *msg = (PersistentDestroyMsg *)env;
  PersistentHandle h = getPersistentHandle(msg->destHandlerIndex, 0);
  CmiAssert(h!=NULL);
  CmiFree(msg);
  PersistentReceivesTable *slot = (PersistentReceivesTable *)h;

  CpvAccess(persistentReceivesTableCount) --;
  if (slot->prev) {
    slot->prev->next = slot->next;
  }
  else
    CpvAccess(persistentReceivesTableHead) = slot->next;
  if (slot->next) {
    slot->next->prev = slot->prev;
  }
  else
    CpvAccess(persistentReceivesTableTail) = slot->prev;

  for (i=0; i<PERSIST_BUFFERS_NUM; i++) 
    if (slot->destBuf[i].destAddress) 
      PerFree((char*)slot->destBuf[i].destAddress);

  clearRecvSlot(slot);

  free(slot);
}

/* FIXME: need to buffer until ReqGranted message come back? */
void CmiDestroyPersistent(PersistentHandle h)
{
  if (h == NULL) return;

  PersistentSendsTable *slot = (PersistentSendsTable *)h;
  /* CmiAssert(slot->destHandle != 0); */

  PersistentDestroyMsg *msg = (PersistentDestroyMsg *)
                              CmiAlloc(sizeof(PersistentDestroyMsg));
  msg->destHandlerIndex = slot->destHandle;

  CmiSetHandler(msg, persistentDestroyHandlerIdx);
  CmiSyncSendAndFree(slot->destPE,sizeof(PersistentDestroyMsg),msg);

  /* free this slot */
  if (slot->prev) {
    slot->prev->next = slot->next;
  }
  else
    CpvAccess(persistentSendsTableHead) = slot->next;
  if (slot->next) {
    slot->next->prev = slot->prev;
  }
  else
    CpvAccess(persistentSendsTableTail) = slot->prev;
  free(slot);

  CpvAccess(persistentSendsTableCount) --;
}


void CmiDestroyAllPersistent()
{
  PersistentSendsTable *sendslot = CpvAccess(persistentSendsTableHead);
  while (sendslot) {
    PersistentSendsTable *next = sendslot->next;
    free(sendslot);
    sendslot = next;
  }
  CpvAccess(persistentSendsTableHead) = CpvAccess(persistentSendsTableTail) = NULL;
  CpvAccess(persistentSendsTableCount) = 0;

  PersistentReceivesTable *slot = CpvAccess(persistentReceivesTableHead);
  while (slot) {
    PersistentReceivesTable *next = slot->next;
    int i;
    for (i=0; i<PERSIST_BUFFERS_NUM; i++)  {
      //if (slot->destBuf[i].destSizeAddress)
      //  CmiPrintf("Warning: CmiDestroyAllPersistent destoried buffered undelivered message.\n");
      if (slot->destBuf[i].destAddress) PerFree((char*)slot->destBuf[i].destAddress);
    }
    free(slot);
    slot = next;
  }
  CpvAccess(persistentReceivesTableHead) = CpvAccess(persistentReceivesTableTail) = NULL;
  CpvAccess(persistentReceivesTableCount) = 0;
}


void CmiPersistentInit()
{
  int i;

  persistentRequestHandlerIdx = 
       CmiRegisterHandler((CmiHandler)persistentRequestHandler);
  persistentReqGrantedHandlerIdx = 
       CmiRegisterHandler((CmiHandler)persistentReqGrantedHandler);
  persistentDestroyHandlerIdx = 
       CmiRegisterHandler((CmiHandler)persistentDestroyHandler);

#if DELTA_COMPRESS
  persistentDecompressHandlerIdx = 
      CmiRegisterHandler((CmiHandler)persistentDecompressHandler);
  persistentNoDecompressHandlerIdx = 
      CmiRegisterHandler((CmiHandler)persistentNoDecompressHandler);
#endif

  CpvInitialize(PersistentHandle*, phs);
  CpvAccess(phs) = NULL;
  CpvInitialize(int, phsSize);
  CpvInitialize(int, curphs);
  CpvAccess(curphs) = 0;

  persist_machine_init();

  CpvInitialize(PersistentSendsTable *, persistentSendsTableHead);
  CpvInitialize(PersistentSendsTable *, persistentSendsTableTail);
  CpvAccess(persistentSendsTableHead) = CpvAccess(persistentSendsTableTail) = NULL;
  CpvInitialize(int, persistentSendsTableCount);
  CpvAccess(persistentSendsTableCount) = 0;

  CpvInitialize(PersistentReceivesTable *, persistentReceivesTableHead);
  CpvInitialize(PersistentReceivesTable *, persistentReceivesTableTail);
  CpvAccess(persistentReceivesTableHead) = CpvAccess(persistentReceivesTableTail) = NULL;
  CpvInitialize(int, persistentReceivesTableCount);
  CpvAccess(persistentReceivesTableCount) = 0;
}

void CmiUsePersistentHandle(PersistentHandle *p, int n)
{
  if (n==1 && *p == NULL) { p = NULL; n = 0; }
#if  CMK_ERROR_CHECKING && 0
  {
  int i;
  for (i=0; i<n; i++)
    if (p[i] == NULL) CmiAbort("CmiUsePersistentHandle: invalid PersistentHandle.\n");
  }
#endif
  CpvAccess(phs) = p;
  CpvAccess(phsSize) = n;
  CpvAccess(curphs) = 0;
}

void CmiPersistentOneSend()
{
  if (CpvAccess(phs)) CpvAccess(curphs)++;
}

#endif
/*@}*/
