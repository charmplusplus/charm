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
#include "compress.c"
#include "machine-persistent.h"

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
  int compressStart;
  int compressSize;
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

typedef struct _PersistentDestoryMsg {
  char core[CmiMsgHeaderSizeBytes];
  PersistentHandle destHandlerIndex;
} PersistentDestoryMsg;

/* Converse handler */
int persistentRequestHandlerIdx;
int persistentReqGrantedHandlerIdx;
int persistentDestoryHandlerIdx;
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
        elan_put;
     3. Converse handler persistentReqGrantedHandler() executed on
        sender for the PersistentReqGrantedMsg. setup finish, send buffered
        message.
******************************************************************************/

PersistentHandle CmiCreateCompressPersistent(int destPE, int maxBytes, int compressStart, int compressSize)
{
  PersistentHandle h;
  PersistentSendsTable *slot;

  if (CmiMyNode() == CmiNodeOf(destPE)) return NULL;

  h = getFreeSendSlot();
  slot = (PersistentSendsTable *)h;

  slot->destPE = destPE;
  slot->sizeMax = maxBytes;
  slot->addrIndex = 0;
  PersistentRequestMsg *msg = (PersistentRequestMsg *)CmiAlloc(sizeof(PersistentRequestMsg));
  msg->maxBytes = maxBytes;
  msg->sourceHandler = h;
  msg->requestorPE = CmiMyPe();
#if DELTA_COMPRESS
  slot->previousMsg  = NULL; 
  slot->compressStart = msg->compressStart = compressStart;
  slot->compressSize = msg->compressSize = compressSize;
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
  slot->sizeMax = maxBytes;
  slot->addrIndex = 0;
  PersistentRequestMsg *msg = (PersistentRequestMsg *)CmiAlloc(sizeof(PersistentRequestMsg));
  msg->maxBytes = maxBytes;
  msg->sourceHandler = h;
  msg->requestorPE = CmiMyPe();

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
    //int historyIndex = (slot->addrIndex + 1)%PERSIST_BUFFERS_NUM;
    //char *history = (char*)(slot->destBuf[historyIndex].destAddress);
    //CmiPrintf("[%d] uncompress[NONO]  history = %p h=%p index=%d\n", CmiMyPe(), history, slot, historyIndex);
    CldRestoreHandler(msg);
    (((CmiMsgHeaderExt*)msg)->xhdl) =  (((CmiMsgHeaderExt*)msg)->xxhdl);
    CmiHandleMessage(msg);
}

static void persistentDecompressHandler(void *msg)
{
#if 1
    //  decompress  delta
    //  recovery message based on previousRecvMsg
    PersistentReceivesTable *slot = (PersistentReceivesTable *) (((CmiMsgHeaderExt*)msg)-> persistRecvHandler);
    int     historyIndex;
    int     i;
    char    *msgdata = (char*)msg;
    int     size = ((CmiMsgHeaderExt*)msg)->size;
    int     compressSize = *(int*)(msg+slot->compressStart);
    char    *decompressData =(char*) malloc(slot->compressSize);
    historyIndex = (slot->addrIndex + 1)%PERSIST_BUFFERS_NUM;
    slot->addrIndex = (slot->addrIndex + 1)%PERSIST_BUFFERS_NUM;
    // uncompress data from historyIndex data
    char *history = (char*)(slot->destBuf[historyIndex].destAddress);

    //CmiPrintf("[%d] begin uncompress message is decompressed [%d:%d]history = %p h=%p index=%d", CmiMyPe(), size, compressSize, history, slot, historyIndex);
    char *base_dst = msg+size-1;
    char *base_src = msg+size-slot->compressSize+compressSize+sizeof(int)-1;
    for(i=0; i<size - slot->compressStart - slot->compressSize-sizeof(int); i++)
    {
       *base_dst = *base_src;
       base_dst--;
       base_src--;
    }

    decompressFloatingPoint(msg+slot->compressStart+sizeof(int), decompressData, slot->compressSize, compressSize, history+slot->compressStart);
    memcpy(msg+slot->compressStart, decompressData, slot->compressSize);
    free(decompressData);
    CldRestoreHandler(msg);
    (((CmiMsgHeaderExt*)msg)->xhdl) =  (((CmiMsgHeaderExt*)msg)->xxhdl);
    if(CmiMyPe() == 5)
        CmiPrintf("[%d] end uncompress message is decompressed history = %p h=%p index=%d", CmiMyPe(), history, slot, historyIndex);
    CmiHandleMessage(msg);
#else
    CmiPrintf("[%d] msg is decompressed\n", CmiMyPe());
    CmiPrintf("\n[%d ] decompress switching   %d:%d\n", CmiMyPe(), CmiGetXHandler(msg), CmiGetHandler(msg));
    CldRestoreHandler(msg);
    (((CmiMsgHeaderExt*)msg)->xhdl) =  (((CmiMsgHeaderExt*)msg)->xxhdl);
    CmiPrintf("\n[%d ] decompress after switching   %d:%d\n", CmiMyPe(), CmiGetXHandler(msg), CmiGetHandler(msg));
    CmiHandleMessage(msg);
#endif
}

int CompressPersistentMsg(PersistentHandle h, int size, void *msg)
{
    PersistentSendsTable *slot = (PersistentSendsTable *)h;
#if 0  
    char *user_buffer = (char*) msg + sizeof(CmiMsgHeaderExt);
    char *old_buffer = (char*)( h->sentPreviousMsg);
    
    char *delta_msg = (char*) malloc(size-sizeof(CmiMsgHeaderExt));
    for(int i=0; i<size-sizeof(CmiMsgHeaderExt); i++)
    {
        delta_msg[i] = user_buffer - old_buffer;
    }

    memcpy(h->sentPreviousMsg, user_buffer, size-sizeof(CmiMsgHeaderExt));
    void  (*compressFn) (void*, void*, int, int*); 
    void *compress_msg = msg;
    switch(compress_mode)
    {
    case CMODE_ZLIB:  compressFn = zlib_compress; break;
    case CMODE_QUICKLZ: compressFn = quicklz_compress; break;
    case CMODE_LZ4:  compressFn = lz4_wrapper_compress; break;
    }
    
    ((CmiMsgHeaderExt*)compress_msg)->compress_flag = 1;
    ((CmiMsgHeaderExt*)compress_msg)->original_size  = size;
    compressFn(delta_msg, compress_msg+sizeof(CmiMsgHeaderExt), size-sizeof(CmiMsgHeaderExt), &compress_size);

    CldSwitchHandler(compress_msg, persistentDecompressHandlerIdx);
    /* ** change handler  */
    return compress_size+ sizeof(CmiMsgHeaderExt);
#else
#if 0
    ((CmiMsgHeaderExt*)msg)->size = size;
    char *history = slot->previousMsg;
    char tmp;
    int i;
    char *msgdata = (char*)msg;
    for(i=sizeof(CmiMsgHeaderExt); i < size; i++)
    {
        //tmp = msgdata[i]; 
        //msgdata[i] = msgdata[i]-history[i]; 
        msgdata[i] = msgdata[i] & 255; 
        //history[i] = tmp;
    }
    ((CmiMsgHeaderExt*)msg)-> persistRecvHandler = slot->destDataHandle;
    CmiPrintf("\n[%d ] compressing... before   old:new %d ===>%d \n", CmiMyPe(), CmiGetXHandler(msg), CmiGetHandler(msg));
    (((CmiMsgHeaderExt*)msg)->xxhdl) = (((CmiMsgHeaderExt*)msg)->xhdl);
    CldSwitchHandler(msg, persistentDecompressHandlerIdx);
    CmiPrintf("\n[%d ] after switching   %d:%d\n", CmiMyPe(), CmiGetXHandler(msg), CmiGetHandler(msg));
    return size;
#else
    int  newSize;
    void *history = slot->previousMsg;
    void *dest;
    int compressSize;
    if(history == NULL)
    {
        newSize = size;
        slot->previousMsg = msg;
        CmiReference(msg);
        (((CmiMsgHeaderExt*)msg)->xxhdl) = (((CmiMsgHeaderExt*)msg)->xhdl);
        CldSwitchHandler(msg, persistentNoDecompressHandlerIdx);
    }else
    {
        dest = malloc(size+(size+7)/8);
        compressFloatingPoint(msg+slot->compressStart, dest, slot->compressSize, &compressSize, history+slot->compressStart);
        if(compressSize>= slot->compressSize-10*sizeof(int)) //not compress
        {
            newSize = size;
            (((CmiMsgHeaderExt*)msg)->xxhdl) = (((CmiMsgHeaderExt*)msg)->xhdl);
            CldSwitchHandler(msg, persistentNoDecompressHandlerIdx);
            CmiFree(slot->previousMsg);
            slot->previousMsg = msg;
            CmiReference(msg);
        }else
        {
            *(int*)(msg+slot->compressStart) = compressSize;
            memcpy(history+slot->compressStart, msg+slot->compressStart, slot->compressSize); 
            memcpy(msg+slot->compressStart+sizeof(int), dest, compressSize);
            memcpy(msg+slot->compressStart+compressSize+sizeof(int), msg+slot->compressStart+slot->compressSize, size-slot->compressStart-slot->compressSize);
            newSize = size-slot->compressSize+compressSize;
            (((CmiMsgHeaderExt*)msg)->xxhdl) = (((CmiMsgHeaderExt*)msg)->xhdl);
            CldSwitchHandler(msg, persistentDecompressHandlerIdx);
            //CmiPrintf("\n[%d ] finish compressing \n", CmiMyPe() );
        }
        free(dest);
    }
    ((CmiMsgHeaderExt*)msg)-> persistRecvHandler = slot->destDataHandle;
    ((CmiMsgHeaderExt*)msg)->size = size;
    return newSize;
#endif
#endif
}
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

static void persistentRequestHandler(void *env)
{             
  PersistentRequestMsg *msg = (PersistentRequestMsg *)env;
  char *buf;
  int i;

  PersistentHandle h = getFreeRecvSlot();
  PersistentReceivesTable *slot = (PersistentReceivesTable *)h;

  /*slot->messagePtr = elan_CmiStaticAlloc(msg->maxBytes);*/

  /* build reply message */
  PersistentReqGrantedMsg *gmsg = CmiAlloc(sizeof(PersistentReqGrantedMsg));

#if DELTA_COMPRESS
  slot->compressStart = msg->compressStart;
  slot->compressSize = msg->compressSize;
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
    LrtsSendPersistentMsg(h, CmiNodeOf(slot->destPE), slot->messageSize, slot->messageBuf);
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
     destory Persistent Comm handler
******************************************************************************/

/* Converse Handler */
void persistentDestoryHandler(void *env)
{             
  int i;
  PersistentDestoryMsg *msg = (PersistentDestoryMsg *)env;
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
    if (slot->destBuf[i].destAddress) /*elan_CmiStaticFree(slot->messagePtr);*/
      PerFree((char*)slot->destBuf[i].destAddress);

  clearRecvSlot(slot);

  free(slot);
}

/* FIXME: need to buffer until ReqGranted message come back? */
void CmiDestoryPersistent(PersistentHandle h)
{
  if (h == NULL) return;

  PersistentSendsTable *slot = (PersistentSendsTable *)h;
  /* CmiAssert(slot->destHandle != 0); */

  PersistentDestoryMsg *msg = (PersistentDestoryMsg *)
                              CmiAlloc(sizeof(PersistentDestoryMsg));
  msg->destHandlerIndex = slot->destHandle;

  CmiSetHandler(msg, persistentDestoryHandlerIdx);
  CmiSyncSendAndFree(slot->destPE,sizeof(PersistentDestoryMsg),msg);

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


void CmiDestoryAllPersistent()
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
      if (slot->destBuf[i].destSizeAddress)
        CmiPrintf("Warning: CmiDestoryAllPersistent destoried buffered undelivered message.\n");
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
  persistentDestoryHandlerIdx = 
       CmiRegisterHandler((CmiHandler)persistentDestoryHandler);

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
