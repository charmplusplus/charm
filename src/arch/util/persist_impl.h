
#define PERSIST_BUFFERS_NUM   2

typedef struct _PersistentSendsTable {
  int destPE;
  int sizeMax;
  PersistentHandle   destHandle;  
  void *destAddress[PERSIST_BUFFERS_NUM];
  void *destSizeAddress[PERSIST_BUFFERS_NUM];
  void *messageBuf;
  int messageSize;
  char used;
} PersistentSendsTable;

typedef struct _PersistentReceivesTable {
  void *messagePtr[PERSIST_BUFFERS_NUM];      /* preallocated message buffer of size "sizeMax" */
  unsigned int *recvSizePtr[PERSIST_BUFFERS_NUM];   /* pointer to the size */
  int sizeMax;
  struct _PersistentReceivesTable *prev, *next;
} PersistentReceivesTable;

extern PersistentReceivesTable *persistentReceivesTableHead;
extern PersistentReceivesTable *persistentReceivesTableTail;

extern PersistentHandle  *phs;
extern int phsSize;

void *PerAlloc(int size);
void PerFree(char *msg);
void CmiSendPersistentMsg(PersistentHandle h, int destPE, int size, void *m);
int PumpPersistent();
void swapSendSlotBuffers(PersistentSendsTable *slot);
void swapRecvSlotBuffers(PersistentReceivesTable *slot);


