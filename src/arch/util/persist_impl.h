
typedef struct _PersistentSendsTable {
  int destPE;
  int sizeMax;
  PersistentHandle   destHandle;  
  void *destAddress;
  void *destSizeAddress;
  void *messageBuf;
  int messageSize;
  char used;
} PersistentSendsTable;

typedef struct _PersistentReceivesTable {
  void *messagePtr;        /* preallocated message buffer of size "sizeMax" */
  int recvSize;
  int sizeMax;
  struct _PersistentReceivesTable *prev, *next;
} PersistentReceivesTable;

extern PersistentReceivesTable *persistentReceivesTableHead;
extern PersistentReceivesTable *persistentReceivesTableTail;

extern PersistentHandle  *phs;
extern int phsSize;

extern void *PerAlloc(int size);
extern void PerFree(char *msg);
extern void CmiSendPersistentMsg(PersistentHandle h, int destPE, int size, void *m);
extern void PumpPersistent();


