/** @file
 * General implementation of persistent communication support
 * @ingroup Machine
 */

/**
 * \addtogroup Machine
*/
/*@{*/

#include "gni_pub.h"

#define PERSIST_BUFFERS_NUM             1

#define PERSIST_SEQ                     0xFFFFFFF

typedef struct  _PersistentBuf {
  void *destAddress;
  void *destSizeAddress;
  gni_mem_handle_t    mem_hndl;
} PersistentBuf;

typedef struct _PersistentSendsTable {
  int destPE;
  int sizeMax;
  PersistentHandle   destHandle;  
  PersistentBuf     destBuf[PERSIST_BUFFERS_NUM];
  void *messageBuf;
  int messageSize;
  char used;
} PersistentSendsTable;

typedef struct _PersistentReceivesTable {
#if 0
  void *messagePtr[PERSIST_BUFFERS_NUM];      /* preallocated message buffer of size "sizeMax" */
  unsigned int *recvSizePtr[PERSIST_BUFFERS_NUM];   /* pointer to the size */
#endif
  PersistentBuf     destBuf[PERSIST_BUFFERS_NUM];
  int sizeMax;
  size_t               index;
  struct _PersistentReceivesTable *prev, *next;
} PersistentReceivesTable;

CpvExtern(PersistentReceivesTable *, persistentReceivesTableHead);
CpvExtern(PersistentReceivesTable *, persistentReceivesTableTail);

CpvExtern(PersistentHandle *, phs);
CpvExtern(int, phsSize);
CpvExtern(int, curphs);

PersistentHandle getPersistentHandle(PersistentHandle h);
void *PerAlloc(int size);
void PerFree(char *msg);
int PumpPersistent();
void swapSendSlotBuffers(PersistentSendsTable *slot);
void swapRecvSlotBuffers(PersistentReceivesTable *slot);
void setupRecvSlot(PersistentReceivesTable *slot, int maxBytes);

/*@}*/
