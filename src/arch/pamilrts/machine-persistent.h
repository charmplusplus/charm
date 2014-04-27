#ifndef MACHINE_PERSISTENT_H
#define MACHINE_PERSISTENT_H

/** @file
 * General implementation of persistent communication support
 * @ingroup Machine
 */

/**
 * \addtogroup Machine
*/
/*@{*/

#include "pami.h"
#define PERSIST_MIN_SIZE               EAGER_CUTOFF 
//#define COPY_HISTORY                          1
// one is for receive one is to store the previous msg
#if DELTA_COMPRESS
#if COPY_HISTORY 
#define PERSIST_BUFFERS_NUM             1
#else
#define PERSIST_BUFFERS_NUM             2
#endif
#else
#define PERSIST_BUFFERS_NUM             1
#endif

#define PERSIST_SEQ                     0xFFFFFFF

#define IS_PERSISTENT_MEMORY(ptr)          (REFFIELD(msg) > PERSIST_SEQ/2)

typedef struct  _PersistentBuf {
  void *destAddress;
//  void *destSizeAddress;
} PersistentBuf;

typedef struct _PersistentSendsTable {
  int destPE;
  int sizeMax;
  PersistentHandle   destHandle; 
  PersistentBuf     destBuf[PERSIST_BUFFERS_NUM];
  void *messageBuf;
  int messageSize;
  struct _PersistentSendsTable *prev, *next;
#if DELTA_COMPRESS
  PersistentHandle destDataHandle;
  void  *previousMsg;
  int   previousSize;
  int   compressStart;
  int   compressSize;
  int 	dataType;
  int   compressFlag;
#endif
  int addrIndex;
} PersistentSendsTable;

typedef struct _PersistentReceivesTable {
  PersistentBuf     destBuf[PERSIST_BUFFERS_NUM];
  int sizeMax;
  size_t               index;
  struct _PersistentReceivesTable *prev, *next;
  int           addrIndex;
#if DELTA_COMPRESS
  int   compressStart;
  int 	dataType;
  void  *history;
#endif
} PersistentReceivesTable;

CpvExtern(PersistentReceivesTable *, persistentReceivesTableHead);
CpvExtern(PersistentReceivesTable *, persistentReceivesTableTail);

CpvExtern(PersistentHandle *, phs);
CpvExtern(int, phsSize);
CpvExtern(int, curphs);

void _initPersistent( pami_context_t *contexts, int nc);
PersistentHandle getPersistentHandle(PersistentHandle h, int toindex);
void *PerAlloc(int size);
void PerFree(char *msg);
int PumpPersistent();
void swapSendSlotBuffers(PersistentSendsTable *slot);
void swapRecvSlotBuffers(PersistentReceivesTable *slot);
void setupRecvSlot(PersistentReceivesTable *slot, int maxBytes);
void clearRecvSlot(PersistentReceivesTable *slot);

#endif
