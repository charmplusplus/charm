// File: evmpool.h
// EventMsgPool: facility for reusing event messages
// Last Modified: 5.29.01 by Terry L. Wilmarth

#ifndef EVMPOOL_H
#define EVMPOOL_H
#include "evmpool.decl.h"

extern CkGroupID EvmPoolID;  // global readonly to access pool anywhere
//extern int MapSizeToIdx(int size);

// Message to initialize EventMsgPool
class PoolInitMsg : public CMessage_PoolInitMsg {
public:
  int numPools;
};

// Basic single pool of same-size messages
class Pool
{
 public:
  int numMsgs;
  void *msgPool[MAX_POOL_SIZE];
  Pool() { numMsgs = 0; }
};

// Set of message pools for various size messages; 1 EventMsgPool per PE
class EventMsgPool : public Group {
private:
  int numPools;    // corresponds to number of message sizes
  Pool *msgPools;  // the Pools
public:
  EventMsgPool(PoolInitMsg *m) { // initialize and allocate number of pools
    numPools = m->numPools;
    CkFreeMsg(m);
    msgPools = new Pool[numPools];
    for (int i=0; i<numPools; i++)
      msgPools[i].numMsgs = 0;
  }
  EventMsgPool(CkMigrateMessage *) { };
  int CheckPool(int idx); // returns number of messages in pool idx
  void *GetEventMsg(int idx); // returns a message from pool idx
  void PutEventMsg(int idx, void *msg); // puts a message in pool idx
};

#endif
