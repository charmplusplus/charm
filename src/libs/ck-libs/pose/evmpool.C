// File: evmpool.C
// EventMsgPool: facility for reusing event messages
// Last Modified: 5.29.01 by Terry L. Wilmarth

#include "charm++.h"
#include "pose.h"
#include "evmpool.def.h"

CkGroupID EvmPoolID;

// Returns number of messages in pool idx
int EventMsgPool::CheckPool(int idx)
{
  if (idx < numPools)
    return msgPools[idx].numMsgs;
  else return MAX_POOL_SIZE;
}

// Returns a message from pool idx; decrements the count
void *EventMsgPool::GetEventMsg(int idx)
{
  msgPools[idx].numMsgs--;
  return msgPools[idx].msgPool[msgPools[idx].numMsgs];
}

// Puts a message in pool idx; increments the count
void EventMsgPool::PutEventMsg(int idx, void *msg)
{
  msgPools[idx].msgPool[msgPools[idx].numMsgs] = msg;
  msgPools[idx].numMsgs++;
}
