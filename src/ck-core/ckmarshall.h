#ifndef _CKMARSHALL_H_
#define _CKMARSHALL_H_

#include "charm++.h"
#include "CkMarshall.decl.h"

// This is the message type marshalled parameters get packed into:
class CkMarshallMsg : public CMessage_CkMarshallMsg {
public:
  char *msgBuf;
};

CkMarshallMsg *CkAllocateMarshallMsgNoninline(int size, const CkEntryOptions *opts);

inline CkMarshallMsg *CkAllocateMarshallMsg(int size, const CkEntryOptions *opts = NULL) {
  if (opts == NULL) {
    CkMarshallMsg *newMemory = new (size, 0) CkMarshallMsg;
    setMemoryTypeMessage(UsrToEnv(newMemory));
    return newMemory;
  } else
    return CkAllocateMarshallMsgNoninline(size, opts);
}

template <typename T>
inline T *CkAllocateMarshallMsgT(int size, const CkEntryOptions *opts) {
  int priobits = 0;
  if (opts != NULL)
    priobits = opts->getPriorityBits();
  // Allocate the message
  T *m = new (size, priobits) T;
  // Copy the user's priority data into the message
  envelope *env = UsrToEnv(m);
  setMemoryTypeMessage(env);
  if (opts != NULL) {
    CmiMemcpy(env->getPrioPtr(), opts->getPriorityPtr(), env->getPrioBytes());
    // Set the message's queueing type
    env->setQueueing((unsigned char)opts->getQueueing());
  }
  return m;
}

#endif
