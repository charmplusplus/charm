#ifndef _CK_OBJQ_H_
#define _CK_OBJQ_H_

#include "cklists.h"

class Chare;

// Converse token message
class ObjectToken {
public:
  char core[CmiMsgHeaderSizeBytes];
  Chare *objPtr;               		// pointer to object
  void *message;			// envelope
};

// object message queue
class CkObjectMsgQ {
private:
  void * objQ;
public:
  CkObjectMsgQ(): objQ(NULL) {}
  ~CkObjectMsgQ();
  inline void *queue() { return objQ; }
  void create();
  int length() const;
  void process();
};

class TokenPool: public SafePool<ObjectToken*> {
private:
    static inline ObjectToken *_alloc(void) {
      return (ObjectToken*)CmiAlloc(sizeof(ObjectToken));
    }
    static inline void _free(ObjectToken *ptr) {
      CmiFree((void*)ptr);
    }
public:
    TokenPool(): SafePool<ObjectToken*>(_alloc, _free) {}
};

CkpvExtern(TokenPool*, _tokenPool);

Chare * CkFindObjectPtr(envelope *);
void _enqObjQueue(Chare *obj, envelope *env);
void _ObjectQHandler(void *converseMsg);
void _TokenHandler(void *tokenMsg);

#endif
