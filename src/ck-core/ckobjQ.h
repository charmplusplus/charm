#ifndef _CK_OBJQ_H_
#define _CK_OBJQ_H_

class Chare;

// Converse token message
class ObjectToken {
public:
  char core[CmiMsgHeaderSizeBytes];
  Chare *objPtr;                   // point to object
  void *message;			 // envelope
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

#define MAXTokenMSGS 52

class TokenPool {
  private:
    int num;
    ObjectToken *msgs[MAXTokenMSGS];
    static void *_alloc(void) {
      register envelope *env = (envelope*)CmiAlloc(sizeof(ObjectToken));
      return env;
    }
  public:
    TokenPool() {
      for(int i=0;i<MAXTokenMSGS;i++) msgs[i] = (ObjectToken*)CmiAlloc(sizeof(ObjectToken));
      num = MAXTokenMSGS;
    }
    ObjectToken *get(void) {
      /* CkAllocSysMsg() called in .def.h is not thread of sigio safe */
      if (CmiImmIsRunning()) return (ObjectToken*)CmiAlloc(sizeof(ObjectToken));
//if (num==0) CmiPrintf("[%d] underflow\n", CkMyPe());
      return (num ? msgs[--num] : (ObjectToken*)CmiAlloc(sizeof(ObjectToken)));
    }
    void put(ObjectToken *m) {
      if (num==MAXTokenMSGS || CmiImmIsRunning()) {
//CmiPrintf("overflow!\n");
        CmiFree(m);
      }
      else
        msgs[num++] = m;
    }
};

CkpvExtern(TokenPool*, _tokenPool);

extern int index_objectQHandler;
extern int index_tokenHandler;

Chare * CkFindObjectPtr(envelope *);
void _enqObjQueue(Chare *obj, envelope *env);
void _ObjectQHandler(void *converseMsg);
void _TokenHandler(void *tokenMsg);

#endif
