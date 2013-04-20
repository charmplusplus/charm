#include "barrier.decl.h"

typedef void (*voidfn)();
extern CkGroupID barrierInit(void);

class FP : public CMessage_FP
{
public:
  voidfn fp;
};

class Barrier : public CBase_Barrier
{
  int myPe, myLeft, myRight, myParent;
  CkGroupID myGroup;
  int kidscount;
  voidfn fnptr;

public:
  /* entry methods */
  Barrier(void);
  void notify(void);
  void callFP(void);
  void reset(void);
  void atBarrier(FP *);
};
