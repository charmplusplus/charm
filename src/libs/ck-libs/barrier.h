#include "Barrier.decl.h"

typedef void (*voidfn)();
extern int barrierInit();

class DUMMY : public CMessage_DUMMY 
{
};

class FP : public CMessage_FP
{
public:
  voidfn fp;
};

class barrier : public Group 
{
  int myPe, myLeft, myRight, myParent, myGroup;
  int kidscount;
  voidfn fnptr;

public:
  /* entry methods */
  barrier(DUMMY *);
  void notify(DUMMY *);
  void callFP(DUMMY *);
  void reset(DUMMY *);
  void atBarrier(FP *);
};
