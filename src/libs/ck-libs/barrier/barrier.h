/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "Barrier.decl.h"

typedef void (*voidfn)();
extern int barrierInit(void);

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
  barrier(void);
  void notify(void);
  void callFP(void);
  void reset(void);
  void atBarrier(FP *);
};
