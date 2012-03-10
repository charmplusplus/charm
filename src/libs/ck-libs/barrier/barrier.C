#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "charm++.h"
#include "barrier.h"
#include "barrier.def.h"

barrier::barrier(void)
{
  myPe = CkMyPe();
  myLeft = (myPe*2)+1;
  myRight = myLeft+1;
  myParent = (myPe % 2 == 0) ? ((myPe-2)/2) : ((myPe-1)/2);
  myGroup = thisgroup;
  kidscount = 0;
  if (myRight >= CkNumPes())
    kidscount++;
  if (myLeft >= CkNumPes())
    kidscount++;
}

void barrier::reset(void)
{
  kidscount = 0;
  if (myRight >= CkNumPes())
    kidscount++;
  if (myLeft >= CkNumPes())
    kidscount++;
}

void barrier::atBarrier(FP *m)
{
  CProxy_barrier grp(myGroup);

  kidscount++;
  if (kidscount == 3) {
    if (myPe == 0)
      grp[CkMyPe()].callFP();
    else
      grp[myParent].notify();
  }
  fnptr = m->fp;
  delete m;
}

void barrier::notify(void)
{
  CProxy_barrier grp(myGroup);

  kidscount++;
  if (kidscount == 3) {
    if (myPe == 0)
      grp[CkMyPe()].callFP();
    else
      grp[myParent].notify();
  }
}

void barrier::callFP(void)
{
  CProxy_barrier grp(myGroup);

  if (myLeft < CkNumPes())
    grp[myLeft].callFP();
  if (myRight < CkNumPes())
    grp[myRight].callFP();
  fnptr();
}

CkGroupID barrierInit(void)
{
  return CProxy_barrier::ckNew();
}

