/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "charm++.h"
#include "barrier.h"
#include "Barrier.def.h"

barrier::barrier(DUMMY *m)
{
  myPe = CkMyPe();
  myLeft = (myPe*2)+1;
  myRight = myLeft+1;
  myParent = (myPe % 2 == 0) ? ((myPe-2)/2) : ((myPe-1)/2);
  myGroup = CkGetGroupID();
  kidscount = 0;
  if (myRight >= CkNumPes())
    kidscount++;
  if (myLeft >= CkNumPes())
    kidscount++;
  delete m;
}

void barrier::reset(DUMMY *m)
{
  kidscount = 0;
  if (myRight >= CkNumPes())
    kidscount++;
  if (myLeft >= CkNumPes())
    kidscount++;
  delete m;
}

void barrier::atBarrier(FP *m)
{
  CProxy_barrier grp(myGroup);

  kidscount++;
  if (kidscount == 3) {
    if (myPe == 0)
      grp.callFP(new DUMMY, CkMyPe());
    else
      grp.notify(new DUMMY, myParent);
  }
  fnptr = m->fp;
  delete m;
}

void barrier::notify(DUMMY *m)
{
  CProxy_barrier grp(myGroup);

  kidscount++;
  if (kidscount == 3) {
    if (myPe == 0)
      grp.callFP(new DUMMY, CkMyPe());
    else
      grp.notify(new DUMMY, myParent);
  }
  delete m;
}

void barrier::callFP(DUMMY *m)
{
  CProxy_barrier grp(myGroup);

  if (myLeft < CkNumPes())
    grp.callFP(new DUMMY, myLeft);
  if (myRight < CkNumPes())
    grp.callFP(new DUMMY, myRight);
  fnptr();
  delete m;
}

int barrierInit()
{
  int g = CProxy_barrier::ckNew(new DUMMY);
  return g;
}

