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

barrier::barrier(void)
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
      grp.callFP(CkMyPe());
    else
      grp.notify(myParent);
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
      grp.callFP(CkMyPe());
    else
      grp.notify(myParent);
  }
}

void barrier::callFP(void)
{
  CProxy_barrier grp(myGroup);

  if (myLeft < CkNumPes())
    grp.callFP(myLeft);
  if (myRight < CkNumPes())
    grp.callFP(myRight);
  fnptr();
}

int barrierInit(void)
{
  int g = CProxy_barrier::ckNew();
  return g;
}

