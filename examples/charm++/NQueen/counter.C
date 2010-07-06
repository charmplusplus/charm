#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "charm++.h"

#include "counter.h"
#include "Counter.def.h"

counter::counter(DUMMY *m)
{ 
  // initialize the local count;
  mygrp = thisgroup;
  myCount = totalCount = 0;
  waitFor = CkNumPes(); // wait for all processors to report
  threadId = NULL;
}

void counter::increment()
{
  myCount++;
}

void counter::sendCounts(DUMMY *m)
  // This method is invoked via a broadcast. Each branch then reports 
  // its count to the branch on 0 (or via a spanning tree.)
{
  CProxy_counter grp(mygrp);
  grp[0].childCount(new countMsg(myCount));
  delete m;
}

void counter::childCount(countMsg *m)
{
  totalCount += m->count;
  waitFor--;
  if (waitFor == 0) 
    if (threadId) { CthAwaken(threadId);}
}

int counter::getTotalCount()
{
  CProxy_counter grp(mygrp);
  grp.sendCounts(new DUMMY);//this is a broadcast, as no processor is mentioned
  threadId = CthSelf();
  while (waitFor != 0)  CthSuspend(); 
  return totalCount;
}
CkGroupID  counterInit()
{
  DUMMY *m = new  DUMMY;
  CkGroupID g =CProxy_counter::ckNew(m);  // create a new group of class "counter"
  return g;
}
