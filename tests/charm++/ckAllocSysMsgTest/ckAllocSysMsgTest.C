#include "charm++.h"
#include "ckAllocSysMsgTest.decl.h"
#include <stdio.h>
#include <math.h>

#include "ckAllocSysMsgTest.h"

/* This test is to demonstrate a race condition in the SMP build of
   charm++.

   Main launches a batch of Array and Group in a loop.
   
   The Array and group each engage in a ring and complete by reporting
   to main.  Main will not start on the new batch until all elements
   of the previous batch complete.
   
   This test is expected to hang in SMP mode if the bug is present.

   By default a periodic timeout is registered which will trigger an abort
   if the number of batches doesn't progress between periods. 

   We do not construct the group or array inside main because the
   CkAllocSysMsg race condition only affects group construction in
   parallel execution.

   The array must be multidimensional so that its indices will have
   non zero values 
*/


CProxy_main mainProxy;		
int period;
void timeout(void *, double thing)
{
  if(--period>0)
    CcdCallOnCondition(CcdPERIODIC_1minute,timeout,NULL);
  else
    CkAbort("hung job failure");
}


main::main(CkArgMsg *msg)
{
  reportedArr=0;
  reportedGrp=0;
  completeBatches=0;
  if(msg->argc>5)
    CkAbort("Usage: ckAllocSysMsgTest arrSize nBatches batchSize timeout\n Where arrSize, nBatches, batchSize are int >0 \n and period is minutes, a timeout of zero disables the default 1 minute timer\n");
  //get arrsize and wastetime from cmd line
  arrSize= 30;
  nBatches =6;
  batchSize=10;
  period=2;
  if(msg->argc>1)
    arrSize =atoi(msg->argv[1]);
  if(msg->argc>2)
    nBatches =atoi(msg->argv[2]);
  if(msg->argc>3)
    batchSize =atoi(msg->argv[3]);
  if(msg->argc>4)
    period=atoi(msg->argv[4]);
  if(arrSize<=0 || nBatches <= 0 || batchSize <= 0)
    CkAbort("Usage: ckAllocSysMsgTest arrSize nBatches batchSize period\n Where arrSize, nBatches, batchSize are int >0  and period is timeout in seconds, a timeout of zero disables the timer\n");
  mainProxy=thisProxy;
  if(period>0)
    CcdCallOnCondition(CcdPERIODIC_1minute,timeout,NULL);
  arrProxy=CProxy_RaceMeArr::ckNew(arrSize,arrSize,arrSize, arrSize);
  arrProxy.doneInserting();
  mainProxy.startBatching();
}

void main::startBatching()
{
  CkPrintf("batch %d\n",completeBatches);
  for(int i=0;i<batchSize;i++)
    {
      CProxy_RaceMeGrp grpProxy=CProxy_RaceMeGrp::ckNew();
      arrProxy(0,0,0).recvMsg();
      grpProxy[0].recvMsg();
      
    }
}



void main::reportInArr()
{
  ++reportedArr;
  if(reportedArr==batchSize)
    nextBatch();
}

void main::reportInGrp()
{
 ++reportedGrp;
  if(reportedGrp==batchSize)
    nextBatch();

}

void main::nextBatch()
{

  if(reportedGrp==batchSize && reportedArr ==batchSize)
    {
      completeBatches++;
      if(completeBatches==nBatches)
	mainProxy.done();
      else
	{
	  reportedArr=reportedGrp=0;
	  startBatching();
	}
    }
}

void main::done()
{
  CkExit();
}

void RaceMeGrp::recvMsg()
{
  if(CkMyPe()==CkNumPes()-1)
    mainProxy.reportInGrp();
  else
    {
      CProxy_RaceMeGrp rtest(thisgroup);
      rtest[CkMyPe()+1].recvMsg();
    }
}

void RaceMeArr::recvMsg()
{
  if(thisIndex.x==nElements-1)
    {
      if(thisIndex.y==nElements-1)
      {
	if(thisIndex.z==nElements-1)
	  mainProxy.reportInArr();
	else
	  thisProxy(thisIndex.x, thisIndex.y, thisIndex.z+1).recvMsg();
      }
      else
	thisProxy(thisIndex.x, thisIndex.y+1, thisIndex.z).recvMsg();
    }
  else
    thisProxy(thisIndex.x+1, thisIndex.y, thisIndex.z).recvMsg();
}
#include "ckAllocSysMsgTest.def.h"
