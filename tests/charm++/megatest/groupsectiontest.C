#include "groupsectiontest.h"


CProxy_groupsectiontest gSectGproxy;  

void groupsectiontest_init(void)
{

  if(CkNumPes()<2) {
    CkError("groupsectiontest: requires at least 2 processors\n");
    megatest_finish();
  } else
	{

	  gSectGproxy=CProxy_groupsectiontest::ckNew();
	  //	      CkPrintf("[%d]made group %d\n",gSectGproxy.ckGetGroupID());

    }

}

void groupsectiontest_moduleinit(void) {
  /*
  if(CkNumPes()<2) {
    CkError("groupsectiontest: requires at least 2 processors\n");
    megatest_finish();
  } else
	{

	  //	 CkGroupID gSectGid=
	 gSectGproxy=CProxy_groupsectiontest::ckNew();
	 CkPrintf("made group %d\n",gSectGproxy.ckGetGroupID());
    }
  */
}

groupsectiontest::groupsectiontest(void)
{
  boundary=CkNumPes()/2;
  low=(CkMyPe()<boundary);
  if(low)
    {
      floor=boundary;
      ceiling=CkNumPes()-1;
    }
  else
    { 
      floor=0;
      ceiling=boundary-1;

    }
  sectionSize=ceiling-floor+1;
  iteration=0;
  msgCount=0;
  //trigger post constructor initialization
  CProxy_groupsectiontest grp(thisgroup);
  
  grp[CkMyPe()].init();
}

void groupsectiontest::init()
{
   
  // make all section
  int *elemsA= new int[CkNumPes()];
  for(int i=0;i<CkNumPes();i++)
    elemsA[i]=i;

  groupAllProxy=CProxySection_groupsectiontest(thisgroup,elemsA,CkNumPes());  

  groupsectiontest_msg *msg= new groupsectiontest_msg;
  msg->iterationCount=0;
  msg->side=low;

 // make the remote section

  int *elems= new int[sectionSize];
  for(int i=floor,j=0;i<=ceiling;i++,j++)
    elems[j]=i;
  //  CkPrintf("[%d] section of groupid %d from %d to %d size %d\n",CkMyPe(), thisgroup,  floor, ceiling, sectionSize);

  groupSectionProxy=CProxySection_groupsectiontest(thisgroup,elems,sectionSize);
  groupSectionProxy.recv(msg);

}


void groupsectiontest::nextIteration(groupsectiontest_msg *msg)
{
  // we run a reduction barrier between iterations so we can
  // implicitely test for delivery
  // also makes iteration stragglers impossible.
  // if someone fails to report in, then this test will hang.
  msg->side=low;
  //  CkPrintf("[%d] nextIteration received iter %d\n",CkMyPe(), iteration);
  groupSectionProxy.recv(msg);
}

void
groupsectiontest::recv(groupsectiontest_msg *msg)
{
  CkAssert(msg->side!=low);
  CkAssert(iteration==msg->iterationCount);

  msgCount++;
  //  CkPrintf("[%d] iteration %d received msg %d of %d\n",CkMyPe(), iteration,msgCount, sectionSize);
  if(msgCount==sectionSize)
    {
      delete msg;
      CkCallback cb(CkIndex_groupsectiontest::doneIteration(NULL), 0,thisgroup);
      contribute(sizeof(int),&msgCount, CkReduction::sum_int,cb);
      msgCount=0;
      ++iteration;
    }
}

void groupsectiontest::doneIteration(CkReductionMsg *rmsg)
{

  //  CkPrintf("[%d] completed iteration %d with %d\n",CkMyPe(), iteration, ((int *)( rmsg->getData()))[0]);
  if(iteration <NITER)
    {
      groupsectiontest_msg *msg= new groupsectiontest_msg;
      msg->iterationCount=iteration;
      /* For reference:
	 the non section way to get and broadcast on a proxy
	 CProxy_groupsectiontest grp(thisgroup);
	 grp.nextIteration(msg); 

	 Instead we use a section containing all members to make sure
	 it works.
      */
      groupAllProxy.nextIteration(msg);
    }
  else
    {
      //      CkPrintf("[%d] completed iteration %d with %d contributions\n",CkMyPe(), iteration, ((int *)( rmsg->getData()))[0]);
      megatest_finish();
    }
  delete rmsg;

}

MEGATEST_REGISTER_TEST(groupsectiontest,"ebohm",1)
#include "groupsectiontest.def.h"
