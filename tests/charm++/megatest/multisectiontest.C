#include "multisectiontest.h"

#define ArraySize 10
#define NUMGROUPS 3
#define NUMARRAYS 3

void multisectiontest_init()
{

  int numgroups=NUMGROUPS;
  int numarrays=NUMARRAYS;
    
  if(CkNumPes()<2) {
    CkError("multisectiontest: requires at least 2 processors\n");
    megatest_finish();
  } else
	{

	  CProxy_multisectiontest_master masterproxy=CProxy_multisectiontest_master::ckNew(numgroups);

	  
	  //	      CkPrintf("[%d]made group %d\n",gSectGproxy.ckGetGroupID());

	  // make 3 arrays and 3 groups
	  // array of group IDs

	  CkGroupID *gidArr= new CkGroupID[numgroups];
	  CkArrayID *aidArr= new CkArrayID[numarrays];

	  CProxy_multisectiontest_grp *Gproxy= new CProxy_multisectiontest_grp[numgroups];
	  CProxy_multisectiontest_array1d *Aproxy= new CProxy_multisectiontest_array1d[numarrays];
	  for(int i=0;i<numgroups;i++)
	    {
	      //	      Gproxy[i]=
	      gidArr[i]=CProxy_multisectiontest_grp::ckNew(numgroups, masterproxy.ckGetGroupID());	  

	    }
	  for(int i=0;i<numarrays;i++)
	    {
	      Aproxy[i]=CProxy_multisectiontest_array1d::ckNew(masterproxy.ckGetGroupID(),ArraySize);	  
	      aidArr[i]=Aproxy[i].ckGetArrayID();
	    }
	  // make sections
	  int boundary=CkNumPes()/2;
	  int floor=boundary;
	  int ceiling=CkNumPes()-1;
	  int sectionSize=ceiling-floor+1;
	  // make sections
	  int aboundary=ArraySize/2;
	  int afloor=aboundary;
	  int aceiling=ArraySize-1;
	  int asectionSize=aceiling-afloor+1;
	  //	  CkPrintf("bound %d floor %d ceiling %d sectionSize %d\n",boundary, floor, ceiling, sectionSize);
	  //	  CkPrintf("abound %d afloor %d aceiling %d asectionSize %d\n",aboundary, afloor, aceiling, asectionSize);

	  // cross section lower half of each group
	  int **elems= new int*[numgroups];
	  elems[0]= new int[sectionSize];
	  elems[1]= new int[sectionSize];
	  elems[2]= new int[sectionSize];
	  int *nelems=new int[numgroups];
	  for(int k=0;k<numgroups;k++)
	    {
	      nelems[k]=sectionSize;
	      for(int i=floor,j=0;i<=ceiling;i++,j++)
		elems[k][j]=i;
	    }
	  //	  CProxySection_multisectiontest_grp
	  //	  groupLowProxy=CProxySection_multisectiontest_grp(numgroups,gidArr,elems,nelems);
	  CProxySection_multisectiontest_grp groupLowProxy(numgroups, gidArr,elems,nelems);
	  //  CkPrintf("[%d] section of groupid %d from %d to %d size
	  //  %d\n",CkMyPe(), thisgroup,  floor, ceiling,
	  //  sectionSize);

	  // cross section lower half of each array
	  CkArrayIndex **aelems= new CkArrayIndex*[numarrays];
	  aelems[0]= new CkArrayIndex[asectionSize];
	  aelems[1]= new CkArrayIndex[asectionSize];
	  aelems[2]= new CkArrayIndex[asectionSize];
	  int *naelems=new int[numarrays];
	  for(int k=0;k<numarrays;k++)
	    {
	      naelems[k]=asectionSize;
	    for(int i=afloor,j=0;i<=aceiling;i++,j++)
	      aelems[k][j]=CkArrayIndex1D(i);
	    }
	  //	  CProxySection_multisectiontest_array1d
	  //	  arrayLowProxy=CProxySection_multisectiontest_array1d(numarrays,aidArr,aelems,naelems);
	  //	  CProxySection_multisectiontest_array1d
	  //	  arrayLowProxy(numarrays,aidArr,constaelems,naelems);
	  CProxySection_multisectiontest_array1d arrayLowProxy(numarrays,aidArr,aelems,naelems);
	  // cross section other half of each group
	  floor=0;
	  ceiling=boundary-1;
	  sectionSize=ceiling-floor+1;
	  afloor=0;
	  aceiling=aboundary-1;
	  asectionSize=aceiling-afloor+1;
	  //	  CkPrintf("bound %d floor %d ceiling %d sectionSize %d\n",boundary, floor, ceiling, sectionSize);
	  //	  CkPrintf("abound %d afloor %d aceiling %d asectionSize %d\n",aboundary, afloor, aceiling, asectionSize);

	  // section could be different size
	  int **elemsH= new int*[numgroups];
	  elemsH[0]= new int[sectionSize];
	  elemsH[1]= new int[sectionSize];
	  elemsH[2]= new int[sectionSize];
	  for(int k=0;k<numgroups;k++)
	    {
	      nelems[k]=sectionSize;
	      for(int i=floor,j=0;i<=ceiling;i++,j++)
		elemsH[k][j]=i;	  	  
	    }
	  CProxySection_multisectiontest_grp groupHighProxy(numgroups,gidArr,elemsH,nelems);

	  // cross section upper half of each array
	  // cross section lower half of each array
	  CkArrayIndex **aelemsH= new CkArrayIndex*[numarrays];
	  aelemsH[0]= new CkArrayIndex[asectionSize];
	  aelemsH[1]= new CkArrayIndex[asectionSize];
	  aelemsH[2]= new CkArrayIndex[asectionSize];
	  for(int k=0;k<numarrays;k++)
	    {
	      naelems[k]=asectionSize;
	      for(int i=afloor,j=0;i<=aceiling;i++,j++)
		aelemsH[k][j]=CkArrayIndex1D(i);
	    }
	  CProxySection_multisectiontest_array1d arrayHighProxy(numarrays,aidArr, aelemsH, naelems);

	  // send IDs to master
	  multisectionGID_msg *mmsg= new (numgroups) multisectionGID_msg;
	  for(int i=0;i<numgroups;++i)
	    mmsg->IDs[i]=gidArr[i];
	  mmsg->numIDs=numgroups;
	  masterproxy[0].recvID(mmsg);
	  // send IDs to sections
	  multisectionAID_msg *amsg= new (numarrays) multisectionAID_msg;
	  for(int i=0;i<numarrays;++i)
	    amsg->IDs[i]=aidArr[i];
	  amsg->numIDs=numarrays;
	  groupLowProxy.recvID(amsg);

	  multisectionAID_msg *amsg2= new (numarrays) multisectionAID_msg;
	  for(int i=0;i<numarrays;++i)
	    amsg2->IDs[i]=aidArr[i];
	  amsg2->numIDs=numarrays;
	  groupHighProxy.recvID(amsg2);
	 
	  
	  multisectionGID_msg *gmsg= new (numgroups) multisectionGID_msg;
	  for(int i=0;i<numgroups;++i)
	    gmsg->IDs[i]=gidArr[i];
	  gmsg->numIDs=numgroups;
	  arrayLowProxy.recvID(gmsg);
	  multisectionGID_msg *gmsg2= new (numgroups) multisectionGID_msg;
	  for(int i=0;i<numgroups;++i)
	    gmsg2->IDs[i]=gidArr[i];
	  gmsg2->numIDs=numgroups;
	  arrayHighProxy.recvID(gmsg2);
	  
    }

}

void multisectiontest_moduleinit(void) {
  /*
  if(CkNumPes()<2) {
    CkError("multisectiontest_grp: requires at least 2 processors\n");
    megatest_finish();
  } else
	{

	  //	 CkGroupID gSectGid=
	 gSectGproxy=CProxy_multisectiontest_grp::ckNew();
	 CkPrintf("made group %d\n",gSectGproxy.ckGetGroupID());
    }
  */
}

multisectiontest_master::multisectiontest_master(int groups):numgroups(groups)
{
  iteration=0;
  msgCount=0;
  startCount=0;
  groupIDs=NULL;
}

void multisectiontest_master::recvID( multisectionGID_msg *m)
{
  groupIDs=new CkGroupID[numgroups];
  // setup the proxies and initiate computation
  if(CkMyPe()==0)
    for(int i=0;i<numgroups;i++)
      {
	groupIDs[i]=m->IDs[i];
      }
  else
    CkAbort("master message incorrectly found off 0\n");
  //  CkPrintf("[%d] master received IDs\n",CkMyPe());
  delete m;  
  finishSetup();
}

void multisectiontest_master::finishSetup()
{
 if(startCount==NUMARRAYS+NUMGROUPS && groupIDs!=NULL)
    {
      // everyone is ready, start the show
      multisectiontest_msg *gmsg= new multisectiontest_msg;
      for(int i=0;i<numgroups;i++)
	{
	  multisectiontest_msg *gmsg=new multisectiontest_msg;
	  gmsg->iterationCount=iteration;
	  CkGroupID bar=groupIDs[i];
	  CProxy_multisectiontest_grp foo(bar);
	  foo.nextIteration(gmsg);
	}
    }
}

/** the test_grps reduce to the master and the master starts iterations*/
void multisectiontest_master::doneSetup(CkReductionMsg *rmsg)
{
  ++startCount;
  delete rmsg;
  finishSetup();
  //  CkPrintf("master setup count %d of %d\n",startCount,NUMARRAYS+NUMGROUPS);
}

/** the test_grps reduce to the master and the master controls iterations*/
void multisectiontest_master::doneIteration(CkReductionMsg *rmsg)
{


  ++msgCount;
  //  CkPrintf("M[%d] completed iteration %d with %d msgcount %d of %d\n",CkMyPe(), iteration, ((int *)( rmsg->getData()))[0],msgCount,numgroups);
  delete rmsg;
  if(msgCount==numgroups) 
    {
      msgCount=0;
      ++iteration;
      if(iteration <NITER)
	{

	  for(int i=0;i<numgroups;i++)
	    {
	      multisectiontest_msg *msg= new multisectiontest_msg;
	      msg->iterationCount=iteration;
	      CProxy_multisectiontest_grp foo=CProxy_multisectiontest_grp(groupIDs[i]);
	      foo.nextIteration(msg);
	    }
	}
      else
	{
	  //      CkPrintf("[%d] completed iteration %d with %d contributions\n",CkMyPe(), iteration, ((int *)( rmsg->getData()))[0]);
	  megatest_finish();
	}
    }


}

multisectiontest_grp::multisectiontest_grp(int numgroups,CkGroupID master):masterGroup(master)
{

  // make sections
  boundary=CkNumPes()/2;
  floor=boundary;
  ceiling=CkNumPes()-1;
  sectionSize=ceiling-floor+1;
  // make sections
  aboundary=ArraySize/2;
  afloor=aboundary;
  aceiling=ArraySize-1;
  asectionSize=aceiling-afloor+1;

  
  boundary=CkNumPes()/2;
  low=(CkMyPe()<boundary);
  if(!low)
    { 
      floor=0;
      ceiling=boundary-1;
      afloor=0;
      aceiling=aboundary-1;

    }
  sectionSize=ceiling-floor+1;
  iteration=0;
  msgCount=0;
  //trigger post constructor initialization
  //  CProxy_multisectiontest grp(thisgroup);
  
  //  grp[CkMyPe()].init();
}

void multisectiontest_grp::recvID( multisectionAID_msg *m)
{
   
  // cross section your destinations in each array
  //  CkPrintf("G(%d)[%d] bound %d floor %d ceiling %d sectionSize %d\n",thisProxy.ckGetGroupID().idx, CkMyPe(), boundary, floor, ceiling, sectionSize);
  //  CkPrintf("G(%d)[%d] abound %d afloor %d aceiling %d asectionSize %d\n",thisProxy.ckGetGroupID().idx, CkMyPe(),aboundary, afloor, aceiling, asectionSize);

  CkArrayIndex **aelems= new CkArrayIndex*[NUMARRAYS];
  aelems[0]= new CkArrayIndex[asectionSize];
  aelems[1]= new CkArrayIndex[asectionSize];
  aelems[2]= new CkArrayIndex[asectionSize];
  int *naelems=new int[NUMARRAYS];
  for(int k=0;k<NUMARRAYS;k++)
    {
      naelems[k]=asectionSize;
      for(int i=afloor,j=0;i<=aceiling;i++,j++)
	aelems[k][j]=CkArrayIndex1D(i);
    }
  multisectionProxy=CProxySection_multisectiontest_array1d(NUMARRAYS,m->IDs,aelems,naelems);
  delete m;  
  CkCallback cb(CkIndex_multisectiontest_master::doneSetup(NULL), 0,masterGroup);
  contribute(sizeof(int),&msgCount, CkReduction::sum_int,cb);

}


void multisectiontest_grp::nextIteration(multisectiontest_msg *msg)
{
  // we run a reduction barrier between iterations so we can
  // implicitely test for delivery
  // also makes iteration stragglers impossible.
  // if someone fails to report in, then this test will hang.
  msg->side=low;
  iteration=msg->iterationCount;
  //  CkPrintf("G(%d)[%d] nextIteration received iter %d\n",thisProxy.ckGetGroupID().idx,CkMyPe(), iteration);
  multisectionProxy.recv(msg);
}



void multisectiontest_grp::recvSProxy(  multisection_proxymsg *m)
{
  CkAbort("don't do this\n");
  //  CkPrintf("G(%d)[%d] group received proxymsg\n",thisProxy.ckGetGroupID().idx,CkMyPe());
  multisectiontest_msg *msg= new multisectiontest_msg;
  m->aproxy.recv(msg);
  delete m;
}
void
multisectiontest_grp::recv(multisectiontest_msg *msg)
{
  CkAssert(msg->side!=low);
  CkAssert(iteration==msg->iterationCount);

  msgCount++;
  //  CkPrintf("[%d] iteration %d received msg %d of %d\n",CkMyPe(),
  //  iteration,msgCount, sectionSize);
  delete msg;
  //  CkPrintf("G(%d)[%d] iteration %d received msgCount %d of %d\n",thisProxy.ckGetGroupID().idx,CkMyPe(),iteration,msgCount, asectionSize*NUMARRAYS);
  if(msgCount==asectionSize*NUMARRAYS)
    {
      //      CkPrintf("G(%d)[%d] iteration %d completed\n",thisProxy.ckGetGroupID().idx,CkMyPe(),iteration);
      CkCallback cb(CkIndex_multisectiontest_master::doneIteration(NULL), 0,masterGroup);
      contribute(sizeof(int),&msgCount, CkReduction::sum_int,cb);
      msgCount=0;
      ++iteration;
    }
}



multisectiontest_array1d::multisectiontest_array1d(CkGroupID master):  masterGroup(master)
{
  gotIds=false;
  boundary=CkNumPes()/2;
  floor=boundary;
  ceiling=CkNumPes()-1;
  sectionSize=ceiling-floor+1;
  aboundary=ArraySize/2;
  afloor=aboundary;
  aceiling=ArraySize-1;
  asectionSize=aceiling-afloor+1;

  
  boundary=CkNumPes()/2;
  low=(thisIndex<aboundary);
  if(!low)
    { 
      floor=0;
      ceiling=boundary-1;
      afloor=0;
      aceiling=aboundary-1;

    }
  sectionSize=ceiling-floor+1;
  iteration=0;
  msgCount=0;
}


void multisectiontest_array1d::recvID(multisectionGID_msg *m)
{
  // CkPrintf("A(%d)[%d] array received IDmsg\n",CkGroupID(thisProxy.ckGetArrayID()).idx,thisIndex);
 CkAssert(gotIds==false);
 gotIds=true;
  // cross section your destinations in each array
  int **elems= new int*[NUMGROUPS];
  elems[0]= new int[sectionSize];
  elems[1]= new int[sectionSize];
  elems[2]= new int[sectionSize];
  int *nelems=new int[NUMGROUPS];
  for(int k=0;k<NUMGROUPS;k++)
    {
      nelems[k]=sectionSize;
      for(int i=floor,j=0;i<=ceiling;i++,j++)
	elems[k][j]=i;
    }
  multisectionProxy=CProxySection_multisectiontest_grp(NUMGROUPS,m->IDs,elems,nelems);
  delete m;
  CkCallback cb(CkIndex_multisectiontest_master::doneSetup(NULL), 0,masterGroup);
  contribute(sizeof(int),&msgCount, CkReduction::sum_int,cb);  

}

void multisectiontest_array1d::recv(multisectiontest_msg *msg)
{
  // when you have all your messages from the group partners
  // reply to them
  CkAssert(msg->side!=low);
  /*  if(iteration!=msg->iterationCount)
    {
    CkPrintf("A(%d)[%d] iteration %d != msg->iterationCount %d\n",CkGroupID(thisProxy.ckGetArrayID()).idx,thisIndex, iteration, msg->iterationCount);
    }
  */
  CkAssert(iteration==msg->iterationCount);

  msgCount++;
  //  CkPrintf("[%d] iteration %d received msg %d of %d\n",CkMyPe(),
  //  iteration,msgCount, sectionSize);
  //  CkPrintf("A(%d)[%d] iteration %d received msgCount %d of %d\n",CkGroupID(thisProxy.ckGetArrayID()).idx,thisIndex,iteration,msgCount, sectionSize*NUMARRAYS);
  delete msg;
  if(msgCount==sectionSize*NUMARRAYS)
    {
      //      CkPrintf("A(%d)[%d] iteration %d completed\n",CkGroupID(thisProxy.ckGetArrayID()).idx,thisIndex,iteration);
      multisectiontest_msg *omsg= new multisectiontest_msg;      
      omsg->side=low;
      omsg->iterationCount=iteration;
      multisectionProxy.recv(omsg);
      msgCount=0;
      ++iteration;
    }

}


void multisectiontest_array1d::recvSProxy(  multisection_proxymsg *m)
{
  CkAbort("don't do this\n");
  //  CkPrintf("A[%d] array received proxymsg\n",thisIndex);
  //multisectiontest_msg *msg= new multisectiontest_msg;
  //  m->gproxy.recv(msg);
  delete m;
}


MEGATEST_REGISTER_TEST(multisectiontest,"ebohm",1)
#include "multisectiontest.def.h"
