#include "charm++.h"
#include "startupTest.decl.h"
#include <stdio.h>
#include <math.h>

#include "startupTest.h"

int intOne;
int arrSize;
double dOne;
double dTwo;
double WasteUnits;
	
CkVec <int> IntArrOne;
CkVec <int> IntArrTwo;
CkVec <int> IntArrThree;
CkVec <int> IntArrFour;
CkVec <int> IntArrFive;
CkHashtableT<intdual, int> mapSix;
CkVec<CProxy_groupTest> groupProxy;
CkVec<CProxy_groupTestX> groupProxyX;
CProxy_main mainProxy;		
CProxy_ReadArrZero zeroProxy;		
CProxy_ReadArrOne oneProxy;		
CProxy_ReadArrTwo twoProxy;		
CProxy_ReadArrThree threeProxy;		
CProxy_ReadArrFour fourProxy;		
CProxy_ReadArrFive fiveProxy;		
CProxy_ReadArrSix sixProxy;		
CProxy_ReadArrSeven sevenProxy;		


main::main(CkArgMsg *msg)
{
  int reported;
  int validateBoundOrder=0;
  if(msg->argc<4)
    CkAbort("Usage: startupTest arrSize1 arrSize2 WasteUnits validateBoundOrder\n Where arrsize is int >0 wasteunit is double >0 validateBoundOrder is 1 or 0\n");
  //get arrsize and wastetime from cmd line
  arrSize=atoi(msg->argv[1]);
  int arrSize2=atoi(msg->argv[2]);
  sscanf(msg->argv[3],"%lg",&WasteUnits);
  validateBoundOrder=atoi(msg->argv[4]);
  mainProxy=thishandle;
  if(arrSize<1 || arrSize2 < 1 || WasteUnits<=0 || (validateBoundOrder!=1 && validateBoundOrder!=0))
    CkAbort("Usage: startupTest arrSize1 arrSize2 WasteUnits validateBoundOrder\n Where arrsize is int >0 wasteunit is double >0 validateBoundOrder is 1 or 0\n");
  doneCount=0;
  //init readonly values
  CkPrintf("Nodes %d Cpus %d Using %d %d from %s %s %g from %s\n",CkNumNodes(), CkNumPes(),arrSize,arrSize2,msg->argv[1],msg->argv[2],WasteUnits, msg->argv[3]);
  intOne=1;
  dOne=1.0;
  dTwo=2.0;
  for(int i=0;i<arrSize;i++)
    {
      IntArrOne.push_back(i);
      IntArrTwo.push_back(i);
      IntArrThree.push_back(i);
      IntArrFour.push_back(i);
      IntArrFive.push_back(i);
    }
  for(int i=0;i<arrSize;i++)
    {
      groupProxy.push_back(CProxy_groupTest::ckNew(i));
    }
  for(int i=0;i<arrSize;i++)
    {
      groupProxyX.push_back(CProxy_groupTestX::ckNew(i));
    }
  //create zero by default map
  zeroProxy  = CProxy_ReadArrZero::ckNew();  
  for(int i=0;i<arrSize;i++)
    zeroProxy(i).insert(arrSize, WasteUnits);
  zeroProxy.doneInserting();

  // make our callbacks

  CkCallback cb[9];
  for(int i=0;i<7;i++)
    {
      cb[i]=CkCallback(CkIndex_ReadArrZero::receiveRed(NULL),CkArrayIndex1D(i),zeroProxy);
    }
  //create one-five by map then array

  CProxy_OneMap oneMap = CProxy_OneMap::ckNew(WasteUnits);
  CkArrayOptions arrOpts;
  arrOpts.setMap(oneMap);
  oneProxy  = CProxy_ReadArrOne::ckNew(arrSize, WasteUnits, cb[0],arrOpts);  
  for(int i=0;i<arrSize;i++)
    oneProxy(i).insert(arrSize, WasteUnits,cb[0]);
  oneProxy.doneInserting();

  CProxy_TwoMap twoMap = CProxy_TwoMap::ckNew(WasteUnits);
  arrOpts.setMap(twoMap);
  twoProxy  = CProxy_ReadArrTwo::ckNew(arrSize, WasteUnits, cb[1],arrOpts);  
  for(int i=0;i<arrSize;i++)
    twoProxy(i).insert(arrSize, WasteUnits,cb[1]);
  twoProxy.doneInserting();

  CProxy_ThreeMap threeMap = CProxy_ThreeMap::ckNew(WasteUnits);
  arrOpts.setMap(threeMap);
  threeProxy  = CProxy_ReadArrThree::ckNew(arrSize, WasteUnits, cb[2],arrOpts);  
  for(int i=0;i<arrSize;i++)
    threeProxy(i).insert(arrSize, WasteUnits,cb[2]);
  threeProxy.doneInserting();

  // make 4 new style
  CkArrayOptions arrOptsBulk(arrSize);
  CProxy_FourMap fourMap = CProxy_FourMap::ckNew(WasteUnits);
  arrOptsBulk.setMap(fourMap);
  fourProxy  = CProxy_ReadArrFour::ckNew(arrSize,WasteUnits,cb[3],arrOptsBulk);  
  /*  for(int i=0;i<arrSize;i++)
      fourProxy(i).insert(arrSize, WasteUnits);*/
  fourProxy.doneInserting();

  // make 5 a shadow of 4
  CkArrayOptions arrOptsBind4(arrSize);
  arrOptsBind4.bindTo(fourProxy);
  fiveProxy  = CProxy_ReadArrFive::ckNew(arrSize, WasteUnits, validateBoundOrder,cb[4],arrOptsBind4);  
  //  for(int i=0;i<arrSize;i++)
  //    fiveProxy(i).insert(arrSize, WasteUnits);
  fiveProxy.doneInserting();


  //#define SIX_INSERT
#ifdef SIX_INSERT
  // if you make six this way it may lose the race with seven 
  CProxy_SixMap sixMap = CProxy_SixMap::ckNew(WasteUnits);
  arrOpts.setMap(sixMap);
  sixProxy  = CProxy_ReadArrSix::ckNew(arrSize, arrSize2, WasteUnits,cb[5], arrOpts);  
  for(int i=0;i<arrSize;i++)
    for(int j=0;j<arrSize2;j++)
      sixProxy(i,j).insert(arrSize,arrSize2, WasteUnits,cb[5]);
  sixProxy.doneInserting();
#else
  // bulk build six
  CkArrayOptions arrOptsSix(arrSize,arrSize2);
  CProxy_SixMap sixMap = CProxy_SixMap::ckNew(WasteUnits);
  arrOptsSix.setMap(sixMap);
  sixProxy  = CProxy_ReadArrSix::ckNew(arrSize,arrSize2, WasteUnits, cb[5],arrOptsSix);  
  sixProxy.doneInserting();

#endif

  // make seven a shadow of six
  //#define SEVEN_INSERT
#ifdef SEVEN_INSERT

  CkArrayOptions arrOptsBind6;
  arrOptsBind6.bindTo(sixProxy);
  sevenProxy  = CProxy_ReadArrSeven::ckNew(arrSize, arrSize2,WasteUnits, validateBoundOrder,cb[6],arrOptsBind6);  
  for(int i=0;i<arrSize;i++)
    for(int j=0;j<arrSize2;j++)
      sevenProxy(i,j).insert(arrSize, arrSize2,WasteUnits,validateBoundOrder,cb[6]);
  sevenProxy.doneInserting();

#else

  CkArrayOptions arrOptsBind6(arrSize,arrSize2);
  arrOptsBind6.bindTo(sixProxy);
  sevenProxy  = CProxy_ReadArrSeven::ckNew(arrSize, arrSize2,WasteUnits, validateBoundOrder,cb[6],arrOptsBind6);  
  sevenProxy.doneInserting();
#endif  

  CheckAllReadOnly();
  CkPrintf("Setup Complete for arrSize %d WasteUnits %g\n",arrSize, WasteUnits);
  zeroProxy.dowork(); 
  oneProxy.dowork(); 
  twoProxy.dowork(); 
  threeProxy.dowork(); 
  fourProxy.dowork(); 
  fiveProxy.dowork(); 
  sixProxy.dowork(); 
  sevenProxy.dowork(); 
};

void main::createReport(CkReductionMsg *msg)
{
  //  int count=((int *) msg->getData())[0];
  int array=(int) msg->getUserFlag();
  delete msg;
  CkPrintf("Create Report for %d\n",array);
  /*
  switch (array){

  case 0 :  zeroProxy.dowork(); break;
  case 1 :  oneProxy.dowork(); break;
  case 2 :  twoProxy.dowork(); break;
  case 3 :  threeProxy.dowork(); break;
  case 4 :  fourProxy.dowork(); break;
  case 5 :  fiveProxy.dowork(); break;
  default: CkAbort("impossible user flag"); break;
  }
  */
};

void main::doneReport(CkReductionMsg *msg)
{
  //  int count=(int) msg->getData()[0];
  int array=(int) msg->getUserFlag();
  delete msg;
  CkPrintf("Done Report for %d\n",array);  
  doneCount++;
  if(doneCount==8)
    {
      CkPrintf("All Done %d\n",doneCount);  
      CkExit();
    }
};

int OneMap::procNum(int hdl, const CkArrayIndex &idx)
{
  CkArrayIndex1D idx1d = *(CkArrayIndex1D *) &idx;
  return(IntArrOne[idx1d.index[0]]%CkNumPes());
};

int TwoMap::procNum(int hdl, const CkArrayIndex &idx)
{
  CkArrayIndex1D idx1d = *(CkArrayIndex1D *) &idx;
  return(IntArrTwo[idx1d.index[0]]%CkNumPes());
};

int ThreeMap::procNum(int hdl, const CkArrayIndex &idx)
{
  CkArrayIndex1D idx1d = *(CkArrayIndex1D *) &idx;
  return(IntArrThree[idx1d.index[0]]%CkNumPes());
};

int FourMap::procNum(int hdl, const CkArrayIndex &idx)
{
  CkArrayIndex1D idx1d = *(CkArrayIndex1D *) &idx;
  return(IntArrFour[idx1d.index[0]]%CkNumPes());
};

int FiveMap::procNum(int hdl, const CkArrayIndex &idx)
{
  CkArrayIndex1D idx1d = *(CkArrayIndex1D *) &idx;
  int retval=IntArrFive[idx1d.index[0]]%CkNumPes();
  //  CkPrintf("procnum five %d %d\n",idx1d.index,retval);
  return(retval);
};


int SixMap::procNum(int hdl, const CkArrayIndex &idx)
{
  CkArrayIndex2D idx2d = *(CkArrayIndex2D *) &idx;
  int retval=(idx2d.index[0]+idx2d.index[1])%CkNumPes();
  //  CkPrintf("procnum Six %d %d %d\n",idx2d.index[0],idx2d.index[1],retval);
  return(retval);
};



void WasteTime(double howmuch)
{
  double start = CmiWallTimer();
  while (CmiWallTimer() - start < howmuch) ;
  
};


bool CheckAllReadOnly()
{

  CkAssert(intOne==1);
  CkAssert(dOne==1.0);
  CkAssert(dTwo==2.0);
  CkAssert(arrSize>0);
  for(int i=0;i<arrSize;i++)
    {
      CkAssert(IntArrOne[i]==i);
      CkAssert(IntArrTwo[i]==i);
      CkAssert(IntArrThree[i]==i);
      CkAssert(IntArrFour[i]==i);
      CkAssert(IntArrFive[i]==i);
    }
  return(true);
};



#include "startupTest.def.h"
