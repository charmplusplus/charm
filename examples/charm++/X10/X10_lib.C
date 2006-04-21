//-------------------------------------------------------------
// file   : X10_test2.C
// author : Isaac Dooley
// date   : April 2006
//

#include <pup.h>
#include <converse.h>
#include "X10_test.decl.h"
#include "X10_lib.h"


/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Places placesProxy;
/*readonly*/ int nPlaces;

class asyncMsg : public CMessage_asyncMsg{
 public:
  int temp;
  asyncMsg& operator=(const asyncMsg& obj) {
	temp = obj.temp;
    return *this;
  }

  asyncMsg(){temp=1;}

};


typedef CkVec<CkFutureID> *finishHandle;

void *beginFinish(){
  CkVec<CkFutureID> *FinishFutureList = new CkVec<CkFutureID>;
  return (void*)FinishFutureList;
} 

void endFinish(void* ffl){
  CkVec<CkFutureID> *FinishFutureList = (CkVec<CkFutureID> *)ffl;
  int last = FinishFutureList->length();
  CkPrintf("MainThread: Future waiting   last=%d\n", last);
  int len = FinishFutureList->length();
  while(len > 0){
	CkPrintf("len=%d\n", len);
	asyncMsg *msg = (asyncMsg *)CkWaitFuture((*FinishFutureList)[len-1]);
	FinishFutureList->remove(len-1);
	len = FinishFutureList->length();
	
  }

  CkPrintf("MainThread: Future awaken\n");
  delete FinishFutureList;
}

void asyncCall(void *ffl, int place, int whichFunction, void *packedParams){
  CkVec<CkFutureID> * FinishFutureList = (CkVec<CkFutureID> *)ffl;
  asyncMsg *msg = new asyncMsg;
  CkFutureID ftHandle = CkCreateAttachedFuture((void*)msg);
  CkAssert(FinishFutureList->size()==0);
  FinishFutureList->push_back(ftHandle);
  CkPrintf("MainThread: Created Future with handle %d\n", ftHandle);
  (*FinishFutureList)[FinishFutureList->length()]=ftHandle;
  placesProxy[place].startAsync(whichFunction,ftHandle,CkMyPe());
  CkPrintf("MainThread: Created Async call with handle %d\n", ftHandle);
}





/*mainchare*/
class Main : public CBase_Main
{
 public:

  Main(CkArgMsg* m)
  {
	nPlaces=CkNumPes();
	CkPrintf("Starting Up: %d Places(Processors)\n", nPlaces);  
	mainProxy = thishandle;

	CkAssert(nPlaces >= 2);

    // Create X10 Places	
 	placesProxy = CProxy_Places::ckNew(nPlaces);

	mainProxy.libThread();

  }
  
  void libThread(){
	CkPrintf("MainThread: executing in Main Chare\n");
	mainThread();	
	CkExit();
  }
  
};

/*mainchare*/
class Places : public CBase_Places
{
public:
  
  Places(CkMigrateMessage *m){}
  Places(){}
  
  void startAsync(int whichStatement, CkFutureID ftHandle, int pe_src){
	asyncMsg *msg = new asyncMsg;
	CkPrintf("Place %d: faking execution of statement %d\n", thisIndex, whichStatement);
	CkPrintf("Place %d: Finished work, sending result to Future [%d] \n", thisIndex, ftHandle);
	CkSendToFuture(ftHandle, (void *)msg, pe_src);
  }
  
};



  
#include "X10_test.def.h"  
