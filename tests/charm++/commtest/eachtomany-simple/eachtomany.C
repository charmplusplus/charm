#include "eachtomany.decl.h"

#include <comlib.h>
#include <cassert>

/*
 * Test of EachToMany Strategy by performing one all-to-all and USE_DIRECT
 */

CProxy_Main mainProxy;
CProxy_EachToManyArray eachToManyArrayProxy;

int nElements;

ComlibInstanceHandle stratEachToManyArray;

class eachToManyMessage : public CMessage_eachToManyMessage{
public:
  int length;
  char* msg;
};

// mainchare

class Main : public CBase_Main
{
private:
  int nDone;
  int nPesDone;

public:

  Main(CkArgMsg *m) {    
    nDone = 0;
    nPesDone = 0;

    nElements=CkNumPes()*4;
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    delete m;

    mainProxy = thishandle;
	
    CkArrayIndex *allElts = new CkArrayIndex[nElements];
    for (int i = 0; i<nElements; i++) {
    	CkArrayIndex1D index = i;
    	allElts[i] = index;
    }
 
    // Create the array
    eachToManyArrayProxy = CProxy_EachToManyArray::ckNew(nElements);

    // Create a strategy
    Strategy *strategy2 = new EachToManyMulticastStrategy(USE_DIRECT, eachToManyArrayProxy, eachToManyArrayProxy, nElements, allElts, nElements, allElts);
    stratEachToManyArray = ComlibRegister(strategy2);
    
    eachToManyArrayProxy.TestEachToMany(); 
    
    delete [] allElts;
    
  } 

  void exit() {
	  nDone++;
	  // Becase each array element sends a message to all array elements, there are a total of nElements^2 messages received
	  if (nDone == nElements*nElements){
		  CkPrintf("Successful Completion\n");
		  CkExit();
	  }
  }
  
};

class EachToManyArray : public CBase_EachToManyArray {
private:
  CProxy_EachToManyArray localProxy;
public:

  EachToManyArray() {
  }

  EachToManyArray(CkMigrateMessage *m) {}
    
  void TestEachToMany() {

    localProxy = thisProxy;

    ComlibAssociateProxy(stratEachToManyArray, localProxy);
    //    CkPrintf("Array Element %d has associated proxy with comlib strategy %d\n", (int)thisIndex, (int)stratEachToManyArray);
    

    // Build a message
    char msg[] = "|This is a short each-to-many array message|";
    eachToManyMessage* b = new(strlen(msg)+1,0) eachToManyMessage;
    memcpy(b->msg, msg, strlen(msg)+1);
    b->length = strlen(msg);

    int iter = 1;

    //    CkPrintf("Array Element %d is about to call ComlibBegin\n", (int)thisIndex);
    ComlibBegin(localProxy, iter);
    //  CkPrintf("Array Element %d is about to broadcast to localProxy\n", (int)thisIndex);
    localProxy.receive(b);
    //CkPrintf("Array Element %d is about to call ComlibEnd\n", (int)thisIndex);
    ComlibEnd(localProxy, iter);

}


  void receive(eachToManyMessage* m) {
    assert(strcmp(m->msg,"|This is a short each-to-many array message|") == 0);
    delete m;
    mainProxy.exit();
  }
  
};

#include "eachtomany.def.h"
