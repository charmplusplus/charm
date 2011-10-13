#include "eachtomany.decl.h"

#include <comlib.h>
#include <cassert>
/*
 * Test of EachToMany Strategy using Hypercube Topology
 */

CProxy_Main mainProxy;
CProxy_EachToManyGroup eachToManyGroupProxy;
CProxy_EachToManyArray eachToManyArrayProxy;

int nElements;

//const static ComlibInstanceHandle stratEachToManyGroup=1;
const static ComlibInstanceHandle stratEachToManyArray=1;

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

    comm_debug = 1;
    nElements=CkNumPes();
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    delete m;

    mainProxy = thishandle;
	
    // create eachToMany strategy for Group
//    eachToManyGroupProxy = CProxy_EachToManyGroup::ckNew();
//
//    Strategy *strategy = 
//      new EachToManyMulticastStrategy(USE_DIRECT, 
//				      eachToManyGroupProxy, eachToManyGroupProxy
//				      /*,CkNumPes(), src, CkNumPes(), dst*/);
//    CkAssert(stratEachToManyGroup == ComlibRegister(strategy));
//    CkAssert(stratEachToManyGroup > 0);


    // create eachToMany strategy for Array
    
    CkArrayIndex *allElts = new CkArrayIndex[nElements];
    for (int i = 0; i<nElements; i++) {
    	allElts[i].data()[0] = i;
    	allElts[i].data()[1] = 0;
    	allElts[i].data()[2] = 0;
    }
 
    eachToManyArrayProxy = CProxy_EachToManyArray::ckNew(nElements);
    Strategy *strategy2 = new EachToManyMulticastStrategy(USE_DIRECT, eachToManyArrayProxy, eachToManyArrayProxy, nElements,allElts,nElements,allElts);
    CkAssert(stratEachToManyArray == ComlibRegister(strategy2));
    CkAssert(stratEachToManyArray > 0);
    
//    eachToManyGroupProxy.TestEachToMany();
    
    eachToManyArrayProxy.TestEachToMany(); 
    
    delete [] allElts;       
    
  }

  void finishGroupEachToMany() {
    nDone++;
    if (nDone == CkNumPes()) {
    	// The group each to many is done, now either quit or do the array based each to many 
    	
      nDone = 0;
      
#define DO_ARRAY_TEST  1 
#if DO_ARRAY_TEST 
      eachToManyArrayProxy.TestEachToMany(); 
#else 
      CkExit();
#endif
    }
  }
  
  void exit() {
    nDone++;
    CkPrintf("exit: %d have completed so far\n", nDone);
    
    if (nDone == nElements)
      CkExit();
  }

};

class EachToManyGroup : public CBase_EachToManyGroup {
private:
  CProxy_EachToManyGroup localProxy;
public:

  EachToManyGroup() {
    localProxy = thisProxy;
//    ComlibAssociateProxy(stratEachToManyGroup, localProxy);
  }

  EachToManyGroup(CkMigrateMessage *m) {}
    
  void TestEachToMany() {
	  CkPrintf("In Group TestEachToMany()\n");

	  char msg[] = "|This is a short each-to-many group message|";
	  eachToManyMessage* b = new(strlen(msg)) eachToManyMessage;

	  CkPrintf("************* [%d] Sending message ... \n", CkMyPe());

	  memcpy(b->msg, msg, strlen(msg)+1);
	  b->length = strlen(msg)+1;
	  ComlibBegin(localProxy);
	  localProxy.receive(b);
	  ComlibEnd(localProxy);
  }

  void receive(eachToManyMessage* m) {
    CkPrintf("Message: %s arrived at element %d\n", m->msg, CkMyPe());
    assert(strcmp(m->msg,"|This is a short each-to-many group message|") == 0);
    delete m;
    mainProxy.finishGroupEachToMany();
  }

};

class EachToManyArray : public CBase_EachToManyArray {
private:
  CProxy_EachToManyArray localProxy;
public:

  EachToManyArray() {}

  EachToManyArray(CkMigrateMessage *m) {}
    
  void TestEachToMany() {
	  CkPrintf("*****************  In Array TestEachToMany()\n");
	  
    localProxy = thisProxy;
    ComlibAssociateProxy(stratEachToManyArray, localProxy);
    char msg[] = "|This is a short each-to-many array message|";
    eachToManyMessage* b = new(strlen(msg)+1,0) eachToManyMessage;
	
    memcpy(b->msg, msg, strlen(msg)+1);
    b->length = strlen(msg);
    ComlibBegin(localProxy);
    localProxy.receive(b);
    ComlibEnd(localProxy);
  }

  void receive(eachToManyMessage* m) {
    CkPrintf("Message: %s arrived at element %d\n", m->msg, thisIndex);
    assert(strcmp(m->msg,"|This is a short each-to-many array message|") == 0);
    delete m;
    mainProxy.exit();
  }

};

#include "eachtomany.def.h"
