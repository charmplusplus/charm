#include "eachtomany.decl.h"

#include <comlib.h>
#include <cassert>
#include <iostream>

/*
 * Test of EachToMany Strategy 
 */

CProxy_Main mainProxy;
CProxy_EachToManyArray eachToManyArrayProxy;

int nElements;

#define DEBUG 0
#define COMDEBUG 0 

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
  int totalIterations;
  int iter;
  
  /// The number of array elements that have reported their successful setup so far
  int nArrElemSetup;
  
public:

  Main(CkArgMsg *m) {    
    nDone = 0;
    totalIterations = 50;
    iter = 0;
    nArrElemSetup = 0;
    
    com_debug = COMDEBUG; // also below
     
    nElements=CkNumPes()*2;
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    if(m->argc >2 ) totalIterations=atoi(m->argv[2]);
    delete m;

    mainProxy = thishandle;
	
    // Create the array
    eachToManyArrayProxy = CProxy_EachToManyArray::ckNew(nElements);

    // Create a strategy
    CkArrayIndex *allElts = new CkArrayIndex[nElements];
    for (int i = 0; i<nElements; i++) {
    	CkArrayIndex1D index = i;
    	allElts[i] = index;
    }
    Strategy *strategy2 = new EachToManyMulticastStrategy(USE_DIRECT, eachToManyArrayProxy, eachToManyArrayProxy, nElements,allElts,nElements,allElts);
    stratEachToManyArray = ComlibRegister(strategy2);

    eachToManyArrayProxy.setup();
    
    delete [] allElts;
    
  } 

  void arraySetupDone(){
	   nArrElemSetup ++;
	   if(nArrElemSetup == nElements){
#if DEBUG
		   CkPrintf("[MainChare] Array setup completed. Starting test.\n");
#endif
		   eachToManyArrayProxy.TestEachToMany(iter);   
	   }
  }
  
  void doNext(){
	  nDone++;
	  // Becase each array element sends a message to all array elements,
	  // there are a total of nElements^2 messages received
	  // We do one comlib all to all and two other non-delegated ones as well. 
	 
#if DEBUG
	  CkPrintf("[MainChare] doNext() nDone=%d\n",nDone);
#endif
	  
	  const int nTotal = nElements*nElements;
	  if (nDone == nTotal){
		  int printIteration = 10;
		  int maxprints = 5;
		  while( (totalIterations / printIteration) > maxprints)
			  printIteration *= 2;
		  if(iter % printIteration == 0)
			  CkPrintf("[MainChare] Completed iteration %d\n",iter);
		  
		  
		  iter ++;
		  nDone = 0;
		  if(iter == totalIterations){
			  CkPrintf("[MainChare] Successful Completion of %d iterations\n", iter);
			  CkPrintf("Test Completed Successfully\n");
			  CkExit();
		  }
	  
		  // Do another iteration
		  int arrElt = 2;
		  int numPes = CkNumPes();
		  int destPe = iter%numPes;
		    
		  if( iter>4 && arrElt<nElements && destPe < CkNumPes() ){
#if DEBUG
			  CkPrintf("[%d] Decided to migrate elt %d to %d iter=%d numpes=%d\n", CkMyPe(), arrElt, destPe, iter, numPes);
#endif
			  eachToManyArrayProxy[arrElt].migrateMeTo(destPe);
		  }
		  
		  eachToManyArrayProxy.TestEachToMany(iter); 
	  
	  }
 
	    
  }
  
};

class EachToManyArray : public CBase_EachToManyArray {
private:
  CProxy_EachToManyArray localProxy;
  
public:

  EachToManyArray() {
	  com_debug = COMDEBUG; // also above
  }

  
  void setup(){
	  // We cannot put this in the constructor because the strategy may not yet have been created. Our references to it would therefore be invald.
	  localProxy = thisProxy;
	  ComlibAssociateProxy(stratEachToManyArray, localProxy); 
	  mainProxy.arraySetupDone();
  }
  
  void pup(PUP::er &p){
	  if(p.isUnpacking()){
		  localProxy = thisProxy;
		  ComlibAssociateProxy(stratEachToManyArray, localProxy);
#if DEBUG
		  CkPrintf("pup() associating proxy with comlib strategy instance\n");
#endif
	  }
  }
  
  
  EachToManyArray(CkMigrateMessage *m) {
#if DEBUG
	  CkPrintf("Object %d has migrated to %d\n", thisIndex, CkMyPe());
#endif
	  localProxy = thisProxy;
	  ComlibAssociateProxy(stratEachToManyArray, localProxy); 
  }
  
  void migrateMeTo(int toPe){
#if DEBUG
	  CkPrintf("==========================================\n\n[%d] object is calling migrateMe(toPe=%d)\n", CkMyPe(),toPe);
#endif 
	  migrateMe (toPe);
  }
  
    
  void TestEachToMany(int iter) {

	  // Build a message
	  char *msg = (char*)malloc(256);
	  sprintf(msg, "%d", thisIndex);
	  eachToManyMessage* b = new(strlen(msg)+1,0) eachToManyMessage;
	  memcpy(b->msg, msg, strlen(msg)+1);
	  b->length = strlen(msg);

	  // Broadcast the message to the whole array
//	  CkPrintf("[%d] Application: array elt=%d about to call ComlibBegin()\n", CkMyPe(), thisIndex);
	  ComlibBegin(localProxy, iter);
//	  CkPrintf("[%d] Application: array elt=%d broadcasting msg=%s\n", CkMyPe(), thisIndex, msg);
	  localProxy.receive(b);
	  ComlibEnd(localProxy, iter);
	  
	  free(msg);
  }

  void receive(eachToManyMessage* m) {
 
#if DEBUG
  //    CkPrintf("[%d] Application: received message %s\n", CkMyPe(), m->msg);
    int src;
    sscanf(m->msg,"%d",&src);
    CkPrintf("Application Received: %d to %d  [%d] \n", src, thisIndex, CkMyPe());
#endif           
    delete m;
    mainProxy.doNext();
  }
  
};

#include "eachtomany.def.h"
