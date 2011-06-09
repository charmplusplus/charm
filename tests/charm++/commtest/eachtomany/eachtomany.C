
#include <comlib.h>
#include <cassert>
#include <iostream>

#include "eachtomany.decl.h"


/*
 * Test of EachToMany Strategy
 * 
 * We perform multiple tests. Each test uses its own array and strategy instance. 
 * All arrays are the same size.
 * 
 * Tests:
 *    1) Migrate a single element around the ring of PEs
 *    2) Different source and destination arrays
 *    3) Migrating objects in various ways, leaving some PEs with no objects
 *   
 *    4) Random Migrations of source and destination arrays <disabled>
 * 
 *    5) Random Migrations same source and destination <unimplemented so far>
 * 
 */

CProxy_Main mainProxy;

CProxy_EachToManyArray testProxy1;
CProxy_EachToManyArray testProxy2src;
CProxy_EachToManyArray testProxy2dest;
CProxy_EachToManyArray testProxy3;
CProxy_EachToManyArray testProxy4src;
CProxy_EachToManyArray testProxy4dest;

ComlibInstanceHandle testStrategy1;
ComlibInstanceHandle testStrategy2;
ComlibInstanceHandle testStrategy3;
ComlibInstanceHandle testStrategy4;


#define DEBUG 0
#define COMMDEBUG 0

class eachToManyMessage : public CMessage_eachToManyMessage{
public:
  int length;
  char* msg;
};

class Main : public CBase_Main
{
private:

  int nDone1;
  int nDone2;
  int nDone3;
  int nDone4;
  int totalIterations;
  int iter1;
  int iter2;
  int iter3;
  int iter4;
  int numCompletedTests;
  int numtests;
  int numsetup;
  int nElements;
  
  /// The number of array elements that have reported their successful setup so far
  int nArrElemSetup;
  

  char *test3Counts;

public:

  Main(CkArgMsg *m) {    
    nDone1 = 0;
    nDone2 = 0;
    nDone3 = 0;
    nDone4 = 0;
    numCompletedTests = 0;
    totalIterations = 20;
    iter1 = 0;
    iter2 = 0;
    iter3 = 0;
    iter4 = 0;
    nArrElemSetup = 0;    
    nElements = 20;
    mainProxy = thishandle;
    numsetup=0;
    numtests=0;
    
    srand(1);
    
    com_debug = COMMDEBUG; // also below
	
    // Create the arrays
    testProxy1 = CProxy_EachToManyArray::ckNew(nElements);
    testProxy2src = CProxy_EachToManyArray::ckNew(nElements);
    testProxy2dest = CProxy_EachToManyArray::ckNew(nElements);
    testProxy3 = CProxy_EachToManyArray::ckNew(nElements);
    testProxy4src = CProxy_EachToManyArray::ckNew(nElements);
    testProxy4dest = CProxy_EachToManyArray::ckNew(nElements);
    
    
    CkArrayIndex *allElts = new CkArrayIndex[nElements];
    for (int i = 0; i<nElements; i++) {
    	CkArrayIndex1D index = i;
    	allElts[i] = index;
    }
    
    Strategy *strategy;
    
    strategy = new EachToManyMulticastStrategy(USE_DIRECT, testProxy1, testProxy1, nElements,allElts,nElements,allElts);
    testStrategy1 = ComlibRegister(strategy);
    
    strategy = new EachToManyMulticastStrategy(USE_DIRECT, testProxy2src, testProxy2dest, nElements,allElts,nElements,allElts);
    testStrategy2 = ComlibRegister(strategy);
    
    strategy = new EachToManyMulticastStrategy(USE_DIRECT, testProxy3, testProxy3, nElements,allElts,nElements,allElts);
    testStrategy3 = ComlibRegister(strategy);

    strategy = new EachToManyMulticastStrategy(USE_DIRECT, testProxy4src, testProxy4dest, nElements,allElts,nElements,allElts);
    testStrategy4 = ComlibRegister(strategy);
    
    delete [] allElts;
 
    test3Counts = new char[nElements*nElements];
    for(int i=0;i<nElements*nElements;i++)
      test3Counts[i] = 0;

    testProxy1.setup(testStrategy1);     numsetup++;
    testProxy2src.setup(testStrategy2);  numsetup++;
    testProxy2dest.setup(testStrategy2); numsetup++;
    testProxy3.setup(testStrategy3);     numsetup++;
    testProxy4src.setup(testStrategy4);  numsetup++;
    testProxy4dest.setup(testStrategy4); numsetup++;
      
  } 

  void arraySetupDone(){
	   nArrElemSetup ++;
	   int numElementsToSetup = nElements*numsetup;
#if DEBUG
	   CkPrintf("Setup %d/%d\n", nArrElemSetup, numElementsToSetup);
#endif
	   if(nArrElemSetup == numElementsToSetup){

#if QDSTARTUP
	     CkCallback *cbstart = new CkCallback(CkIndex_Main::startTests(), thisProxy);
	     CkStartQD(*cbstart);
#else
	     startTests();
#endif

	   }
  }
   

  void startTests(){
    numtests = 0;
    testProxy1.test1(iter1); CkPrintf("Running test 1\n"); numtests++;
    testProxy2src.test2(iter2); CkPrintf("Running test 2\n"); numtests++;
    testProxy3.test3(iter3);  CkPrintf("Running test 3\n"); numtests++;
    testProxy4src.test4(iter4); CkPrintf("Running test 4\n"); numtests++;
  }

  
  void testComplete(int whichTest){
	  CkPrintf("Test #%d completed successfully\n", whichTest, numtests);
	  fflush(stdout);

	  numCompletedTests ++;
	   
	  if(numCompletedTests == numtests)
		  CkExit();

  }
  
  
  void next1(){
	  nDone1++;
	  
#if DEBUG
	  CkPrintf("[MainChare] Test 1 doNext()  nDone=%d\n", nDone1); 
	  fflush(stdout);
#endif
	  
	  // Becase each array element sends a message to all array elements, there are a total of nElements^2 messages received
	  const int nTotal = nElements*nElements;
	  if (nDone1 == nTotal){
#if DEBUG
	    CkPrintf("[MainChare] Test 1 Completed iteration %d\n",iter1);
	    fflush(stdout);
#endif
		  iter1 ++;
		  nDone1 = 0;
		  if(iter1 == totalIterations){
			  testComplete(1);
		  } else {

			  // Do another iteration
			  int arrElt = 2;
			  int numPes = CkNumPes();
			  int destPe = iter1%numPes;

			  if( iter1>4 && arrElt<nElements && destPe < CkNumPes() ){
#if DEBUG
				  CkPrintf("[%d] Test 1 Decided to migrate elt %d to %d iter=%d numpes=%d\n", CkMyPe(), arrElt, destPe, iter1, numPes);
#endif
				  testProxy1[arrElt].migrateMeTo(destPe);
			  } 
			  testProxy1.test1(iter1);

		  }
	  }    
  }
  
  
  void next2(){
	  nDone2++;

	  // Becase each array element sends a message to all array elements, there are a total of nElements^2 messages received
	  const int nTotal = nElements*nElements;
	  if (nDone2 == nTotal){
#if DEBUG
		  CkPrintf("[MainChare] Test 2 Completed iteration %d\n",iter2);
#endif
		  iter2 ++;
		  nDone2 = 0;
		  if(iter2 == totalIterations){
			  testComplete(2);
		  } else {
			  // Do another iteration
			  testProxy2src.test2(iter2);
		  }
	  }    
  }


  void next3(int srcElt, int srcPE, int destElt, int destPE){ 
	  nDone3++; 
#if DEBUG
	  CkPrintf("[MainChare] next3 iter=%d srcElt=%d srcPe=%d destElt=%d destPe=%d nDone3=%d\n", iter3, srcElt, srcPE,destElt, destPE, nDone3);
	  fflush(stdout);
#endif

	  if(test3Counts[nElements*destElt+srcElt]!=0){
	    CkPrintf("next3 received more than one message for destElt=%d srcElt=%d\n", destElt, srcElt);
	    CkAbort("test3Counts[nElements*destElt+srcElt]!=0");
	  }
	  test3Counts[nElements*destElt+srcElt] ++;
	  
	  // Becase each array element sends a message to all array elements, there are a total of nElements^2 messages received 
	  const int nTotal = nElements*nElements; 
	  if (nDone3 == nTotal){ 
#if DEBUG
		  CkPrintf("[MainChare] Test 3 Completed iteration %d\n",iter3); 
	  fflush(stdout);
#endif

	  
	  for(int i=0;i<nElements*nElements;i++)
	    test3Counts[i] = 0;


		  iter3 ++; 
		  nDone3 = 0; 
		  if(iter3 == totalIterations){ 
			  testComplete(3); 
		  } else { 
			  // Do another iteration 

			  int numPes = CkNumPes();
			  int destPe;

#if 0
			  // Have all objects migrate to PE 1
			  destPe = 1;
			  if( iter3==3 && numPes>1 && destPe<numPes){ 
				  for(int e=0; e<nElements; e++){
#if DEBUG
					  CkPrintf("[%d] Test 3 Decided to migrate elt %d to %d iter=%d numpes=%d\n", CkMyPe(), e, destPe, iter3, numPes); 
					  fflush(stdout);
#endif
					  testProxy3[e].migrateMeTo(destPe); 
				  }
			  }	

			  // Have all objects migrate to PE 2
			  destPe = 2;
			  if( iter3==6 && numPes>1 && destPe<numPes){ 
				  for(int e=0; e<nElements; e++){
#if DEBUG
					  CkPrintf("[%d] Test 3 Decided to migrate elt %d to %d iter=%d numpes=%d\n", CkMyPe(), e, destPe, iter3, numPes); 
					  fflush(stdout);
#endif
					  testProxy3[e].migrateMeTo(destPe); 
				  }
			  }

			  // Have all objects migrate to PE 0
			  destPe = 0;
			  if( iter3==9 && numPes>1 && destPe<numPes){ 
				  for(int e=0; e<nElements; e++){
#if DEBUG
					  CkPrintf("[%d] Test 3 Decided to migrate elt %d to %d iter=%d numpes=%d\n", CkMyPe(), e, destPe, iter3, numPes); 
					  fflush(stdout);
#endif
					  testProxy3[e].migrateMeTo(destPe); 
				  }
			  }
			  

			  // Migrate back to all PEs
			  if( iter3==10){ 
				  for(int e=0; e<nElements; e++){
					  destPe= (e+3) % CkNumPes();
					  if(destPE < CkNumPes()){
#if DEBUG
					    CkPrintf("[%d] Test 3 Decided to migrate elt %d to %d iter=%d numpes=%d\n", CkMyPe(), e, destPe, iter3, numPes); 
					    fflush(stdout);
#endif
					    
					    testProxy3[e].migrateMeTo(destPe); 
					  }
				  }
			  }

#endif
			  
			  
			  // Migrate half to PE1 and half to PE2
			  if( iter3==12 && numPes>2){ 
				  for(int e=0; e<nElements; e++){
					  destPe= e % 2 + 1;
#if DEBUG
					  CkPrintf("[%d] Test 3 Decided to migrate elt %d to %d iter=%d numpes=%d\n", CkMyPe(), e, destPe, iter3, numPes); 
					  fflush(stdout);
#endif
					  testProxy3[e].migrateMeTo(destPe); 
				  }
			  }


			  // Migrate half to PE0 and half to PE1
			  if( iter3==14 && numPes>2){ 
				  for(int e=0; e<nElements; e++){
					  destPe= e % 2;
					  
					  if(destPE < CkNumPes()){

#if DEBUG
					    CkPrintf("[%d] Test 3 Decided to migrate elt %d to %d iter=%d numpes=%d\n", CkMyPe(), e, destPe, iter3, numPes); 
					    fflush(stdout);
#endif
					    testProxy3[e].migrateMeTo(destPe); 
					  }
				  }
			  }


			  	
			  // Migrate back to all PEs
			  if( iter3==16 ){ 
				  for(int e=0; e<nElements; e++){
					  destPe= e % CkNumPes();
					  if(destPE < CkNumPes()){

#if DEBUG
					    CkPrintf("[%d] Test 3 Decided to migrate elt %d to %d iter=%d numpes=%d\n", CkMyPe(), e, destPe, iter3, numPes); 
					    fflush(stdout);
#endif
					    testProxy3[e].migrateMeTo(destPe); 
					  }
				  }
			  }

#if DEBUG					  	
			  CkPrintf("[%d] Test 3 continuing on to iteration %d\n", CkMyPe(), iter3); 
			  fflush(stdout);
#endif
			  testProxy3.test3(iter3); 
		  } 
	  }     
  } 
  
  
  
  
  
  
  void next4(){
	  nDone4++;
//	  CkPrintf("[MainChare] Test 4 nDone4=%d\n",nDone4);

	  // Becase each array element sends a message to all array elements, there are a total of nElements^2 messages received
	  const int nTotal = nElements*nElements;
	  if (nDone4 == nTotal){
		  ComlibPrintf("[MainChare] Test 4 Completed iteration %d\n",iter4);

		  iter4 ++;
		  nDone4 = 0;
		  if(iter4 == totalIterations){
			  testComplete(4);
		  } else {
			  // Do another iteration

			  int numToMigrate = 4;
			  int numPes = CkNumPes();
			  
			  // Migrate back to all PEs
			  for(int i=0;i<numToMigrate;i++){
				  int e = rand() % nElements;
				  int destPe= rand() % numPes;
				  
//				  CkPrintf("[%d] Test 4 randomly decided to migrate src elt %d to %d iter=%d\n", CkMyPe(), e, destPe, iter4); 
//				  testProxy4src[e].migrateMeTo(destPe); 

//				  int e2 = rand() % nElements;
//				  int destPe2= rand() % numPes;
//
//				  CkPrintf("[%d] Test 4 randomly decided to migrate dest elt %d to %d iter=%d\n", CkMyPe(), e2, destPe2, iter4); 
//				  testProxy4dest[e2].migrateMeTo(destPe2); 

				  
			  }


			  testProxy4src.test4(iter4);
		  }
	  }    
  }

  
  
  
  
  
};



class EachToManyArray : public CBase_EachToManyArray {
private:
  CProxy_EachToManyArray myDelegatedProxy;
  ComlibInstanceHandle cinst;
  
public:

  EachToManyArray() {
	  com_debug = COMMDEBUG; // also above
  }

  void pup(PUP::er &p){
	  CBase_EachToManyArray::pup(p);
	  p | cinst;
	  if(p.isUnpacking()){
		  myDelegatedProxy = thisProxy;
		  ComlibAssociateProxy(cinst, myDelegatedProxy);
	  }
  }

  
  EachToManyArray(CkMigrateMessage *m) {
#if DEBUG
	  CkPrintf("Object %d has migrated to %d\n", thisIndex, CkMyPe());
	  fflush(stdout);
#endif
  }
  
  void setup(ComlibInstanceHandle inst){
	  // We cannot put this in the constructor because the strategy may not yet have been created. Our references to it would therefore be invald.
	  cinst = inst;
	  myDelegatedProxy = thisProxy;
	  ComlibAssociateProxy(cinst, myDelegatedProxy); 
	  mainProxy.arraySetupDone();
  }
  
  
  void migrateMeTo(int toPe){
	  migrateMe (toPe);
  }
  
    
  void test1(int iter) {
#if DEBUG
    CkPrintf("[%d] test1 iter=%d thisIndex=%d\n", CkMyPe(), iter, thisIndex);
    fflush(stdout);
#endif
    
    // Build a message
    char *msg = (char*)malloc(256);
    sprintf(msg, "%d %d %d", thisIndex, CkMyPe(), iter); 
    eachToManyMessage* b = new(strlen(msg)+1,0) eachToManyMessage;
    memcpy(b->msg, msg, strlen(msg)+1);
    b->length = strlen(msg);
    // Broadcast the message to the whole array
    ComlibBegin(myDelegatedProxy, iter);
    myDelegatedProxy.receive1(b);
    ComlibEnd(myDelegatedProxy, iter);
    // Free the message
    free(msg);
  }

   
  void test2(int iter) {
	  // Build a message
	  char *msg = (char*)malloc(256);
	  sprintf(msg, "%d", thisIndex);
	  eachToManyMessage* b = new(strlen(msg)+1,0) eachToManyMessage;
	  memcpy(b->msg, msg, strlen(msg)+1);
	  b->length = strlen(msg);
	  // Broadcast the message to the whole array
	  ComlibBegin(myDelegatedProxy, iter);
	  myDelegatedProxy.receive2(b);
	  ComlibEnd(myDelegatedProxy, iter);
	  // Free the message
	  free(msg);
  }


  void test3(int iter) { 
#if DEBUG
    CkPrintf("[%d] test3 iter=%d thisIndex=%d\n", CkMyPe(), iter, thisIndex);
    fflush(stdout);
#endif

	  // Build a message 
	  char *msg = (char*)malloc(256); 
	  sprintf(msg, "%d %d %d", thisIndex, CkMyPe(), iter); 
	  eachToManyMessage* b = new(strlen(msg)+1,0) eachToManyMessage; 
	  memcpy(b->msg, msg, strlen(msg)+1); 
	  b->length = strlen(msg); 
	  // Broadcast the message to the whole array 
	  ComlibBegin(myDelegatedProxy, iter); 
	  myDelegatedProxy.receive3(b); 
	  ComlibEnd(myDelegatedProxy, iter); 
	  // Free the message 
	  free(msg); 
  } 
 
  void test4(int iter) {
 	  // Build a message
 	  char *msg = (char*)malloc(256);
 	  sprintf(msg, "%d", thisIndex);
 	  eachToManyMessage* b = new(strlen(msg)+1,0) eachToManyMessage;
 	  memcpy(b->msg, msg, strlen(msg)+1);
 	  b->length = strlen(msg);
 	  // Broadcast the message to the whole array
 	  ComlibBegin(myDelegatedProxy, iter);
 	  myDelegatedProxy.receive4(b);
 	  ComlibEnd(myDelegatedProxy, iter);
 	  // Free the message
 	  free(msg);
   }


  
  void receive1(eachToManyMessage* m) {      
    int srcElt, srcPE, destPE, iter;
    sscanf(m->msg,"%d %d %d",&srcElt, &srcPE, &iter);
    delete m; 
#if DEBUG
    CkPrintf("[%d] receive1 from srcElt=%d srcPE=%d iter=%d thisElt=%d\n", CkMyPe(), srcElt, srcPE, iter, thisIndex);
    fflush(stdout);
#endif
    mainProxy.next1();
  }
  
  

  void receive2(eachToManyMessage* m) {
//#if DEBUG
//  //    CkPrintf("[%d] Application: received message %s\n", CkMyPe(), m->msg);
//    int src;
//    sscanf(m->msg,"%d",&src);
//    CkPrintf("Test 2 Received: %d to %d  [%d] \n", src, thisIndex, CkMyPe());
//#endif           
    delete m;
    mainProxy.next2();
  }
  

  void receive3(eachToManyMessage* m) { 
    int srcElt, srcPE, destPE, iter;
    sscanf(m->msg,"%d %d %d",&srcElt, &srcPE, &iter);
    delete m; 
    mainProxy.next3(srcElt, srcPE, (int)thisIndex, CkMyPe());
#if DEBUG
    if(iter==14){
      CkPrintf("[%d] receive3 from srcElt=%d srcPE=%d iter=%d thisElt=%d\n", CkMyPe(), srcElt, srcPE, iter, thisIndex);
      fflush(stdout);
    }
#endif
  } 
  
  
  void receive4(eachToManyMessage* m) {
    delete m;
//    CkPrintf("receive4\n");
    mainProxy.next4();
  }
  
  
};

#include "eachtomany.def.h"
