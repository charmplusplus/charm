/**
Tiny test program that
  1.) Creates a simple 1D array
  2.) Asks the array to do a multicast operation--
      each element is told what to send, and 
      what it should expect to receive.
  3.) Synchronize, and go back to 2.

If SKIP_COMLIB is not set, uses the comlib
for the multicast operation.

Orion Sky Lawlor, olawlor@acm.org, 2003/7/15
Migration test in commlib added on 2004/05/12, Sameer Kumar
*/

#include <stdio.h>
#include "EachToManyMulticastStrategy.h" /* for ComlibManager Strategy*/

#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

/*mainchare*/
class Main : public CBase_Main
{
  int **commMatrix; /* commMatrix[s][r]= # messages sent from s to r */
  int *commSend, *commRecv; /* temporary storage for "send" */
  CProxy_Hello arr;
  int nIter; // Iterations remaining
public:
  Main(CkArgMsg* m)
  {
    //Process command-line arguments
    nElements=5;
    int strat=USE_MESH;
    if(m->argc>1) nElements=atoi(m->argv[1]);
    if (m->argc>2) strat=atoi(m->argv[2]); /* FIXME: use "+strategy" flag */
    delete m;
    
    // For the first step, use an all-to-all communication pattern.
    commMatrix=new int*[nElements];
    for (int i=0;i<nElements;i++) {
      commMatrix[i]=new int[nElements];
      for (int j=0;j<nElements;j++)
        commMatrix[i][j]=1;
    }
    commSend=new int[nElements]; 
    commRecv=new int[nElements];

    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements: strategy %d\n",
	     CkNumPes(),nElements, strat);
    mainProxy = thishandle;
    
    ComlibInstanceHandle cinst=CkGetComlibInstance();

    arr = CProxy_Hello::ckNew();
   
    EachToManyMulticastStrategy *strategy = new EachToManyMulticastStrategy
        (strat, arr,arr);
    cinst.setStrategy(strategy);
    
    CProxy_Hello hproxy(arr);
    for(int count = 0; count < nElements; count++)
        hproxy[count].insert(cinst);

    hproxy.doneInserting();

    nIter=0;
    send();
  };
  
  // Tell each element what to send and receive:
  void send(void) {
    CkPrintf("-------- starting iteration %d ---------\n",nIter);
    for (int me=0;me<nElements;me++) {
      for (int you=0;you<nElements;you++) {
          commSend[you]=commMatrix[me][you]; //Stuff I send to you
          commRecv[you]=commMatrix[you][me]; //Stuff you send to me
      }
      arr[me].startMcast(nIter,commSend,commRecv);
    }
  }

  // This multicast iteration is complete.
  void done(void)
  {
    CkPrintf("-------- finished iteration %d ---------\n",nIter);
    nIter++;
    if (nIter == 10) {
      CkPrintf("All done\n");
      CkExit();
    }
    else {
      reset();
      send();
    }
  }
  
  // Reset the send/recv matrix to random values
  void reset(void) {
    for (int i=0;i<nElements;i++)
      for (int j=0;j<nElements;j++)
          commMatrix[i][j]=(rand()%3);
  }
};

/*array [1D]*/
class Hello : public CBase_Hello 
{
  int curIter; // Current iteration number (only one can be running at a time)
  int *willRecv; // counts # of message we will recv, per source
  int *haveRecv; // counts # of messages we have recv'd, per source
  bool verbose; // generate debugging output for every start/send/recv/end.
  ComlibInstanceHandle comlib;
  CProxy_Hello hProxy; // delegated comlib proxy.
  
  void reset(void) {
    int i;
    for (i=0;i<nElements;i++) haveRecv[i]=0; // Haven't got any yet
    for (i=0;i<nElements;i++) willRecv[i]=-1; // Don't know how many we will get
  }
  // Call endMcast if we have received everything for this iteration.
  void tryEnd(void) {
    for (int i=0;i<nElements;i++) 
      if (willRecv[i]!=haveRecv[i])
        return;
    endMcast();
  }
  // Verify that we don't have too many messages from "src".
  void checkOver(int src) {
    if (willRecv[src]!=-1 && haveRecv[src]>willRecv[src]) {
      CkError("Element %d received too many messages from %d (expected %d, got %d)\n",
      	thisIndex, src, willRecv[src], haveRecv[src]);
      CkAbort("Too many multicast messages!\n");
    }
  }
public:
  Hello(ComlibInstanceHandle comlib_)       
  {
    comlib = comlib_;

    verbose=false; // true;
    if (verbose) CkPrintf("Element %d created\n",thisIndex);
    willRecv=new int[nElements];
    haveRecv=new int[nElements];
    reset();
    curIter=0;
    hProxy=thisProxy;
#ifndef SKIP_COMLIB
    ComlibDelegateProxy(&hProxy);
#endif
  }

  Hello(CkMigrateMessage *m) { 
  }
  
  // Send out the number of messages listed in "send" for each element.
  //  You'll receive from each element the number of messages listed in "recv".
  // This routine is called by main, which knows both senders and receivers.
  void startMcast(int nIter,const int *send,const int *recv)
  {
    if (curIter!=nIter) {
      CkError("Element %d asked to start iter %d, but we're at %d\n",
           thisIndex, nIter,curIter);
      CkAbort("Unexpected iteration start message!\n");
    }
    
    if(verbose) CkPrintf("[%d] Element %d iteration %d starting\n",CkMyPe(), thisIndex,curIter);
    
    comlib.beginIteration();
    for (int dest=0;dest<nElements;dest++) {
      for (int m=0;m<send[dest];m++) {
          if(verbose) CkPrintf("Element %d iteration %d send to %d\n",thisIndex,curIter,dest);
          hProxy[dest].midMcast(curIter,thisIndex);
      }
    }
    comlib.endIteration();
    
    for (int src=0;src<nElements;src++) {
    	willRecv[src]=recv[src]; 
	checkOver(src);
    }
    tryEnd();
  }
  
  // Receive a multicast from this array element.
  void midMcast(int nIter,int src) {
    if (curIter!=nIter) {
      CkError("Element %d received unexpected message from %d for iter %d (we're at %d)\n",
           thisIndex,src, nIter,curIter);
      CkAbort("Unexpected mcast message!\n");
    }
    if (verbose) CkPrintf("Element %d iteration %d recv from %d\n",thisIndex,curIter,src);
    haveRecv[src]++; checkOver(src);
    tryEnd();
  }
  
  // This is the end of one multicast iteration.
  //  Update state and contribute to mainchare reduction.
  void endMcast(void) {
    if (verbose) CkPrintf("Element %d iteration %d done\n",thisIndex,curIter);
    curIter++;
    reset();
    contribute(0,0,CkReduction::sum_int,CkCallback(
    	CkIndex_Main::done(),mainProxy));

    srandom((CkMyPe()+1) * thisIndex);
    int dest_proc = random() % CkNumPes();

    if(curIter > 4) {
        //if (verbose) 
        if(verbose) CkPrintf("[%d] Migrating to %d\n", CkMyPe(), dest_proc);
        migrateMe(dest_proc);
    }
  }

  void pup(PUP::er &p) {
      p | comlib;
      p | verbose;

      p | hProxy;
      p | curIter;
      p | verbose;        

      if(p.isUnpacking()) {
          willRecv = new int[nElements];
          haveRecv = new int[nElements];
      }
      
      p(willRecv, nElements);
      p(haveRecv, nElements);        
  }
};

#include "hello.def.h"
