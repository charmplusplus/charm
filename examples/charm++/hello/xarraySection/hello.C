#include <stdio.h>
#include "hello.decl.h"
#include "ckmulticast.h"

/*readonly*/ CProxy_Main mainProxy;


class pingMsg: public CkMcastBaseMsg, public CMessage_pingMsg
{
    public:
        pingMsg(int num=0): hiNo(num) {}
        int hiNo;
};

/*mainchare*/
class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m): numArrays(2), numElements(5), sectionSize(numElements), maxIter(3), numReceived(0), numIter(0)
  {
    //Process command-line arguments
    if (m->argc > 1) numElements = atoi(m->argv[1]);
    if (m->argc > 2) numArrays   = atoi(m->argv[2]);
    if (m->argc > 3) sectionSize = atoi(m->argv[3]); else sectionSize = numElements;
    if (m->argc > 4) maxIter     = atoi(m->argv[4]);
    if (m->argc > 5)
    { CkPrintf("Syntax: %s [numElements numArrays sectionSize maxIter]",m->argv[0]); CkAbort("Wrong Usage\n"); }
    delete m;

    CkPrintf("Running a cross-array section demo.\n");
    CkPrintf("\tnum PEs = %d\n\tnum arrays = %d\n\tnum elements = %d\n\tsection size = %d\n\tnum iterations = %d\n",
             CkNumPes(), numArrays, numElements, sectionSize, maxIter);

    mainProxy = thisProxy;
    // Setup section member index bounds
    int afloor = 0, aceiling = sectionSize-1;

    // Allocate space
    CProxy_Hello *arrayOfArrays = new CProxy_Hello[numArrays];
    CkArrayID *arrID            = new CkArrayID[numArrays];
    int *nelems                 = new int[numArrays];
    CkArrayIndexMax **elems     = new CkArrayIndexMax*[numArrays];

    // Create a list of array section members
    for(int k=0; k < numArrays; k++)
    {
        // Create the array
        arrayOfArrays[k] = CProxy_Hello::ckNew(k, numElements);
        // Store the AID
        arrID[k]  = arrayOfArrays[k].ckGetArrayID();
        // Create a list of section member indices in this array
        nelems[k] = sectionSize;
        elems[k]  = new CkArrayIndexMax[sectionSize];
        for(int i=afloor,j=0; i <= aceiling; i++,j++)
            elems[k][j] = CkArrayIndex1D(i);
    }
    // Create the x-array-section
    sectionProxy = CProxySection_Hello(numArrays, arrID, elems, nelems);

    // Create a multicast manager group
    CkGroupID mcastMgrGID = CProxy_CkMulticastMgr::ckNew();
    CkMulticastMgr *mcastMgr = CProxy_CkMulticastMgr(mcastMgrGID).ckLocalBranch();
    // Delegate the section comm to the CkMulticast library
    sectionProxy.ckSectionDelegate(mcastMgr);

    // Start the test by pinging the section
    pingMsg *msg = new pingMsg(numIter);
    sectionProxy.SayHi(msg);
  };

  /// Test controller method
  void done(void)
  {
      if (++numReceived >= numArrays * sectionSize)
      {
          numReceived = 0;
          if (++numIter == maxIter) {
              CkPrintf("----------------- testController: All %d iterations done\n", numIter);
              CkExit();
          }
          else {
              // Ping the section
              CkPrintf("----------------- testController: Iteration %d done\n", numIter);
              pingMsg *msg = new pingMsg(numIter);
              sectionProxy.SayHi(msg);
          }
      }
  };

private:
  /// Input parameters
  int numArrays, numElements, sectionSize, maxIter;
  /// Counters
  int numReceived, numIter;
  /// The cross-array section proxy
  CProxySection_Hello sectionProxy;
};

/*array [1D]*/
class Hello : public CBase_Hello
{
public:
  Hello(int _aNum): aNum(_aNum)
  {
    CkPrintf("AID[%d] Hello %d created on pe %d\n", aNum, thisIndex, CkMyPe());
  }

  Hello(CkMigrateMessage *m) {}

  void SayHi(pingMsg *msg)
  {
    CkPrintf("AID[%d] Hi[%d] from element %d\n", aNum, msg->hiNo, thisIndex);
    mainProxy.done();
    delete msg;
  }
private:
  int aNum;
};

#include "hello.def.h"

