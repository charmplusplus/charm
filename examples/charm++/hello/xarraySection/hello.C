#include "hello.decl.h"
#include "ckmulticast.h"
#include <stdio.h>

/*readonly*/ CProxy_Main mainProxy;

struct pingMsg: public CkMcastBaseMsg, public CMessage_pingMsg
{
  ck::future<int> done;
  int hiNo;
  pingMsg(ck::future<int> f, int num): done(f), hiNo(num) {}
};


/// Test controller. Main chare
class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m): numArrays(2), numElements(5), sectionSize(numElements), maxIter(3)
  {
    // Save a proxy to main for use as a reduction target
    mainProxy = thisProxy;

    //Process command-line arguments
    if (m->argc > 1) numElements = atoi(m->argv[1]);
    if (m->argc > 2) numArrays   = atoi(m->argv[2]);
    if (m->argc > 3) sectionSize = atoi(m->argv[3]); else sectionSize = numElements;
    if (m->argc > 4) maxIter     = atoi(m->argv[4]);
    if (m->argc > 5)
    { CkPrintf("Syntax: %s [numElements numArrays sectionSize maxIter]",m->argv[0]); CkExit(1); }
    delete m;

    CkPrintf("Running a cross-array section demo.\n");
    CkPrintf("\tnum PEs = %d\n\tnum arrays = %d\n\tnum elements = %d\n\tsection size = %d\n\tnum iterations = %d\n",
             CkNumPes(), numArrays, numElements, sectionSize, maxIter);

    // Setup section member index bounds
    int afloor = 0, aceiling = sectionSize-1;

    // Allocate space
    std::vector<CkArrayID> arrID(numArrays);
    std::vector<std::vector<CkArrayIndex> > elems(numArrays);

    // Create a list of array section members
    for(int k=0; k < numArrays; k++)
    {
        // Create the array
        CProxy_Hello array = CProxy_Hello::ckNew(k, numElements);
        // Store the AID
        arrID[k]  = array.ckGetArrayID();
        // Create a list of section member indices in this array
        elems[k].resize(sectionSize);
        for(int i=afloor,j=0; i <= aceiling; i++,j++)
            elems[k][j] = CkArrayIndex1D(i);
    }
    // Create the x-array section, which is autodelegated to CkMulticast
    sectionProxy = CProxySection_Hello(arrID, elems);
    thisProxy.awaitPong();
  }

  /// Test controller method
  void awaitPong(void)
  {
    for (auto numIter = 0; numIter < maxIter; numIter++) {
      // Make a new ping message
      ck::future<int> redn;
      pingMsg *nextPing = new pingMsg(redn, numIter);
      // Then send it to the section
      sectionProxy.mcastPing(nextPing);
      // Get the value of, then release future
      auto data = redn.get();
      redn.release();
      // Print the message (and test the value)
      CkPrintf("----------------- testController:  Received pong (%d) via future for iter %d\n", data, numIter+1);
      CkAssert(data == (numIter * numArrays * sectionSize));
    }
    CkPrintf("----------------- testController: All %d iterations done\n", maxIter);
    CkExit();
  }

private:
  /// Input parameters
  int numArrays, numElements, sectionSize, maxIter;
  /// The cross-array section proxy
  CProxySection_Hello sectionProxy;
};



/** 1D chare array.
 *
 * Sections of multiple instances of this array participate in a cross-array section reduction
 */
class Hello : public CBase_Hello
{
public:
  Hello(int _aNum): aNum(_aNum)
  {
    CkPrintf("Array %d, Element %d created on PE %d\n", aNum, thisIndex, CkMyPe());
  }

  void mcastPing(pingMsg *msg)
  {
    // Say hello world
    CkPrintf("Array %d, Element %d received ping number %d\n", aNum, thisIndex, msg->hiNo);
    CkGetSectionInfo(sectionCookie, msg);
    // Contribute to the section reduction
    CProxySection_Hello::contribute(sizeof(int), &(msg->hiNo), CkReduction::sum_int, sectionCookie, CkCallback(msg->done.handle()));
    delete msg;
  }

private:
  int aNum;
  CkSectionInfo sectionCookie;
};

#include "hello.def.h"

