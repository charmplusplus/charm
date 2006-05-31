#include <stdio.h>
#include "hello.decl.h"
#include "hello_shared.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

/*mainchare*/

class Main : public CBase_Main {

  public:
    Main(CkArgMsg* m) {

      //Process command-line arguments
      nElements = 5;
      if (m->argc > 1) nElements = atoi(m->argv[1]);
      delete m;

      //Start the computation
      CkPrintf("Running Hello on %d processors for %d elements\n", CkNumPes(), nElements);
      mainProxy = thisProxy;

      CProxy_Hello arr = CProxy_Hello::ckNew(nElements);
      arr[0].SayHi(17);
    };

    void done(void) {
      CkPrintf("All done\n");
      CkExit();
    };
};


/*array [1D]*/
class Hello : public CBase_Hello {

  public:
    Hello() {
      CkPrintf("Hello %d created\n",thisIndex);
    }

    Hello(CkMigrateMessage *m) {}
  
    void SayHi(int hiNo) {

      ///// First, Test the Standard Method for Work Requests /////

      CkPrintf("[%d] BEFORE First\n", thisIndex);

      char buf[16] __attribute__((aligned(16)));
      sprintf(buf, "%d", thisIndex);
      sendWorkRequest(FUNC_SAYHI,
                      NULL, 0,            // readWrite data
                      buf, strlen(buf)+1, // readOnly data
                      NULL, 0,            // writeOnly data
                      CthSelf()
                     );
      CthSuspend();

      CkPrintf("[%d] AFTER First\n", thisIndex);


      ///// Second, Test the Scatter/Gather Method for Work Requests /////

      CkPrintf("[%d] BEFORE Second\n", thisIndex);

      // An array of all the strings to pass to the Offload API for processing
      char* strBufs[] = { "This is the first of the read only buffers.",
                          "Yet another read only buffer (the second one to be exact).",
                          "Oh man, not another read only buffer.  How many of these will there be.",
                          "How long is this madness going to go on?",
                          "Finally something other than a read only buffer.",
                          "Wait a sec, how many of these are there going to be?",
                          "Oh no, not again!",
                          "Ah a write only buffer.  At least this is the last set of buffers.",
                          "Please stop the madness!"
                        };

      // Copy each of the strings into a properly aligned buffer and create the DMA list that
      //   will be passed to the Offload API.
      DMAListEntry dmaList[9];
      for (int i = 0; i < 9; i++) {

        // Copy the string into a properly aligned buffer
        char* tmpPtr = (char*)malloc_aligned(strlen(strBufs[i]) + 1, 16);
        strcpy(tmpPtr, strBufs[i]);

        // Add the aligned buffer into the DMA list
        dmaList[i].size = strlen(strBufs[i]) + 1;
        dmaList[i].ea = (unsigned int)(tmpPtr);

        // Display the contents of the buffers before the work request is made
        CkPrintf("dmaList[%d].ea = 0x%08x  before:(\"%s\") (%d)\n",
                 i, dmaList[i].ea, (char*)(dmaList[i].ea), strlen((char*)(dmaList[i].ea))
                );
      }

      // Send the Work Request to the Offload API (scatter/gather work request type)
      sendWorkRequest_list(FUNC_STRBUFS,
                           0,
                           dmaList,
                           3, 3, 3,
                           CthSelf()
                          );
      // Goto sleep, the Offload API will wake this thread back up when the work request has
      //   finished.
      CthSuspend();

      // Display the contents of the buffers now that the work request is finished
      for (int i = 0; i < 9; i++) {
        CkPrintf("dmaList[%d].ea = 0x%08x   after:(\"%s\") (%d)\n",
                 i, dmaList[i].ea, (char*)(dmaList[i].ea), strlen((char*)(dmaList[i].ea))
                );
      }

      CkPrintf("[%d] AFTER Second\n", thisIndex);


      // Send a message onto the next element in the array (or to
      //   main when the last element is finished)
      if (thisIndex < nElements-1)
        thisProxy[thisIndex+1].SayHi(hiNo+1); // Pass on the hello
      else
        mainProxy.done();  // All have said hello, program is done
    }
};


#include "hello.def.h"
