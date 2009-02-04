#include <stdio.h>
#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;


// DMK - DEBUG
#if (!(defined(CMK_CELL))) || (CMK_CELL == 0)
  void* malloc_aligned(int size, int align) { return malloc(size); }
  void free_aligned(void* p) { free(p); }
#endif


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
      arr[0].testAccelEntry(-1);
    };

    void done(void) {
      CkPrintf("All done\n");
      CkExit();
    };
};


class Hello : public CBase_Hello {

  public:
    int  s;
    int* a;

  public:

    Hello() {
      CkPrintf("Hello %d created\n",thisIndex);
      s = thisIndex;
      a = (int*)malloc_aligned(128, 128);  // Make sure 'a' is aligned for DMA to an SPE
      for (int i = 0; i < 4; i++) { a[i] = thisIndex + i; }
    }

    Hello(CkMigrateMessage *m) {}

    ~Hello() {
      if (a != NULL) { free_aligned(a); }
    }
  
    void testAccelEntry_callback() {

      // Print the current state of the member variables 's' and 'a'
      CkPrintf("Hello[%d]::testAccelEntry_callback() - Called... s:%d, a[0-3]:{ %d %d %d %d }...\n",
               thisIndex, s, a[0], a[1], a[2], a[3]
              );

      // Call the next array object's testAccelEntry method (or, if this is the last
      //   object, tell main we are done)
      if (thisIndex < nElements - 1) {
        thisProxy[thisIndex+1].testAccelEntry(thisIndex);
      } else {
        mainProxy.done();
      }
    }
};


#include "hello.def.h"
