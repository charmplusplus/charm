#include <stdio.h>
#include "hello.decl.h"
#include <amp.h>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

/*mainchare*/
class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    //Process command-line arguments
    nElements=5;
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    delete m;

    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    mainProxy = thisProxy;

    CProxy_Hello arr = CProxy_Hello::ckNew(nElements);

    arr[0].SayHi(17);
  };

  void done(void)
  {
    CkPrintf("All done\n");
    CkExit();
  };
};

/*array [1D]*/
class Hello : public CBase_Hello
{
public:
  Hello()
  {
    CkPrintf("Hello %d created\n",thisIndex);
  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo)
  {
    CkPrintf("Hi[%d] from element %d\n",hiNo,thisIndex);
#if 1
    using namespace concurrency;
    std::vector<int> data(5);
    for (int count = 0; count < 5; count++)
    {
         data[count] = thisIndex + count;
    }

    array<int, 1> a(5, data.begin(), data.end());

    parallel_for_each(
        a.get_extent(),
        [=, &a](index<1> idx) restrict(amp)
        {
            a[idx] = a[idx] * 10;
        }
    );

    data = a;
    for (int i = 0; i < 5; i++)
    {
        CkPrintf("%d ", data[i]);
    }
    CkPrintf("\n");
#endif
    if (thisIndex < nElements-1)
      //Pass the hello on:
      thisProxy[thisIndex+1].SayHi(hiNo+1);
    else 
      //We've been around once-- we're done.
      mainProxy.done();
  }
};

#include "hello.def.h"
