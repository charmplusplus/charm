
#include <array>

#include "std_array.decl.h"

/*readonly*/ CProxy_Main mainProxy;

static const std::array<int, 2> array_2{ 1, 2 };
static const std::array<int, 3> array_3{ 3, 4, 5 };

struct Main : public CBase_Main
{
  Main(CkArgMsg* m)
  {
    mainProxy = thisProxy;
    CProxy_std_array::ckNew();
  };

  void done()
  {
    CkExit();
  }
};

struct std_array : public CBase_std_array
{
  std_array()
  {
    thisProxy.Test1(array_2);
    thisProxy.Test2(array_2);

    const auto a = thisProxy.Test3();
    CkAssert(a[0] == array_3[0]);
    CkAssert(a[1] == array_3[1]);
    CkAssert(a[2] == array_3[2]);

    ckout << "Test #3 passed" << endl;

    mainProxy.done();
  }

  void Test1(std::array<int, 2> a)
  {
    CkAssert(a[0] == array_2[0]);
    CkAssert(a[1] == array_2[1]);

    ckout << "Test #1 passed" << endl;
  }

  void Test2(const std::array<int, 2> & a)
  {
    CkAssert(a[0] == array_2[0]);
    CkAssert(a[1] == array_2[1]);

    ckout << "Test #2 passed" << endl;
  }

  std::array<int, 3> Test3()
  {
    return array_3;
  }
};

#include "std_array.def.h"
