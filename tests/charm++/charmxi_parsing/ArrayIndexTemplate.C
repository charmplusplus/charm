
#include "charm++.h"

template <int X>
struct MyIndex
{
  int dimensions[X];
};

template <int X>
struct CkArrayIndexMyIndex : public CkArrayIndex
{
  using custom_t = MyIndex<X>;

  CkArrayIndexMyIndex()
  {
    nInts = X;
    idx = new (index) custom_t();
  }
  CkArrayIndexMyIndex(const custom_t &i)
  {
    nInts = X;
    idx = new (index) custom_t(i);
  }

protected:
  custom_t *idx;
};

#include "ArrayIndexTemplate.decl.h"

/*readonly*/ CProxy_Main mainProxy;

class Main : public CBase_Main
{
  template <typename CProxy_MyArray, size_t N>
  void TestArray(const typename CProxy_MyArray::array_index_t::custom_t (&initializers)[N])
  {
    CProxy_MyArray arr = CProxy_MyArray::ckNew();
    for (const auto & index : initializers)
      arr[index].insert();
    arr.doneInserting();

    for (const auto & index : initializers)
      arr[index].Test();
  }

public:
  Main(CkArgMsg* m)
  {
    mainProxy = thisProxy;

    const MyIndex<1> Index1[] = { {{0}}, {{1}} };
    TestArray<CProxy_MyArray1>(Index1);
    const MyIndex<2> Index2[] = { {{0, 0}}, {{0, 1}}, {{1, 1}}, {{1, 0}} };
    TestArray<CProxy_MyArray2>(Index2);

    CkCallback qd(CkIndex_Main::quiescence(), mainProxy);
    CkStartQD(qd);
  }

  void quiescence(void)
  {
    CkExit();
  };
};

/*array [MyIndex<1>]*/
struct MyArray1 : public CBase_MyArray1
{
  MyArray1() {}
  MyArray1(CkMigrateMessage *m) {}

  void Test()
  {
    ckout << "[" << thisIndex.dimensions[0] << "]: test passed" << endl;
  }
};

/*array [MyIndex<2>]*/
struct MyArray2 : public CBase_MyArray2
{
  MyArray2() {}
  MyArray2(CkMigrateMessage *m) {}

  void Test()
  {
    ckout << "[" << thisIndex.dimensions[0] << "," << thisIndex.dimensions[1] << "]: test passed" << endl;
  }
};

#include "ArrayIndexTemplate.def.h"
