
struct MyStruct
{
  typedef int MyType;
};

#include "Typename.decl.h"

/*readonly*/ CProxy_Main mainProxy;

struct Main : public CBase_Main
{
  Main(CkArgMsg* m)
  {
    mainProxy = thisProxy;
    CProxy_Typename::ckNew(m->argc);
  };

  void done()
  {
    CkExit();
  }
};

struct Typename : public CBase_Typename
{
  Typename(const int x)
  {
    thisProxy.Test1(x);
    thisProxy.Test2(x);
    thisProxy.Test3();
    mainProxy.done();
  }

  void Test1(typename MyStruct::MyType x)
  {
    ckout << "Test #1 passed (" << x << ")" << endl;
  }

  void Test2(const typename MyStruct::MyType &x)
  {
    ckout << "Test #2 passed (" << x << ")" << endl;
  }

  void Test3()
  {
    ckout << "Test #3 passed (" << thisProxy.Test3_Value() << ")" << endl;
  }

  typename MyStruct::MyType Test3_Value()
  {
    return 1;
  }
};

#include "Typename.def.h"
