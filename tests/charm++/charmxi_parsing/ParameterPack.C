
#include <tuple>

#include "ParameterPack.decl.h"

/*readonly*/ CProxy_Main mainProxy;

struct Main : public CBase_Main
{
  Main(CkArgMsg* m)
  {
    mainProxy = thisProxy;
    CProxy_ParameterPack::ckNew();
  };

  void done()
  {
    CkExit();
  }
};

struct ParameterPack : public CBase_ParameterPack
{
  ParameterPack()
  {
    auto myTuple = std::make_tuple(1, 2, 3.14);
    thisProxy.Test1(myTuple);
    ckout << "Test #1 passed" << endl;
    thisProxy.Test2(myTuple);
    ckout << "Test #2 passed" << endl;
    mainProxy.done();
  }

  template <typename... Args>
  size_t Test1(std::tuple<Args...>&& args)
  {
    return std::tuple_size<std::tuple<Args...>>::value;
  }

  template <class... Args>
  std::tuple<Args...> Test2(std::tuple<Args...>&& args)
  {
    return args;
  }
};

#define CK_TEMPLATES_ONLY
#include "ParameterPack.def.h"
#undef CK_TEMPLATES_ONLY

#include "ParameterPack.def.h"
