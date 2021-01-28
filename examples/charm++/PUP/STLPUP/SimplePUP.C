/////////////////////////////////////
//
//  SimplePUP.C
//
//  Definition of chares in SimplePUP
//
//  Author: Eric Bohm
//  Date: 2012/7/30
//
/////////////////////////////////////

#include "SimplePUP.h"
#include "SimplePUP.def.h"

template <typename U> void execute_example(std::vector<U> &dataToCompare)
{
  //normal object construction
  HeapObject<U> exampleObject(20, false);
  exampleObject.data = dataToCompare;

  //normal chare array construction
  CProxy_SimpleArray<U> simpleProxy= CProxy_SimpleArray<U>::ckNew(30);

  //pass object to remote method invocation on the chare array
  simpleProxy[29].acceptData(exampleObject, dataToCompare);
}

main::main(CkArgMsg *m)
{
    std::vector<float> dataToCompare1{ 10.23f, 20.92f, 30.71f };
    execute_example<float>(dataToCompare1);

    std::vector<int> dataToCompare2{ 10, 20, 30 };
    execute_example<int>(dataToCompare2);

    std::vector<bool> dataToCompare3{ false, false, true};
    execute_example<bool>(dataToCompare3);

    thisProxy.accept(std::make_shared<Ping>(42));
}

void main::accept(std::shared_ptr<Ping> ping)
{
    (*ping)();
}
