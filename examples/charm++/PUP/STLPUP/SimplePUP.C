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

    // create a vector of heap-allocated pings
    std::vector<Ping*>* pings = new std::vector<Ping*>();
    for (int i = 0; i < CkNumPes() * 4; i++) {
      pings->push_back(new Ping(i + 1));
    }
    // send the vector as a CkPointer
    thisProxy.accept(CkPointer<std::vector<Ping*>>(pings));
    // then clean it up
    for (auto p: *pings) delete p;
    delete pings;

    // start QD
    CkStartQD(CkCallback(CkCallback::ckExit));
}

void main::accept(std::shared_ptr<Ping> ping) {
  (*ping)();
}

void main::accept(std::vector<Ping*>* pings) {
  for (auto p: *pings) {
    std::shared_ptr<Ping> p_(p);
    thisProxy.accept(p_);
  }
  delete pings;
}
