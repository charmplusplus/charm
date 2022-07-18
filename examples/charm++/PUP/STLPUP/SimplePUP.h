///////////////////////////////////////
//
//  SimplePUP.h  
//
//  Declaration of chares in SimplePUP
//
//  Author: Eric Bohm
//  Date: 2012/01/23
//
///////////////////////////////////////

class Ping;

#include "SimplePUP.decl.h"
#include <vector>
#include <cassert>

class Ping: public PUP::able {
  PUPable_decl(Ping);

  Ping(int value) : value_(value) {}
  Ping(CkMigrateMessage *m) : PUP::able(m) { }
  virtual ~Ping() { }

  virtual void pup(PUP::er &p) override {
    PUP::able::pup(p);
    p | value_;
  }

  void operator()() const {
    CkPrintf("Ping %d!\n", value_);
  }

  int value_;
};

class main : public CBase_main {

public:

  main(CkMigrateMessage *m) {}

  main(CkArgMsg *m);

  void accept(std::shared_ptr<Ping> ping);
  void accept(std::vector<Ping*>* pings);
};

template <typename U> class SimpleArray : public CBase_SimpleArray<U> {

 public:
  
  SimpleArray(CkMigrateMessage *m) {}

  SimpleArray(){}

  void done(){
    CkPrintf("done int %d\n",localCopy.publicInt);
  }

  ~SimpleArray(){}

  void acceptData(const HeapObject<U> &inData,
          const std::vector<U> &dataToCompare){
    
    //do something to the object
    localCopy=inData;

    assert(inData.data.size() == dataToCompare.size());
    for (int i = 0; i < dataToCompare.size(); i++)
    {
        assert(inData.data[i] == dataToCompare[i]);
    }

    localCopy.doWork();

    if(this->thisIndex==0) //no one lower to pass to
      {
	done();
      }
    else
      { // pass object down one index
	this->thisProxy[this->thisIndex-1].acceptData(localCopy, dataToCompare);
      }
  }

 private:

  HeapObject<U> localCopy;

};

#define CK_TEMPLATES_ONLY
#include "SimplePUP.def.h"
#undef CK_TEMPLATES_ONLY
