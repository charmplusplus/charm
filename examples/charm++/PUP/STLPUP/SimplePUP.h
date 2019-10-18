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

#include "SimplePUP.decl.h"
#include <vector>
#include <cassert>

class main : public CBase_main {

public:

  main(CkMigrateMessage *m) {}

  main(CkArgMsg *m);

};

template <typename U> class SimpleArray : public CBase_SimpleArray<U> {

 public:
  
  SimpleArray(CkMigrateMessage *m) {}

  SimpleArray(){}

  void done(){
    CkPrintf("done int %d\n",localCopy.publicInt);
    CkExit();
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
