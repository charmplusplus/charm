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

class main : public CBase_main {

public:

  main(CkMigrateMessage *m) {}

  main(CkArgMsg *m);

};


class SimpleArray : public CBase_SimpleArray {

 public:
  
  SimpleArray(CkMigrateMessage *m) {}

  SimpleArray(){}

  void done(){
    CkPrintf("done int %d\n",localCopy.publicInt);
    CkExit();
  }

  ~SimpleArray(){}

  void acceptData(HeapObject &inData){

    //do something to the object
    localCopy=inData;
    localCopy.doWork();

    if(thisIndex==0) //no one lower to pass to
      {
	done();
      }
    else
      { // pass object down one index
	thisProxy[thisIndex-1].acceptData(localCopy);
      }
  }

 private:

  HeapObject localCopy;

};

