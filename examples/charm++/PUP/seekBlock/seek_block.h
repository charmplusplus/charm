
class C;
#include "seek_block.decl.h"
#include <pup.h>

class main : public CBase_main {
  public:
    main(CkMigrateMessage *m) {}
    main(CkArgMsg *m);
};

class A {
  public:
	int x, y;
	double z;
	void pup(PUP::er& p);
};

class B {
  public:
	char c;
	void pup(PUP::er& p);
};

class C {
  public:
	A a;
	B b;
	void pup(PUP::er& p);
};

class SimpleArray : public CBase_SimpleArray {
  public:
	SimpleArray(CkMigrateMessage *m) {}
	SimpleArray(){}
	~SimpleArray(){}

	void done(){
		CkPrintf("done int %d\n",c.a.x);
		CkExit();
	}

	void acceptData(C& inData){
		//do something to the object
		c=inData;
        c.a.x++;

		if(thisIndex==0) {
			done();
		} else {
			thisProxy[thisIndex-1].acceptData(c);
		}
	}

  private:
	C c;
};
