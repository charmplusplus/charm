#include "streamtest.decl.h"
#include <iostream>
#include <time.h>

class Main : public CBase_Main {
public:
	Main(CkArgMsg* m){
		delete m;
		CkPrintf("Main program chare created\n");
		Ck::Stream::dummyFunction();
		sleep(10);
		CkPrintf("Main chare has finished sleeping, hopefully initial program has been completed\n");
		CkExit();
	}
};
#include "streamtest.def.h"
