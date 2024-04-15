#include "streamtest.decl.h"
#include <iostream>
#include <time.h>

class Main : public CBase_Main {
public:
	Main_SDAG_CODE
	Main(CkArgMsg* m){
		delete m;
		Ck::Stream::createNewStream(CkCallback(CkIndex_Main::streamMade(0), thisProxy));
	}
};
#include "streamtest.def.h"
