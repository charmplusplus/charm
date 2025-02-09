#include "streamtest.decl.h"
#include <iostream>
#include <time.h>

class Main : public CBase_Main {
	CProxy_Producers producers;
	CProxy_Consumers consumers;
public:
	Main_SDAG_CODE
	Main(CkArgMsg* m){
		delete m;
		Ck::Stream::createNewStream(CkCallback(CkIndex_Main::streamMade(0), thisProxy));
	}
};

class Producers : public CBase_Producers {
	StreamToken _stream;
public:
	Producers(StreamToken stream){
				_stream = stream;
				CkPrintf("Producer[%d] is on PE=%d\n", thisIndex, CkMyPe());
				for(int i = 0; i < 10; ++i){
					size_t brudda = i + 10 * thisIndex;
					Ck::Stream::put(_stream, &brudda, sizeof(size_t), 1);
				}
				Ck::Stream::flushLocalStream(_stream);
				CkPrintf("Producer %d has written %d size_t to the stream, which is %d total bytes...\n", thisIndex, 10, 10 * sizeof(size_t));
				contribute(CkCallback(CkReductionTarget(Producers, doneWriting), thisProxy[0]));
	}

	void doneWriting(){
		Ck::Stream::closeWriteStream(_stream);
	}
};

class Consumers : public CBase_Consumers {
	StreamToken _stream;
	size_t _num_ints_received = 0;
public:
	Consumers_SDAG_CODE
	Consumers(StreamToken stream) {
		_stream = stream;	
		CkPrintf("DEBUG CONSUMERS: CONSUMER[%d] is on PE[%zu]\n", thisIndex, CkMyPe());
		Ck::Stream::get(_stream, sizeof(size_t), 1, CkCallback(CkIndex_Consumers::recvData(0), thisProxy[thisIndex]));
	}
};
#include "streamtest.def.h"
