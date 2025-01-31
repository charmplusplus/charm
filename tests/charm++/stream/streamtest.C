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
				CkPrintf("size of size_t: %d\n", sizeof(size_t));
				_stream = stream;
				size_t num_bytes_written = 0;
				for(int i = 0; i < 10; ++i){
					size_t brudda = i + 10 * thisIndex;
					Ck::Stream::put(_stream, &brudda, sizeof(size_t), 1);
					num_bytes_written += sizeof(size_t);
				}
				Ck::Stream::flushLocalStream(_stream);
				CkPrintf("Producer %d has written %d bytes to the stream...\n", thisIndex, num_bytes_written);
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
        CkPrintf("PE %d: Calling Get\n", CkMyPe());

		Ck::Stream::get(_stream, sizeof(size_t), 1, CkCallback(CkIndex_Consumers::recvData(0), thisProxy[thisIndex]));
	}
};
#include "streamtest.def.h"
