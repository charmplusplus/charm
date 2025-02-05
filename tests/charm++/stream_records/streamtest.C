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

				for(int i = 0; i < 10; ++i){
					std::string record = generateRandomString();
                    CkPrintf("Created: %s\n", record.c_str());
					Ck::Stream::putRecord(_stream, (void*)record.c_str(), sizeof(char) * record.size() + 1);
				}
				Ck::Stream::flushLocalStream(_stream);
				CkPrintf("Producer %d has written %d size_t to the stream...\n", thisIndex, 10);
				contribute(CkCallback(CkReductionTarget(Producers, doneWriting), thisProxy[0]));
	}

	void doneWriting(){
		Ck::Stream::closeWriteStream(_stream);
	}

    std::string generateRandomString() {
        const std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        std::random_device rd;
        std::mt19937 generator(rd());  
        std::uniform_int_distribution<> lengthDistribution(5, 20);
        std::uniform_int_distribution<> distribution(0, characters.size() - 1);
        size_t length = lengthDistribution(generator);
        std::string randomString = "";
        for (size_t i = 0; i < length; ++i) {
            randomString += characters[distribution(generator)];
        }
        return randomString;
        // CkPrintf("i'm here lmao\n");
    }
};

class Consumers : public CBase_Consumers {
	StreamToken _stream;
	size_t _num_bytes_received = 0;
public:
	Consumers_SDAG_CODE
	Consumers(StreamToken stream) {
		_stream = stream;	
		Ck::Stream::getRecord(_stream, CkCallback(CkIndex_Consumers::recvData(0), thisProxy[thisIndex]));
	}
};
#include "streamtest.def.h"