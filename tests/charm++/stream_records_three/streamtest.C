#include "streamtest.decl.h"
#include <iostream>
#include <time.h>

#define INVALID_STREAM_NO 99999999
#define NUM_REQUESTS 10

class Main : public CBase_Main {
	CProxy_Producers producers;
	CProxy_Middle middle;
	CProxy_Consumers consumers;
public:
	Main_SDAG_CODE
	Main(CkArgMsg* m){
		delete m;
		Ck::Stream::createNewStream(CkCallback(CkIndex_Main::producerMiddleStreamMade(0), thisProxy));
	}
	
};

class Producers : public CBase_Producers {
	StreamToken output_stream = INVALID_STREAM_NO;
    size_t count;
public:
	Producers(){
		beginWork();
	}

	void setOutputStreamId(StreamToken stream) {
		output_stream = stream;
		beginWork();
	}

	void beginWork() {
		if (output_stream == INVALID_STREAM_NO) return;
		for(int i = 0; i < NUM_REQUESTS; ++i){
            ++count;
			std::string record = generateRandomString();
			CkPrintf("Created: %s\n", record.c_str());
			Ck::Stream::putRecord(output_stream, (void*)record.c_str(), sizeof(char) * record.size() + 1);
		}
		Ck::Stream::flushLocalStream(output_stream);
		CkPrintf("Producer %d has written %d records to the stream %d...\n", thisIndex, NUM_REQUESTS, output_stream);
		contribute(sizeof(count), &count, CkReduction::sum_int, CkCallback(CkReductionTarget(Producers, doneWriting), thisProxy[0]));
	}

	void doneWriting(int sum){
		CkPrintf("Producers are done, closing stream id %d. Sent %d records.\n", output_stream, sum);
		Ck::Stream::closeWriteStream(output_stream);
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
    }
};

class Middle : public CBase_Middle {
	StreamToken input_stream = INVALID_STREAM_NO;
	StreamToken output_stream = INVALID_STREAM_NO;
	size_t _num_bytes_received = 0;
    size_t count;
public:
	Middle_SDAG_CODE
	Middle() {
		beginWork();
	}

	void setInputStreamId(StreamToken id) {
		input_stream = id;
		beginWork();
	}

	void setOutputStreamId(StreamToken id) {
		output_stream = id;
		beginWork();
	}

	void beginWork() {
		if (input_stream == INVALID_STREAM_NO || output_stream == INVALID_STREAM_NO) return;
		Ck::Stream::getRecord(input_stream, CkCallback(CkIndex_Middle::recvData(0), thisProxy[thisIndex]));
	}
};

class Consumers : public CBase_Consumers {
	private:
	StreamToken input_stream = INVALID_STREAM_NO;
	size_t _num_bytes_received = 0;
    size_t count = 0;
public:
	Consumers_SDAG_CODE
	Consumers() {
		beginWork();
	}

	void setInputStreamId(StreamToken id) {
		input_stream = id;
		beginWork();
	}

	void beginWork() {
		if (input_stream == INVALID_STREAM_NO) return;
		Ck::Stream::getRecord(input_stream, CkCallback(CkIndex_Consumers::recvData(0), thisProxy[thisIndex]));
	}

	void recvData(Ck::Stream::StreamDeliveryMsg* msg){
		CkPrintf("Consumer: In %d\n", input_stream);
		char* data = (char*)(msg -> data);
		_num_bytes_received += msg -> num_bytes;
		if (msg -> num_bytes != 0) {
            ++count;
			CkPrintf("Consumer received on stream id %d: %s\n", input_stream, data);
			CkPrintf("\n");
		} else {
            CkPrintf("Record was empty, ignoring\n");
        }
		if(msg -> status == Ck::Stream::StreamStatus::STREAM_OK) {
			CkPrintf("issuing another get request on stream id %d...\n", input_stream);
			Ck::Stream::getRecord(input_stream, CkCallback(CkIndex_Consumers::recvData(0), thisProxy[thisIndex]));
		} else {
			CkPrintf("Consumer %d has received the done signal and consumed %d records\n", thisIndex, count);
			CkCallback cb = CkCallback(CkReductionTarget(Consumers, finishReading), thisProxy[0]);
			contribute(sizeof(count), &count, CkReduction::sum_int, cb);
		}
	};
};
#include "streamtest.def.h"