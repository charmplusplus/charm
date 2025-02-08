#pragma once
#include <unordered_map>

#include "streamtoken.h"
namespace Ck { namespace Stream { namespace impl {
class StreamMessageCounter{
	// keep track of how many bytes were sent to each PE in the stream 
	std::unordered_map<size_t, size_t> _sent_counter;
	// keep track of how many bytes were received by each PE in the stream
	std::unordered_map<size_t, size_t> _received_counter;
	StreamToken _stream = 0;
	size_t _num_bytes_get = 0;
	size_t _number_of_expected_receives = 0;
	bool _close_initiated = false;
public:
	StreamMessageCounter();
	StreamMessageCounter(StreamToken);
	void setStreamWriteClosed();
	// given a source PE and the number of bytes, track the incoming message
	void processIncomingMessage(size_t, size_t);
	// given a destination PE and the number of bytes, track the outgoing message
	void processOutgoingMessage(size_t, size_t);
	// gets the sent counters in array format
	u_long* getSentCounterArray();
	// gets the received counters in array format
	u_long* getReceivedCounterArray();
	// method that sums everything in receive array
	size_t totalReceivedMessages();
	// determines if we have received all the data we should && stream is trying to be closed
	bool receivedAllData();

	size_t getNumberOfExpectedReceives();

	// set the number of receive messages we should expect after all messages on this pe have arrived 
	void setExpectedReceives(size_t);
	bool isCloseFlagSet();
	void setCloseFlag();

	void debugIncrementGetBytesCounter(int i) {
		_num_bytes_get += i;
	}

	void debugPrintTotalBytesFulfilled() {
		CkPrintf("DEBUG: Printing Total Bytes Fulfilled of stream %d on PE %d: %d\n", _stream, CkMyPe(), _num_bytes_get);

	}
	
	void debugPrintReceivedCounter() {
		CkPrintf("DEBUG: Printing Received Counter of stream %d on PE %d\n", _stream, CkMyPe());
		for (auto& fuck : _received_counter) {
			CkPrintf("r PE %zu: %zu\n", fuck.first, fuck.second);
		}
		CkPrintf("DEBUG: Done Printing Received Buffer\n");
	}

	void debugPrintSentCounter() {
		CkPrintf("DEBUG: Printing Sent Counter of stream %d on PE %d\n", _stream, CkMyPe());
		for (auto& fuck : _sent_counter) {
			CkPrintf("s PE %zu: %zu\n", fuck.first, fuck.second);
		}
		CkPrintf("DEBUG: Done Printing Sent Buffer\n");
	}

	void debugPrintAllFinal() {
		debugPrintTotalBytesFulfilled();
		debugPrintReceivedCounter();
		debugPrintSentCounter();
	}
};

}}}
