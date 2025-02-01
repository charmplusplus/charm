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
public:
	StreamMessageCounter();
	StreamMessageCounter(StreamToken);
	bool isStreamClosed();
	void setStreamWriteClosed();
	// given a source PE and the number of bytes, track the incoming message
	void processIncomingMessage(size_t, size_t);
	// given a destination PE and the number of bytes, track the outgoing message
	void processOutgoingMessage(size_t, size_t);
};

}}}
