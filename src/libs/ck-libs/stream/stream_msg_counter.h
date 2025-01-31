#pragma once
#include <unordered_map>

#include "streamtoken.h"
namespace Ck { namespace Stream { namespace impl {
class StreamMessageCounter{
	// counting the number of message received from a PE
	std::unordered_map<size_t, size_t> _counter;
	size_t _num_sent_messages = 0;
	size_t _write_acks = 0;
	bool _stream_write_closed = false;
	StreamToken _stream = 0;
public:
	StreamMessageCounter();
	StreamMessageCounter(StreamToken);
	bool isStreamClosed();
	void setStreamWriteClosed();
	void processIncomingMessage(size_t);
	void addSentMessage();
	void addWriteAck(size_t);
	bool allAcked();


};

}}}
