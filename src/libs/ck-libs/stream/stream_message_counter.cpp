#pragma 
#include "stream.h"
#include <iostream>
#include <unordered_map>
#include <cstring>
#include <time.h>
#include <stdlib.h>


namespace Ck { namespace Stream { namespace impl {

		StreamMessageCounter::StreamMessageCounter() = default;

		StreamMessageCounter::StreamMessageCounter(StreamToken stream) : _stream(stream) {}

		bool StreamMessageCounter::isStreamClosed() {
			return _stream_write_closed;
		}

		void StreamMessageCounter::setStreamWriteClosed(){
			_stream_write_closed = true;
			for(auto& p : _counter){
				size_t pe = p.first;	
				CkpvAccess(stream_manager) -> sendAck(_stream, pe, p.second);
			}
		}

		void StreamMessageCounter::processIncomingMessage(size_t incoming_pe){
			if(_stream_write_closed){
				CkpvAccess(stream_manager) -> sendAck(_stream, incoming_pe, 1); // I have received 1 new message 
				return;
			}
			if(incoming_pe != CkMyPe()) {// no need to ack messages that aren't hitting the network and going to the same core
				_counter[incoming_pe]++;
			}

		}

		void StreamMessageCounter::addWriteAck(size_t num_messages){
			_write_acks+=num_messages;
		}

		bool StreamMessageCounter::allAcked(){
			return _num_sent_messages == _write_acks;
		}

		void StreamMessageCounter::addSentMessage() {
			_num_sent_messages++;
		}

	}
}
}



#include "CkStream.def.h"
#include "CkStream_impl.def.h"
