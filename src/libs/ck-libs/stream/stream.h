#ifndef CK_STREAM_H
#define CK_STREAM_H

#include <iostream>
#include <cstring>
#include "CkStream.decl.h"
#include "CkStream_impl.decl.h"

typedef size_t StreamToken;

namespace Ck { namespace Stream {
	void dummyFunction();
	// API to insert into the stream
	void put(StreamToken stream, void* data, size_t elem_size, size_t num_elems);
	// create a new stream, with the callback taking a message returning the StreamToken
	void createNewStream(CkCallback cb);
	// flush the buffer of the local stream
	void flushLocalStream(StreamToken stream);
	namespace impl {
		// message to send data to stream managers and whatnot
		class DeliverStreamBytesMsg : public CMessage_DeliverStreamBytesMsg{
		public:
			char* data;
			size_t num_bytes;
			StreamToken stream_id;
			DeliverStreamBytesMsg(){}
			DeliverStreamBytesMsg(char* in_data, size_t in_num_bytes){
				num_bytes = in_num_bytes;
				std::memcpy(data, in_data, num_bytes);
			}
		};
		// used by StreamManagers to organize the data of multiple streams
		class StreamBuffers {
			char* _in_buffer; // the buffer for incoming data; once filled, just drop extra data
			char* _out_buffer; // the buffer for outgoing data
			size_t _in_buffer_capacity= 4 * 1024 * 1024;
			size_t _out_buffer_capacity= 4 * 1024 * 1024;
			size_t _in_buffer_size = 0;
			size_t _out_buffer_size = 0;
			size_t _stream_id= 0;
			void _sendOutBuffer(char* data, size_t size);
			size_t _pickTargetPE();
		public:
			StreamBuffers(); // used by the hashmap
			StreamBuffers(size_t stream_id);
			StreamBuffers(size_t stream_id, size_t in_buffer_capacity, size_t out_buffer_capacity);
			void insertToStream(char* data, size_t num_bytes);
			void flushOutBuffer();
			void flushOutBuffer(char* extra_data, size_t extra_bytes);
		};


	}
	class StreamIdMessage: public CMessage_StreamIdMessage {
	public:
		StreamToken id;
		StreamIdMessage() {}
		StreamIdMessage(StreamToken id_in) : id(id_in){}
	};
	
}}

#endif
