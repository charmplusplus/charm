#ifndef CK_STREAM_H
#define CK_STREAM_H
#include "CkStream.decl.h"
#include "CkStream_impl.decl.h"
#include "stream_msg_counter.h"

#include <iostream>
#include <cstring>
#include <queue>
#include <vector>


// typedef size_t StreamToken;

namespace Ck { namespace Stream {
	void dummyFunction();
	// API to insert into the stream
	void put(StreamToken stream, void* data, size_t elem_size, size_t num_elems);
	// Insert record into stream
	void putRecord(StreamToken stream, void* data, size_t data_size);
	// create a new stream, with the callback taking a message returning the StreamToken
	void createNewStream(CkCallback cb);
	// flush the buffer of the local stream
	void flushLocalStream(StreamToken stream);
	// extract data from stream
	void get(StreamToken stream, size_t elem_size, size_t num_elems, CkCallback cb);
	// extract record from stream
	void getRecord(StreamToken stream, CkCallback cb);
	// closing the write side of a stream
	void closeWriteStream(StreamToken);
	namespace impl {
		// used when buffering the request
		struct GetRequest {
			size_t requested_bytes;
			CkCallback cb;
			GetRequest(size_t, CkCallback);
		};

		struct StreamMetaData {
			std::vector<size_t> _registered_pes;
		};
		class DeliverStreamBytesMsg;
		class CMessage_DeliverStreamBytesMsg;
		// used to organize incoming data entries to be served to user
		struct InData {
			DeliverStreamBytesMsg* _msg;
			char* curr;
			size_t num_bytes_rem;
			InData(DeliverStreamBytesMsg* msg, size_t num_bytes);
			void freeData();
		};

		// message to send data to stream managers and whatnot
		class DeliverStreamBytesMsg;
		class DeliverStreamBytesMsg : public CMessage_DeliverStreamBytesMsg {
		public:
			char* data;
			size_t num_bytes;
			size_t sender_pe;
			StreamToken stream_id;
			DeliverStreamBytesMsg(){}
			DeliverStreamBytesMsg(char* in_data, size_t in_num_bytes){
				num_bytes = in_num_bytes;
				std::memcpy(data, in_data, num_bytes);
			}
		};
		// used by StreamManagers to organize the data of multiple streams
		class StreamBuffers {
			char* _in_buffer; // the buffer for incoming data; once filled, just drop extra data; (_in_buffer is used for the data going out; I should rename this at some point). This is what the setters push to.
			std::deque<InData> _out_buffer; // the buffer for outgoing data, this is the buffer that the getters pull from
			std::deque<GetRequest> _buffered_reqs;
			std::deque<DeliverStreamBytesMsg*> _msg_out_buffer;
			size_t _in_buffer_capacity= 4 * 1024 * 1024;
			size_t _out_buffer_capacity= 4 * 1024 * 1024;
			size_t _in_buffer_size = 0;
			size_t _out_buffer_size = 0;
			size_t _stream_id= 0;
			size_t _coordinator_pe = 0;
			std::vector<size_t> _registered_pes;
			bool _registered_pe = false;
			StreamMessageCounter _counter;
			void _sendOutBuffer(char* data, size_t size);
			ssize_t _pickTargetPE();

		public:
			StreamBuffers(); // used by the hashmap
			StreamBuffers(size_t stream_id);
			StreamBuffers(size_t stream_id, size_t coordinator_pe);
			StreamBuffers(size_t stream_id, size_t in_buffer_capacity, size_t out_buffer_capacity);
			StreamBuffers(size_t stream_id, size_t coordinator_pe, size_t in_buffer_capacity, size_t out_buffer_capacity);
			void insertToStream(char* data, size_t num_bytes);
			void flushOutBuffer();
			void flushOutBuffer(char* extra_data, size_t extra_bytes);
			void addToRecvBuffer(DeliverStreamBytesMsg* data);
			void fulfillRequest(GetRequest& gr);
			void handleGetRequest(GetRequest gr);
			void pushBackRegisteredPE(size_t pe);
			size_t numBufferedDeliveryMsg();
			void popFrontMsgOutBuffer();
			size_t coordinator();
			bool isStreamClosed();
			void setStreamClosed();
			void insertAck(size_t);
			bool allAcked();
			void clearBufferedGetRequests();
		};
		
		class StreamCoordinator {
			StreamMetaData _meta_data; // contains the metadata of the stream
			StreamToken _stream;
		public:
			StreamCoordinator();
			StreamCoordinator(StreamToken stream);
			void registerThisPE(size_t pe);
		};

	}

	enum class StreamStatus {
		STREAM_OK,
		STREAM_CLOSED,
	};
	class StreamIdMessage: public CMessage_StreamIdMessage {
	public:
		StreamToken id;
		StreamIdMessage() {}
		StreamIdMessage(StreamToken id_in) : id(id_in){}
	};

	class StreamDeliveryMsg: public CMessage_StreamDeliveryMsg {
		public:
			StreamToken stream_id;
			char *data;
			size_t num_bytes;
			StreamStatus status;

			StreamDeliveryMsg(StreamToken id) : stream_id(id) {}
	};
	
}}

#endif
