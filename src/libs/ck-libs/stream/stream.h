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
	// create a new stream, with the callback taking a message returning the StreamToken
	void createNewStream(CkCallback cb);
	// flush the buffer of the local stream
	void flushLocalStream(StreamToken stream);
	// extract data from stream
	void get(StreamToken stream, size_t elem_size, size_t num_elems, CkCallback cb);
	// closing the write side of a stream
	void closeWriteStream(StreamToken);
	namespace impl {
		// used when buffering the request
		struct GetRequest {
			size_t requested_bytes;
			CkCallback cb;
			GetRequest(size_t, CkCallback);
		};
		// keep this struct here in case we need to trakc more metadata in the future
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
			char* _put_buffer;
			std::deque<InData> _get_buffer; // the buffer for outgoing data
			std::deque<GetRequest> _buffered_gets; // buffered get requests
			// the messages that need to be sent
			std::deque<DeliverStreamBytesMsg*> _buffered_msg_to_deliver;
			// size of buffer for data to be sent out to OTHER PEs (from puts)
			size_t _put_buffer_capacity=  4 * 1024 * 1024; // Note: Old code had in = put and out = get
			size_t _get_buffer_capacity=  4 * 1024 * 1024;
			size_t _put_buffer_size = 0;
			size_t _get_buffer_size = 0;
			StreamToken _stream_id= 0;
			size_t _coordinator_pe = 0;
			std::vector<size_t> _registered_pes;
			bool _registered_pe = false;
			StreamMessageCounter _counter;
			void _sendOutBuffer(char* data, size_t size);
			void _sendOutBuffer(DeliverStreamBytesMsg*);
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
			void handleGetRequest(GetRequest& gr);
			void pushBackRegisteredPE(size_t pe);
			size_t numBufferedDeliveryMsg();
			// just create the message from given info
			DeliverStreamBytesMsg* createDeliverBytesStreamMsg();
			DeliverStreamBytesMsg* createDeliverBytesStreamMsg(char* extra_data, size_t extra_bytes);
			void clearBufferedDeliveryMsg();
			void popFrontMsgOutBuffer();
			size_t coordinator();
			CkReductionMsg* setStreamClosed();
			void clearBufferedGetRequests();
			void setExpectedReceivesUponClose(size_t num_messages_to_receive);
			void setCloseFlag();
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
