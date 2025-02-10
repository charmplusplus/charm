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
	/// Used to put in data to stream when the things being put are of known size.
	/// It isn't recommended to intermix put and putRecord calls.
	void put(StreamToken stream, void* data, size_t elem_size, size_t num_elems);
	/// used to put a record 'data' into the stream. If the things being put into
	/// the stream are of variable length for different puts, it is best to use
	/// putRecord. It isn't recommended to mix put and putRecord calls
	void putRecord(StreamToken stream, void* data, size_t data_size);
	/// create a new stream, with the callback taking a message returning the StreamToken
	void createNewStream(CkCallback cb);
	/// flush the buffer of the local stream
	void flushLocalStream(StreamToken stream);
	/// extract data from stream, when knowing the exact shape of the data and what you want
	void get(StreamToken stream, size_t elem_size, size_t num_elems, CkCallback cb);
	/// extract a record of variable length not known prior	
	void getRecord(StreamToken stream, CkCallback cb);
	// closing the write side of a stream (currently the only side that is worth closing)
	void closeWriteStream(StreamToken);

	namespace impl {
		// used when buffering the request
		struct GetRequest {
			/// when this flag is true, we request sizeof(size_t) bytes first, then invoke a new fulfill request
			bool get_record = false;
			size_t requested_bytes;
			CkCallback cb;
			GetRequest(size_t, CkCallback);
		};
		/// returned when extracting data from the get buffer on get requests
		struct ExtractedData {
			char* buffer = 0;
			size_t num_bytes_copied;
		};
		// keep this struct here in case we need to trakc more metadata in the future
		struct StreamMetaData {
			std::vector<size_t> _registered_pes;
			bool close_buffered;
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

		/// message to send data to stream managers with the contents of a put_buffer
		/// this data is used to feed the gets from others
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
		/// used by StreamManagers to organize the data of multiple streams
		/// is used to manage both put and get requests
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
			/// on put requests, insert the data into the stream
			void insertToStream(char* data, size_t num_bytes);
			/// flush out the put buffer to some PE (no extra data)
			void flushOutBuffer();
			/// flush out the put buffer to some PE, along with the extra data
			void flushOutBuffer(char* extra_data, size_t extra_bytes);
			/// add the data sent to the receive buffer to get requests
			void addToRecvBuffer(DeliverStreamBytesMsg* data);
			/// fulfill a get request 
			void fulfillRequest(GetRequest& gr);
			/// serially get some data from the get buffer and return that to the user
			ExtractedData extractFromGetBuffer(char* ret_buffer, size_t bytes_to_copy);
			/// actual logic for handling an incoming get request.
			void handleGetRequest(GetRequest& gr);
			void pushBackRegisteredPE(size_t pe);
			/// the num of delivery messages that are buffered (no target pes to send to), 
			size_t numBufferedDeliveryMsg();
			// just create the message from given info
			DeliverStreamBytesMsg* createDeliverBytesStreamMsg();
			DeliverStreamBytesMsg* createDeliverBytesStreamMsg(char* extra_data, size_t extra_bytes);
			/// clears all of the buffered delivery messages when a new PE comes into play and things are buffered
			void clearBufferedDeliveryMsg();
			void popFrontMsgOutBuffer();
			/// returns the coordinator for which this stream's StreamCoordinator resides on
			size_t coordinator();
			CkReductionMsg* setStreamClosed();
			void clearBufferedGetRequests();
			// when the "Starter" chare sends the PEs what they should have received, this sets it
			void setExpectedReceivesUponClose(size_t num_messages_to_receive);
			// sets the close flag to true in the StreamMessageCounter
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
	
	/// returned to the user
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
