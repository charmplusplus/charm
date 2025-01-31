#include "stream.h"
#include <iostream>
#include <unordered_map>
#include <cstring>
#include <time.h>
#include <stdlib.h>


namespace Ck { namespace Stream {
	namespace impl {
		CProxy_Starter starter;
		CkpvDeclare(StreamManager*, stream_manager);
	} 
	namespace impl {

		GetRequest::GetRequest(size_t bytes_req, CkCallback cb_in){
			requested_bytes = bytes_req;	
			cb = cb_in;
		}

		InData::InData(DeliverStreamBytesMsg* msg, size_t num_bytes){
			_msg = msg;
			curr = msg -> data;
			num_bytes_rem = num_bytes;
		}
		
		void InData::freeData(){
			delete _msg;
		}

		class Starter : public CBase_Starter {
			CProxy_StreamManager stream_managers;
			size_t _curr_stream_token_id = 0;
			size_t _curr_stream_coordinator = 0;
			std::unordered_map<StreamToken, StreamMetaData> _meta_data;
			Starter_SDAG_CODE
		public:
			Starter(CkArgMsg* m){
				srand(time(NULL));   // Initialization, should only be called once.
				delete m;
				CkPrintf("The starter is alive\n");
				starter = thisProxy;
				stream_managers = CProxy_StreamManager::ckNew();
			}

			Starter(CkMigrateMessage* m) : CBase_Starter(m) {}
  			void pup(PUP::er& p){

			}

			void starterHello(){
				CkPrintf("Hello from the starter chare of CkStream!\n");
				std::cout << "this is a print from the std::cout library\n";
				CkExit();
			}

			void addRegisteredPE(StreamToken token, size_t pe){
				_meta_data[token]._registered_pes.push_back(pe);
				stream_managers.addRegisteredPE(token, pe);
			}

		};

		class StreamManager : public CBase_StreamManager {
			std::unordered_map<StreamToken, StreamBuffers> _stream_table;
			std::unordered_map<StreamToken, StreamCoordinator> _stream_coordinators;
			bool _registered_receive = false; // is this PE have a reader on it
			std::vector<size_t> _pes_to_send_to; // a list of valid pes to send to	
			StreamManager_SDAG_CODE
		public:
				StreamManager(){
					CkPrintf("Stream Manager created on PE=%d\n", CkMyPe());
					CkpvInitialize(StreamManager*, stream_manager);
					CkpvAccess(stream_manager) = this;
				}

				StreamManager(CkMigrateMessage* m) : CBase_StreamManager(m) {}
  				void pup(PUP::er& p){
					// do I bother populating this?
				}
				
				void registerPEWithCoordinator(StreamToken stream, size_t pe){
					_stream_coordinators[stream].registerThisPE(pe); // register PE with the coordinator
				}
				void broadcastAddThisPE(StreamToken stream, size_t pe){
					thisProxy.addRegisteredPE(stream, pe);
				}
				void addRegisteredPE(StreamToken token, size_t pe){
					StreamBuffers& sb = _stream_table[token];
					if(pe != CkMyPe()) 
						sb.pushBackRegisteredPE(pe);
					if(sb.numBufferedDeliveryMsg()){
						thisProxy[CkMyPe()].clearBufferedDeliveryMsg(token);	
					}
				}

				void clearBufferedDeliveryMsg(StreamToken token){
					StreamBuffers& sb = _stream_table[token];	
					sb.popFrontMsgOutBuffer();
					if(sb.numBufferedDeliveryMsg()){
						clearBufferedDeliveryMsg(token);
					}
				}

				void serveGetRequest(StreamToken stream, GetRequest gr){
					StreamBuffers& sb = _stream_table[stream];	
					sb.handleGetRequest(gr);
				}

				void sayHello(){
					CkPrintf("StreamManager %d says hello!\n", CkMyPe());
				}

				void initializeStream(int id, size_t coordinator_pe){
					_stream_table[id] = StreamBuffers(id, coordinator_pe);
					CkPrintf("On %d, we created stream %d with coordinator=%d\n", CkMyPe(), id, coordinator_pe);
					contribute(sizeof(id), &id, CkReduction::max_int, CkCallback(CkReductionTarget(Starter, streamCreated), starter));
				}

				void putToStream(StreamToken stream, void* data, size_t elem_size, size_t num_elems){
					StreamBuffers& buffer = _stream_table[stream];	
					buffer.insertToStream((char*) data, elem_size * num_elems);
				}
				
				void sendAck(StreamToken stream, size_t dest_pe, size_t num_messages){
					CkPrintf("sendAck on source PE[%d]: sending destination PE[%d] acks for %d messages\n", CkMyPe(), dest_pe, num_messages);
					thisProxy[dest_pe].ackWrites(stream, num_messages);
				}

				void recvData(DeliverStreamBytesMsg* in_msg){
					CkPrintf("PE %d: recv msg contents: num_bytes: %d\n", CkMyPe(), in_msg->num_bytes);

					StreamBuffers& buff = _stream_table[in_msg -> stream_id];
					char* temp_data = in_msg -> data;
					buff.addToRecvBuffer(in_msg);
					// delete in_msg; send should free this
				}

				void flushLocalStream(StreamToken stream){
					StreamBuffers& local_buffer = _stream_table[stream];
					local_buffer.flushOutBuffer();
				}

				inline void sendDeliverMsg(DeliverStreamBytesMsg* in_msg, ssize_t target_pe){
					CkPrintf("sendDeliverMsg msg contents: num_bytes: %d\n", in_msg->num_bytes);

					thisProxy[target_pe].recvData(in_msg);
				}

				void tellCoordinatorCloseWrite(StreamToken stream){
					CkPrintf("Coordinator on PE=%d starting the close process for stream=%d...\n", CkMyPe(), stream);
					StreamBuffers& sb = _stream_table[stream];
					size_t pe = sb.coordinator();
					thisProxy[pe].startCloseWriteStream(stream);
				}

				void startCloseWriteStream(StreamToken token){
					CkPrintf("PE=%d will initiate the closing process on all PEs for stream=%d\n", CkMyPe(), token);
					thisProxy.initiateWriteStreamClose(token);
				}

				void initiateWriteStreamClose(StreamToken token){
					CkPrintf("Stream Manager %d received a close request for stream=%d\n", CkMyPe(), token);
					StreamBuffers& sb = _stream_table[token];
					sb.setStreamClosed();
				}

				void tellCoordinatorLifetimeBytesWritten(StreamToken stream_id, size_t origin_pe, size_t bytes_written) {
					StreamCoordinator sc = _stream_coordinators[stream_id];
					sc.tellCoordinatorLifetimeBytesWritten(stream_id, origin_pe, bytes_written);
				} 

				void ackWrites(StreamToken stream, size_t num_messages){
					StreamBuffers& sb = _stream_table[stream];
					CkPrintf("PE #%d, calling ackWrites, which calls insertAck with %d acks\n", CkMyPe(), num_messages);
					sb.insertAck(num_messages);
					if(sb.isStreamClosed() && sb.allAcked()){
						CkPrintf("All of pe=%d messages for stream=%d have been acked\n", CkMyPe(), stream);
						sb.clearBufferedGetRequests();
					}

				}

		};

		void dummyImplFunction(){
			CkpvAccess(stream_manager)-> sayHello();	
		}
		StreamBuffers::StreamBuffers() = default;

		StreamBuffers::StreamBuffers(size_t stream_id) : _counter(stream_id){
			_stream_id = stream_id;
			_in_buffer = new char[_in_buffer_capacity];
		}

		StreamBuffers::StreamBuffers(size_t stream_id, size_t coordinator_pe) : _counter(stream_id){
			_stream_id = stream_id;
			_coordinator_pe = coordinator_pe;
			_in_buffer = new char[_in_buffer_capacity];
		}

		StreamBuffers::StreamBuffers(size_t stream_id, size_t in_buffer_capacity, size_t out_buffer_capacity) : _counter(stream_id){
			_stream_id = stream_id;
			_in_buffer_capacity = in_buffer_capacity;
			_out_buffer_capacity = out_buffer_capacity;
			_in_buffer = new char[_in_buffer_capacity];
		}

		StreamBuffers::StreamBuffers(size_t stream_id, size_t coordinator_pe, size_t in_buffer_capacity, size_t out_buffer_capacity) : _counter(stream_id) {
			_stream_id = stream_id;
			_coordinator_pe = coordinator_pe;
			_in_buffer_capacity = in_buffer_capacity;
			_out_buffer_capacity = out_buffer_capacity;
			_in_buffer = new char[_in_buffer_capacity];
		}

		void StreamBuffers::insertAck(size_t acks){
			CkPrintf("PE #%d, calling insertAck with %d ack\n", CkMyPe(), acks);
			_counter.addWriteAck(acks);
		}

		bool StreamBuffers::allAcked(){
			return 	_counter.allAcked();
		}

		bool StreamBuffers::isStreamClosed(){
			return _counter.isStreamClosed();
		}

		void StreamBuffers::setStreamClosed(){
			flushOutBuffer();
			_counter.setStreamWriteClosed();
			// tell coordinator how much data we have sent in our lifetime


			// check if all of my messages have been acked already
			if(allAcked()){
				CkPrintf("For stream=%d on pe=%d, All of my messages have been acked!\n", _stream_id, CkMyPe());
			}
		}

		size_t StreamBuffers::coordinator() {
			return _coordinator_pe;
		}

		void StreamBuffers::flushOutBuffer(){
			if(!_in_buffer_size) return;
			_sendOutBuffer(_in_buffer, _in_buffer_size);	
		}

		void StreamBuffers::popFrontMsgOutBuffer(){
			if(_msg_out_buffer.empty()){
				return;
			}
			DeliverStreamBytesMsg* msg = _msg_out_buffer.front();
			_msg_out_buffer.pop_front();
			ssize_t target_pe = _pickTargetPE();
			CkpvAccess(stream_manager) -> sendDeliverMsg(msg, target_pe);
			if(target_pe != CkMyPe()){
				_counter.addSentMessage();
			}
		}

		void StreamBuffers::flushOutBuffer(char* extra_data, size_t extra_bytes){
			if(!_in_buffer_size) return;
			char* bigger_buffer = new char[_in_buffer_size + extra_bytes];
			std::memcpy(bigger_buffer, _in_buffer, _in_buffer_size);
			std::memcpy(bigger_buffer + _in_buffer_size, extra_data, extra_bytes);
			// for (int i = 0; i < 4; ++i) {
			// 	CkPrintf("Shit in Bigger_buffer: %zu\n", ((size_t*)bigger_buffer)[i]);
			// }
			_sendOutBuffer(bigger_buffer, _in_buffer_size + extra_bytes);
			delete[] bigger_buffer;
		}

		void StreamBuffers::_sendOutBuffer(char* data, size_t size){
			DeliverStreamBytesMsg* msg = new (size) DeliverStreamBytesMsg(data,size);
			CkPrintf("sendOutBuffer msg contents: data: %zu, num_bytes: %d\n", *((size_t*)data + 1), msg->num_bytes);

			msg -> stream_id = _stream_id;
			msg -> sender_pe = CkMyPe();
			ssize_t target_pe = _pickTargetPE();
			CkPrintf("sender_pe: %d, target_pe: %d\n", CkMyPe(), target_pe);
			if(target_pe == -1){
				CkPrintf("Sending to local pe\n");
				_msg_out_buffer.push_back(msg);
				_in_buffer_size = 0;
				return;
			}
			// insert sending code here
			CkpvAccess(stream_manager) -> sendDeliverMsg(msg, target_pe);
			_in_buffer_size = 0;
			_counter.addSentMessage();
			_amount_data_sent += size;
		}

		ssize_t StreamBuffers::_pickTargetPE(){
			#define CHANCE_OF_LOCAL 4
			// insert some random picking logic
			if(!_registered_pe){
				if(!(_registered_pes.size())) return -1;
				else return _registered_pes[(rand() % _registered_pes.size())];
			} else {
				if(!(_registered_pes.size()) || !(rand() % CHANCE_OF_LOCAL)){
					return CkMyPe();
				} else {
					return _registered_pes[(rand() % _registered_pes.size())];
				}
			}

		}

		void StreamBuffers::handleGetRequest(GetRequest gr){
			if(!_registered_pe){
				_registered_pe = true;
				// tell this stream's coordinator
				CkpvAccess(stream_manager) -> registerPEWithCoordinator(_stream_id, CkMyPe()); // tell coordinator + all the other PEs
			}
			if((gr.requested_bytes > _out_buffer_size) || !_buffered_reqs.empty()){
				CkPrintf("pushing back bc not enough data, data has not arrived, requested_bytes=%d, out_buffer_size=%d\n", gr.requested_bytes, _out_buffer_size);
				_buffered_reqs.push_back(gr);
			} else {
				CkPrintf("fulfilling req\n");

				fulfillRequest(gr);
			}
		}

		void StreamBuffers::insertToStream(char* data, size_t num_bytes){
			if(num_bytes + _in_buffer_size > _in_buffer_capacity){
				flushOutBuffer(data, num_bytes);
				return;
			}
			std::memcpy(_in_buffer + _in_buffer_size, data, num_bytes);
			_in_buffer_size += num_bytes;
			if(_in_buffer_size == _in_buffer_capacity){
				CkPrintf("I have reached capacity; time to fly!\n");
				flushOutBuffer();
			}
		}

		void StreamBuffers::addToRecvBuffer(DeliverStreamBytesMsg* data){
			size_t num_bytes = data -> num_bytes;
			_counter.processIncomingMessage(data -> sender_pe);
			if(!_out_buffer_capacity || ((_out_buffer_size + num_bytes) <= _out_buffer_capacity)){
				_out_buffer.push_back(InData(data, num_bytes));
				_out_buffer_size += num_bytes;
				CkPrintf("-- addToRecvBuffer === PE#%d, Adding to out buffer size, out_buffer_size=%d, num_bytes=%d\n", CkMyPe(), _out_buffer_size, num_bytes);
			} else {
				CkPrintf("capacity has been reached on the recv buffer, so dropping incoming message\n");
			}
			CkPrintf("In addToRecvBuffer, attempting to process buffered reqs\n");
			while(!_out_buffer.empty() && !_buffered_reqs.empty()){ // keep fulfilling buffered requests in FIFO order
				GetRequest& gr = _buffered_reqs.front();
				if(gr.requested_bytes > _out_buffer_size){
					CkPrintf("PE %d: Breaking out of addToRecvBuffer as there is not enough data to serve more buffered getRequests, requested_bytes=%d, _out_buffer_size=%d\n", CkMyPe(), gr.requested_bytes, _out_buffer_size);
					break;
				}
				CkPrintf("PE %d: Servicing a buffered GetRequest\n", CkMyPe());
				// we know there are enough bytes in the stream to fulfill the request
				GetRequest curr_req = gr;
				_buffered_reqs.pop_front();
				fulfillRequest(curr_req);
			}
		}

		void StreamBuffers::fulfillRequest(GetRequest& gr){
			size_t num_bytes_copied = 0;
			size_t num_bytes_requested = std::min(gr.requested_bytes, _out_buffer_size);
			StreamDeliveryMsg* res;
			if(isStreamClosed() && allAcked() && _out_buffer.empty()){
				CkPrintf("]]]]]] Short circuiting on PE %d\n", CkMyPe());
				res = new(0) StreamDeliveryMsg(_stream_id); // avoid allocating useless memory
				goto sendingRequest;
			}
			res = new (num_bytes_requested) StreamDeliveryMsg(_stream_id);

			while(!_out_buffer.empty() && (num_bytes_copied != num_bytes_requested)){ // the request hasn't been fulfilled and ther's still data to copy
				CkPrintf("////// is out buffer empty=%d, num_bytes_copied=%d, num_bytes_reqed=%d\n", _out_buffer.empty(), num_bytes_copied, num_bytes_requested);
				InData& front = _out_buffer.front();
				size_t bytes_rem = num_bytes_requested - num_bytes_copied;
				if(front.num_bytes_rem > bytes_rem){
					std::memcpy((res -> data + num_bytes_copied), front.curr, bytes_rem);
					num_bytes_copied += bytes_rem;
					front.num_bytes_rem -= bytes_rem;
					front.curr += bytes_rem;
				} else {
					std::memcpy((res -> data + num_bytes_copied), front.curr, front.num_bytes_rem);
					num_bytes_copied += (front.num_bytes_rem);

					// delete this buffer at the front of the queue 
					front.freeData();
					_out_buffer.pop_front();
				}
			}
			_out_buffer_size -= num_bytes_copied;
sendingRequest:
			// copied all of the data we could
			res -> num_bytes = num_bytes_copied;
			CkPrintf("+++++++++ out_buffer_size: %d, gr.requested_bytes=%d, num_bytes_copied=%d, isStreamClosed= %d, allAcked= %d\n", _out_buffer_size, gr.requested_bytes, num_bytes_copied, isStreamClosed(), allAcked());
			if(isStreamClosed() && allAcked() && !_out_buffer_size){
				res -> status = StreamStatus::STREAM_CLOSED;
				CkPrintf("???????? PE[%d] stream closed stats: out_buffer_size: %d, gr.requested_bytes=%d, num_bytes_copied=%d \n", CkMyPe(), _out_buffer_size, gr.requested_bytes, num_bytes_copied);
				CkPrintf("???? PE[%d] stream closed message info: %zu, %d\n", CkMyPe(), res -> stream_id, res -> num_bytes);
				CkPrintf("Stream has been closed: out_buffer_size=%d\n", _out_buffer_size);
			} else {
				res -> status = StreamStatus::STREAM_OK;
			}
			gr.cb.send(res);
			return;
		}

	void StreamBuffers::clearBufferedGetRequests(){
		// clear all of the buffered get requests
		while(!_buffered_reqs.empty()){
			GetRequest& gr = _buffered_reqs.front();
			_buffered_reqs.pop_front();
			fulfillRequest(gr);
		}
	}

		void StreamBuffers::pushBackRegisteredPE(size_t pe){
			_registered_pes.push_back(pe);
		}

		size_t StreamBuffers::numBufferedDeliveryMsg(){
			return _msg_out_buffer.size();
		}

		StreamCoordinator::StreamCoordinator() = default;
		
		StreamCoordinator::StreamCoordinator(StreamToken stream) : _stream(stream) {}

		void StreamCoordinator::tellCoordinatorLifetimeBytesWritten(StreamToken stream_id, size_t origin_pe, size_t bytes_written) {
			(_bytes_written[stream_id])[origin_pe] += bytes_written;
		}
		
		
		void StreamCoordinator::registerThisPE(size_t pe){
				_meta_data._registered_pes.push_back(pe);
				CkpvAccess(stream_manager) -> broadcastAddThisPE(_stream, pe);
		}



		StreamMessageCounter::StreamMessageCounter() = default;

		StreamMessageCounter::StreamMessageCounter(StreamToken stream) : _stream(stream) {}

		bool StreamMessageCounter::isStreamClosed() {
			return _stream_write_closed;
		}

		void StreamMessageCounter::setStreamWriteClosed(){
			CkPrintf("[[[[[[ Write side is closed on PE#%d\n", CkMyPe());
			_stream_write_closed = true;
			for(auto& p : _counter){
				size_t pe = p.first;	
				CkpvAccess(stream_manager) -> sendAck(_stream, pe, p.second);
			}
		}

		void StreamMessageCounter::processIncomingMessage(size_t sender_pe){
			if(sender_pe == CkMyPe())
				return;
			// we know that it is from a remote PE, thus we should ack if required / keep track of incoming messages
			if(_stream_write_closed){
				CkPrintf("On PE[%d], _stream_write_closed is True, going to sendAck from me to source_pe [%d]\n", CkMyPe(), sender_pe);
	
				CkpvAccess(stream_manager) -> sendAck(_stream, sender_pe, 1); // I have received 1 new message 
				return;
			}
			CkPrintf("On PE[%d], incrememting the messages sent from source_pe [%d]\n", CkMyPe(), sender_pe);
			_counter[sender_pe]++;
		}

		void StreamMessageCounter::addWriteAck(size_t num_messages){
			CkPrintf("On PE[%d], addWriteAck called with num message [%d]\n", CkMyPe(), num_messages);

			_write_acks+=num_messages;
		}

		bool StreamMessageCounter::allAcked(){
			CkPrintf("PE %d: _num_sent_messages=%d, _write_acks=%d\n", CkMyPe(), _num_sent_messages, _write_acks);
			return _num_sent_messages == _write_acks;
		}

		void StreamMessageCounter::addSentMessage() {
			_num_sent_messages++;
		}

		inline void impl_put(StreamToken stream, void* data, size_t elem_size, size_t num_elems){
			CkpvAccess(stream_manager) -> putToStream(stream, data, elem_size, num_elems);		
		}
		inline void impl_flushLocalStream(StreamToken stream){
			CkpvAccess(stream_manager) -> flushLocalStream(stream);
		}

		inline void impl_get(StreamToken stream, size_t elem_size, size_t num_elems, CkCallback cb){
			GetRequest gr(elem_size * num_elems, cb);
			CkpvAccess(stream_manager) -> serveGetRequest(stream, gr);
			return;
		}

		inline void impl_closeWriteStream(StreamToken stream){
			CkpvAccess(stream_manager) -> tellCoordinatorCloseWrite(stream);
		}
	}
	void dummyFunction(){
		impl::dummyImplFunction();
		std::cout << "about to try execute starterHello()\n";
		impl::starter.starterHello();
		std::cout << "should have finished starterHello()\n";
	}

	void createNewStream(CkCallback cb){
		impl::starter.createNewStream(cb);
	}

	void put(StreamToken stream, void* data, size_t elem_size, size_t num_elems){
		impl::impl_put(stream, data, elem_size, num_elems);
	}

	void putRecord(StreamToken stream, void* data, size_t data_size) {
		size_t total_size = sizeof(size_t) + data_size;

		char* record_buffer = new char[total_size];
		std::memcpy(record_buffer, &data_size, sizeof(size_t));
		std::memcpy(record_buffer + sizeof(size_t), data, data_size);

		impl::impl_put(stream, record_buffer, sizeof(char), total_size);
	}

	void flushLocalStream(StreamToken stream){
		impl::impl_flushLocalStream(stream);
	}

	void get(StreamToken stream, size_t elem_size, size_t num_elems, CkCallback cb){
		impl::impl_get(stream, elem_size, num_elems, cb);
	}

	void getRecord(StreamToken stream, CkCallback cb) {
		StreamDeliveryMsg* msg;
		impl::impl_get(stream, sizeof(size_t), 1, CkCallbackResumeThread((void*&)msg));
		size_t record_size = *(size_t*)msg->data;
		impl::impl_get(stream, sizeof(char), record_size, cb);
	}

	void closeWriteStream(StreamToken stream){
		impl::impl_closeWriteStream(stream);
		return;
	}


}
}

#include "CkStream.def.h"
#include "CkStream_impl.def.h"
