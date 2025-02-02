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

			void tellManagersExpectedReceives(CkReductionMsg* msg){
				CkReduction::tupleElement* results;
				int numReductions;
				msg -> toTuple(&results, &numReductions);
				double* totalMessageSentToPEs = (double*)results[0].data;
				StreamToken st = *((size_t*)results[1].data);
				for(int i = 0; i < CkNumPes(); ++i){
					// invoke some entry method
					stream_managers[i].expectedReceivesUponClose(st, size_t(totalMessageSentToPEs));
				}
			}

		};

		class StreamManager : public CBase_StreamManager {
			// maps the StreamToken to the StreamBuffers
			std::unordered_map<StreamToken, StreamBuffers> _stream_table;
			// maps the stream token to the StreamCoordinator if the coordinator for the given stream resides on this PE
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

				void expectedReceivesUponClose(StreamToken token, size_t num_messages_to_receive){
					StreamBuffers& sb = _stream_table[token];
					sb.setExpectedReceivesUponClose(num_messages_to_receive);
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

				void recvPutBufferFromPE(DeliverStreamBytesMsg* in_msg){
					StreamBuffers& buff = _stream_table[in_msg -> stream_id];
					char* temp_data = in_msg -> data;
					buff.addToRecvBuffer(in_msg);
				}

				void flushLocalStream(StreamToken stream){
					StreamBuffers& local_buffer = _stream_table[stream];
					local_buffer.flushOutBuffer();
				}

				inline void sendDeliverMsg(DeliverStreamBytesMsg* msg_to_send, ssize_t target_pe){
					// invoke the entry method for sending "in_msg" to the correct PE
					thisProxy[target_pe].recvPutBufferFromPE(msg_to_send);
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

				void addRegisteredPE(StreamToken stream, size_t pe){
					StreamBuffers& sb = _stream_table[stream];
					sb.pushBackRegisteredPE(pe);
					sb.clearBufferedDeliveryMsg();
				}

				void serveGetRequest(StreamToken stream, GetRequest& gr){
					StreamBuffers& local_buffer = _stream_table[stream];
					local_buffer.handleGetRequest(gr);
				}

		};

		// TODO: Can we get rid of this?
		// void dummyImplFunction(){
		// 	CkpvAccess(stream_manager)-> sayHello();	
		// }


		StreamBuffers::StreamBuffers() = default;

		StreamBuffers::StreamBuffers(size_t stream_id) : _counter(stream_id){
			_stream_id = stream_id;
			_put_buffer = new char[_put_buffer_capacity];
		}

		StreamBuffers::StreamBuffers(size_t stream_id, size_t coordinator_pe) : _counter(stream_id){
			_stream_id = stream_id;
			_coordinator_pe = coordinator_pe;
			_put_buffer = new char[_put_buffer_capacity];
		}

		StreamBuffers::StreamBuffers(size_t stream_id, size_t in_buffer_capacity, size_t out_buffer_capacity) : _counter(stream_id){
			_stream_id = stream_id;
			_put_buffer_capacity = in_buffer_capacity;
			_get_buffer_capacity = out_buffer_capacity;
			_put_buffer = new char[_put_buffer_capacity];
		}

		StreamBuffers::StreamBuffers(size_t stream_id, size_t coordinator_pe, size_t in_buffer_capacity, size_t out_buffer_capacity) : _counter(stream_id) {
			_stream_id = stream_id;
			_coordinator_pe = coordinator_pe;
			_put_buffer_capacity = in_buffer_capacity;
			_get_buffer_capacity = out_buffer_capacity;
			_put_buffer = new char[_put_buffer_capacity];
		}
		
		void StreamBuffers::setExpectedReceivesUponClose(size_t num_messages_to_receive){
			_counter.setExpectedReceives(num_messages_to_receive);
			// if we have already received the numbero f messages, we just mark ourselves as closed
			if(_counter.receivedAllData()){
				CkPrintf("On PE[%d], the StreamManager has received all %zu messages\n", CkMyPe(), num_messages_to_receive);
			}
		}

		void StreamBuffers::setStreamClosed(){
			flushOutBuffer();
			// send the coordinator how many bytes were sent
			u_long* sent_arr = _counter.getSentCounterArray();
			// original type is size_t, but changing it to int for the reducer. Do I need to do this?
			int st = _stream_id;
			// make a tuple reduction
			CkReduction::tupleElement tupleRedn[] = {
				CkReduction::tupleElement(CkNumPes() * sizeof(double), sent_arr, CkReduction::sum_int),
				CkReduction::tupleElement(CkNumPes() * sizeof(StreamToken), &st, CkReduction::max_int)
			};
			int tuple_size = 2;
			CkReductionMsg* msg = CkReductionMsg::buildFromTuple(tupleRedn, tuple_size);
			CkCallback cb(CkIndex_Starter::tellManagersExpectedReceives(0), starter); // This calls StreamManager::expectedReceivesUponClose
			msg -> setCallback(cb);
			contribute(msg);
		}

		size_t StreamBuffers::coordinator() {
			return _coordinator_pe;
		}

		void StreamBuffers::flushOutBuffer(){
			if(!_put_buffer_size) return;
			DeliverStreamBytesMsg* msg = createDeliverBytesStreamMsg();
			_sendOutBuffer(msg);	
		}

		void StreamBuffers::popFrontMsgOutBuffer(){
			if(_buffered_msg_to_deliver.empty()){
				return;
			}
			DeliverStreamBytesMsg* msg = _buffered_msg_to_deliver.front();
			_buffered_msg_to_deliver.pop_front();
			ssize_t target_pe = _pickTargetPE();
			if (target_pe == -1){
				CkPrintf("how are we able to have -1 as a target_pe when the popFrontMessage method is invoked...\n");
			}
			size_t num_bytes = msg -> num_bytes;
			CkpvAccess(stream_manager) -> sendDeliverMsg(msg, target_pe);
			_counter.processOutgoingMessage(target_pe, num_bytes);
		}

		void StreamBuffers::flushOutBuffer(char* extra_data, size_t extra_bytes){
			if(!_put_buffer_size) return;
			DeliverStreamBytesMsg* msg = createDeliverBytesStreamMsg(extra_data, extra_bytes);
			_sendOutBuffer(msg);
		}

		DeliverStreamBytesMsg* StreamBuffers::createDeliverBytesStreamMsg(){
			DeliverStreamBytesMsg* msg = new (_put_buffer_size) DeliverStreamBytesMsg(_put_buffer, _put_buffer_size);
			msg -> stream_id = _stream_id;
			msg -> sender_pe = CkMyPe();
			return msg;
		}

		DeliverStreamBytesMsg* StreamBuffers::createDeliverBytesStreamMsg(char* extra_data, size_t extra_bytes){
			size_t total_size = extra_bytes + _put_buffer_size;
			DeliverStreamBytesMsg* msg = new (total_size) DeliverStreamBytesMsg();
			std::memcpy(msg -> data, _put_buffer, _put_buffer_size);
			std::memcpy(msg -> data + _put_buffer_size, extra_data, extra_bytes);
			msg -> num_bytes = total_size;
			msg -> stream_id = _stream_id;
			msg -> sender_pe = CkMyPe();
			return msg;
		}

		void StreamBuffers::_sendOutBuffer(DeliverStreamBytesMsg* msg){
			ssize_t target_pe = _pickTargetPE();
			if(target_pe == -1){
				// no one to send data to
				_buffered_msg_to_deliver.push_back(msg);
				_put_buffer_size = 0;
				return;
			}
			// insert sending code here
			CkpvAccess(stream_manager) -> sendDeliverMsg(msg, target_pe);
			_put_buffer_size = 0;
			_counter.processOutgoingMessage(msg -> num_bytes, target_pe);
		}

		ssize_t StreamBuffers::_pickTargetPE(){
			#define CHANCE_OF_LOCAL 4
			// insert some random picking logic
			if(!_registered_pe){
				// there is no registered PEs, you just return -1 aka no one to send it to
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

		void StreamBuffers::handleGetRequest(GetRequest& gr){
			if (!_registered_pe) {
				_registered_pe = true;
				starter.addRegisteredPE(_stream_id, CkMyPe());
			}
			// if the stream is closed, we do something? f
			// if we don't have enough data, then we say "fuck" and buffer it (assuming the stream isn't closed)
			if(_get_buffer_size < gr.requested_bytes) {
				_buffered_gets.push_back(gr);	
				return;
			}
			// if we have enough data, then we fullfill the request
			fulfillRequest(gr);
		}

		void StreamBuffers::insertToStream(char* data, size_t num_bytes){
			// check to see if we should flush the buffer
			if(_put_buffer_size + num_bytes > _put_buffer_capacity) {
				// flush the current buffer and the new data in one shot
				flushOutBuffer(data, num_bytes);
				_put_buffer_size = 0;
				return;
			}
			// can fit in the buffer still, so we just memcpy the data
			std::memcpy(_put_buffer + _put_buffer_size, data, num_bytes);
			_put_buffer_size += num_bytes;
		}

		void StreamBuffers::addToRecvBuffer(DeliverStreamBytesMsg* data){
			// wrap it in a InData object and then push to the get_queue
			_counter.processIncomingMessage(data -> sender_pe, data -> num_bytes);
			InData in_data(data, data -> num_bytes);
			_get_buffer.push_back(in_data);
			_get_buffer_size += data -> num_bytes;

			if(_counter.receivedAllData()){
				CkPrintf("On PE[%d], the StreamManager has received all %zu messages\n", CkMyPe(), _counter.getNumberOfExpectedReceives());
			}
			// process all the buffered get requests when new data comes in
			clearBufferedGetRequests();
		}
		// this is called if the stream is closed and we have buffered requests
		// or we have a request and enough data to serve it
		void StreamBuffers::fulfillRequest(GetRequest& gr){
			size_t num_bytes_to_copy = std::min(gr.requested_bytes, _get_buffer_size);
			size_t num_bytes_copied = 0;
			// we already know that we have enough data to satisfy stuff
			StreamDeliveryMsg* res = new (num_bytes_to_copy) StreamDeliveryMsg(_stream_id);
			while(!_get_buffer.empty()) {
				InData& front = _get_buffer.front();
				if(front.num_bytes_rem <= num_bytes_to_copy){
						std::memcpy(res -> data + num_bytes_copied, front.curr + front.num_bytes_rem, front.num_bytes_rem);
						num_bytes_copied += front.num_bytes_rem;
						num_bytes_to_copy -= front.num_bytes_rem;
						_get_buffer.pop_front();
				} else {
					std::memcpy(res -> data + num_bytes_copied, front.curr + front.num_bytes_rem, num_bytes_to_copy);
					num_bytes_to_copy -= num_bytes_to_copy;
					num_bytes_copied += num_bytes_to_copy;
					front.curr += num_bytes_to_copy;
					front.num_bytes_rem -= num_bytes_to_copy;
					break;
				}
			}
			// we now have all the data, now we send it
			_get_buffer_size -= num_bytes_copied;
			res -> num_bytes = num_bytes_copied;
			// if we know no more data is coming in, received all the data we should, and nothing left in buffer, mark stream as closed in the message
			if(_counter.receivedAllData() && !_get_buffer_size){
				res -> status = StreamStatus::STREAM_CLOSED;
			} else {
				res -> status = StreamStatus::STREAM_OK;
			}
			CkCallback cb = gr.cb;
			cb.send(res);
		}


		void StreamBuffers::clearBufferedGetRequests(){
			while(!_buffered_gets.empty()){
				GetRequest& fr = _buffered_gets.front();
				if(!_counter.receivedAllData() && _get_buffer_size < fr.requested_bytes) return; // not enough bytes to satisfy front of queue
				_buffered_gets.pop_front();
				fulfillRequest(fr);
			}
		}

		// TODO: make it so it injects the message such that at first if one pe is registered, not all messages will go to this one pe
		void StreamBuffers::clearBufferedDeliveryMsg() {
			while(!_buffered_msg_to_deliver.empty()) {
				popFrontMsgOutBuffer();
			}
		}

		void StreamBuffers::pushBackRegisteredPE(size_t pe){
			_registered_pes.push_back(pe);
		}

		size_t StreamBuffers::numBufferedDeliveryMsg(){
			return _buffered_msg_to_deliver.size();
		}

		StreamCoordinator::StreamCoordinator() = default;
		
		StreamCoordinator::StreamCoordinator(StreamToken stream) : _stream(stream) {}

		void StreamCoordinator::registerThisPE(size_t pe){
			_meta_data._registered_pes.push_back(pe);
		}

		StreamMessageCounter::StreamMessageCounter() = default;

		StreamMessageCounter::StreamMessageCounter(StreamToken stream) : _stream(stream) {}

		size_t StreamMessageCounter::getNumberOfExpectedReceives() {
			return _number_of_expected_receives;
		}
		
		void StreamMessageCounter::setExpectedReceives(size_t num_expected_receives){
			setStreamWriteClosed();
			_number_of_expected_receives = num_expected_receives;
		}


		bool StreamMessageCounter::receivedAllData(){
			size_t total_received_messages = totalReceivedMessages();
			return _close_initiated && (total_received_messages == _number_of_expected_receives);
		}

		size_t StreamMessageCounter::totalReceivedMessages(){
			size_t sum = 0;
			for(auto& p: _received_counter){
				sum += p.second;
			}
			return sum;
		}

		// This will depend on if we reall
		u_long* StreamMessageCounter::getSentCounterArray() {
			u_long* sent_arr = new u_long[CkNumPes()];
			for(int i = 0; i < CkNumPes(); ++i){
				sent_arr[i] = _sent_counter[i];
			}
			return sent_arr;
		}

		u_long* StreamMessageCounter::getReceivedCounterArray(){
			u_long* received_arr = new u_long[CkNumPes()];
			for(int i = 0; i < CkNumPes(); ++i){
				received_arr[i] = _received_counter[i];
			}
			return received_arr;
		}


		void StreamMessageCounter::setStreamWriteClosed(){
			_close_initiated = true;
		}

		void StreamMessageCounter::processIncomingMessage(size_t num_bytes, size_t src_pe){
			_received_counter[src_pe] += num_bytes;
		}

		void StreamMessageCounter::processOutgoingMessage(size_t num_bytes, size_t dest_pe){
			_sent_counter[dest_pe] += num_bytes;
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
			CkpvAccess(stream_manager) -> initiateWriteStreamClose(stream);
		}
	}

	void dummyFunction(){
		// impl::dummyImplFunction();
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

	void flushLocalStream(StreamToken stream){
		impl::impl_flushLocalStream(stream);
	}

	void get(StreamToken stream, size_t elem_size, size_t num_elems, CkCallback cb){
		impl::impl_get(stream, elem_size, num_elems, cb);
	}

	void closeWriteStream(StreamToken stream){
		impl::impl_closeWriteStream(stream);
		return;
	}


}
}

#include "CkStream.def.h"
#include "CkStream_impl.def.h"
