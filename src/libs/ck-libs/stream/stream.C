#include "stream.h"
#include <iostream>
#include <unordered_map>
#include <cstring>
#include <time.h>
#include <stdlib.h>

typedef unsigned long ulong;

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
// 			CkPrintf("--- PE #%d: Constructing InData Object:", CkMyPe());
			for (int i = 0; i < num_bytes / sizeof(size_t); ++i) {
// 				CkPrintf("%zu, ", ((size_t*)msg->data)[i]);
			}
// 			CkPrintf("-----\n");
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
// 				CkPrintf("The starter is alive\n");
				starter = thisProxy;
				stream_managers = CProxy_StreamManager::ckNew();
			}

			Starter(CkMigrateMessage* m) : CBase_Starter(m) {}
  			void pup(PUP::er& p){

			}

			void starterHello(){
// 				CkPrintf("Hello from the starter chare of CkStream!\n");
// 				std::cout << "this is a print from the std::cout library\n";
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
				ulong* totalMessageSentToPEs = (ulong*)results[0].data;
				StreamToken st = *((size_t*)results[1].data);
				for(int i = 0; i < CkNumPes(); ++i){
					// invoke some entry method
// 					CkPrintf("From global starter, PE[%d] should expected %zu messages\n", i, totalMessageSentToPEs[i]);
					stream_managers[i].expectedReceivesUponClose(st, size_t(totalMessageSentToPEs[i]));
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
// 					CkPrintf("Stream Manager created on PE=%d\n", CkMyPe());
					CkpvInitialize(StreamManager*, stream_manager);
					CkpvAccess(stream_manager) = this;
				}

				StreamManager(CkMigrateMessage* m) : CBase_StreamManager(m) {}
				
				void startWriteStreamClose(StreamToken token){
					thisProxy.initiateWriteStreamClose(token);
				}

				void clearBufferedDeliveryMsg(StreamToken token){
					StreamBuffers& sb = _stream_table[token];
					sb.clearBufferedDeliveryMsg();	
				}

  				void pup(PUP::er& p){
					// do I bother populating this?
				}

				void expectedReceivesUponClose(StreamToken token, size_t num_messages_to_receive){
					StreamBuffers& sb = _stream_table[token];
					sb.setExpectedReceivesUponClose(num_messages_to_receive);
				}

				void initializeStream(int id, size_t coordinator_pe){
					_stream_table[id] = StreamBuffers(id, coordinator_pe);
// 					CkPrintf("On %d, we created stream %d with coordinator=%d\n", CkMyPe(), id, coordinator_pe);
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
					CkPrintf("CALLING FLUHSOUT ON LOCAL\n");
					local_buffer.flushOutBuffer();
					CkPrintf("DONE CALLING FLUHSOUT ON LOCAL\n");

				}

				inline void sendDeliverMsg(DeliverStreamBytesMsg* msg_to_send, ssize_t target_pe){
					// invoke the entry method for sending "in_msg" to the correct PE
					thisProxy[target_pe].recvPutBufferFromPE(msg_to_send);
				}

				void initiateWriteStreamClose(StreamToken token){
// 					CkPrintf("Stream Manager %d received a close request for stream=%d\n", CkMyPe(), token);
					StreamBuffers& sb = _stream_table[token];
					sb.setCloseFlag();
					if(sb.numBufferedDeliveryMsg()){
// 						CkPrintf("there are still messages that are buffered that have to be delivered on PE %d\n", CkMyPe());	
						return;
					}
					closeStreamBuffer(token);
				}
				
				// used to actually close the stream buffer by creating the reduction to send to the Starter
				void closeStreamBuffer(StreamToken token){
// 					CkPrintf("PE[%d] has invoked the closeStreamBuffer method...\n", CkMyPe());
					StreamBuffers& sb = _stream_table[token];
					CkReductionMsg* msg = sb.setStreamClosed();
					contribute(msg);
				}

				void addRegisteredPE(StreamToken stream, size_t pe){
					StreamBuffers& sb = _stream_table[stream];
					sb.pushBackRegisteredPE(pe);
// 					CkPrintf("From StreamManager[%d], just added pe=%zu\n", CkMyPe(), pe);
					// TODO: change this to do message injection instead of just emptying the entire buffered delivery messages
					sb.clearBufferedDeliveryMsg();		
				}

				void serveGetRequest(StreamToken stream, GetRequest& gr){
					CkPrintf("In serveGetRequest, service get request on stream id %zu\n", stream);
					StreamBuffers& local_buffer = _stream_table[stream];
					local_buffer.handleGetRequest(gr);
				}

		};

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

		void StreamBuffers::clearBufferedDeliveryMsg(){
			if(_buffered_msg_to_deliver.empty()){
// 				CkPrintf("clearBufferedDeliverMsg was called when there's nothing there in the queue...\n");
				return;
			}
			while(!_buffered_msg_to_deliver.empty()){
				popFrontMsgOutBuffer();
			}
			// if the closing process has begun but was waiting on sending all the buffered messages, resume the closing process
			if(_counter.isCloseFlagSet()){
// 				CkPrintf("clearBufferedDeliveryMsg: the stream closed flag is set, so we can proceed with the stream closing operations\n");
				CkpvAccess(stream_manager) -> closeStreamBuffer(_stream_id);
			}
		}
		
		void StreamBuffers::setExpectedReceivesUponClose(size_t num_messages_to_receive){
// 			CkPrintf("inside setExpectedReceivesUponClose on PE[%d]\n", CkMyPe());
			_counter.setExpectedReceives(num_messages_to_receive);
			// if we have already received the numbero f messages, we just mark ourselves as closed
			if(_counter.receivedAllData()){
// 				CkPrintf("On PE[%d], from within setExpectedReceivesUponClose, the StreamManager has received all %zu messages. Currently %d get requests buffered...\n", CkMyPe(), num_messages_to_receive, _buffered_gets.size());
			} else {
// 				CkPrintf("On PE[%d], the StreamManager has not receieved all %zu bytes; has only receieved %zu bytes\n", CkMyPe(), num_messages_to_receive, _counter.totalReceivedMessages());
			}
			clearBufferedGetRequests();
		}

		CkReductionMsg* StreamBuffers::setStreamClosed(){
// 			CkPrintf("flush on PE[%d] has started...\n", CkMyPe());
			flushOutBuffer();
// 			CkPrintf("flush on PE[%d] has finished\n", CkMyPe());
			// send the coordinator how many bytes were sent
			u_long* sent_arr = _counter.getSentCounterArray();
			// print out the stuff
			for(int i = 0; i < CkNumPes(); ++i){
// 				CkPrintf("DEBUG CLOSING: From PE[%d], sent_arr[%d]=%zu\n", CkMyPe(), i, sent_arr[i]);
			}
			// original type is size_t, but changing it to int for the reducer. Do I need to do this?
			ulong st = _stream_id;
			// make a tuple reduction
			CkReduction::tupleElement tupleRedn[] = {
				CkReduction::tupleElement(CkNumPes() * sizeof(ulong), sent_arr, CkReduction::sum_int),
				CkReduction::tupleElement(CkNumPes() * sizeof(ulong), &st, CkReduction::max_ulong)
			};
			int tuple_size = 2;
			CkReductionMsg* msg = CkReductionMsg::buildFromTuple(tupleRedn, tuple_size);
			CkCallback cb(CkIndex_Starter::tellManagersExpectedReceives(0), starter);
			msg -> setCallback(cb);
			return msg;
		}

		size_t StreamBuffers::coordinator() {
			return _coordinator_pe;
		}

		void StreamBuffers::flushOutBuffer(){
			if(!_put_buffer_size) {
// 				CkPrintf("StreamBuffers::flushOutBuffer returning because put_buffer_size is 0\n");
				return;
			}
			DeliverStreamBytesMsg* msg = createDeliverBytesStreamMsg();
			CkPrintf("StreamBuffers::flushOutBuffer sending out message\n");
			_sendOutBuffer(msg);	
			_counter.printSentCounter();

		}

		void StreamBuffers::popFrontMsgOutBuffer(){
			if(_buffered_msg_to_deliver.empty()){
				return;
			}
			DeliverStreamBytesMsg* msg = _buffered_msg_to_deliver.front();
			_buffered_msg_to_deliver.pop_front();
			ssize_t target_pe = _pickTargetPE();
			if (target_pe == -1){
// 				CkPrintf("how are we able to have -1 as a target_pe when the popFrontMessage method is invoked...\n");
			}
			size_t num_bytes = msg -> num_bytes;
			CkpvAccess(stream_manager) -> sendDeliverMsg(msg, target_pe);
			_counter.processOutgoingMessage(num_bytes, target_pe);
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
			CkPrintf("IN _SENDOUTBUFFER\n ");
			ssize_t target_pe = _pickTargetPE();
			CkPrintf("Target Pe: %d\n ", target_pe);

			if(target_pe == -1){
				// no one to send data to
// 				CkPrintf("buffering the data now...\n");
				_buffered_msg_to_deliver.push_back(msg);
				_put_buffer_size = 0;
				CkPrintf("no target pes, buffering the delivery...\n");
				return;
			}
			// insert sending code here
			CkPrintf("processOutgoingMessage about to begin...\n");
			_counter.processOutgoingMessage(msg -> num_bytes, target_pe);
// 			CkPrintf("processOutgoingMessage complete. Starting sendDeliverMsg...\n");
			CkpvAccess(stream_manager) -> sendDeliverMsg(msg, target_pe);
// 			CkPrintf("sendDeliverMsg is complete.\n");
			_put_buffer_size = 0;
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
			if(!_registered_pe){
				_registered_pe = true;
				// register yourself with the system
				starter.addRegisteredPE(_stream_id, CkMyPe());

			}
			// if the stream is closed, we do something?
			if(_counter.receivedAllData()){
// 				CkPrintf("From PE[%d], receivedAllData\n", CkMyPe());

			}

			CkPrintf("recveivedAllData value %d\n", _counter.receivedAllData());
			// if we don't have enough data, then we say "fuck" and buffer it (assuming the stream isn't closed)
			if(!_counter.receivedAllData() && _get_buffer_size < gr.requested_bytes) {
				_buffered_gets.push_back(gr);	
// 				CkPrintf("From PE[%d], buffering a get request with %d requested bytes...\n", CkMyPe(), gr.requested_bytes);
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
			// DEBUG PRINT
			// CkPrintf("--- PE #%d: In addToRecvBuffer:", CkMyPe());
			// for (int i = 0; i < data->num_bytes / sizeof(size_t); ++i) {
			// 	CkPrintf("%zu, ", ((size_t*)data->data)[i]);
			// }
			// CkPrintf("-----\n");


			_counter.processIncomingMessage(data -> num_bytes, data -> sender_pe);
			InData in_data(data, data -> num_bytes);
			_get_buffer.push_back(in_data);
			_get_buffer_size += data -> num_bytes;
// 			CkPrintf("PROCESS THE INCOMING MESSAGE: PE[%d] has receieved %d bytes from %d. Total bytes received: %zu\n", CkMyPe(), data -> num_bytes, data -> sender_pe, _counter.totalReceivedMessages());

			if(_counter.receivedAllData()){
// 				CkPrintf("On PE[%d], within addToRecvBuffer, the StreamManager has received all %zu messages. Currently %d get requests buffered...\n", CkMyPe(), _counter.getNumberOfExpectedReceives(), _buffered_gets.size());
			}
			// process all the buffered get requests when new data comes in
			clearBufferedGetRequests();
		}
		// this is called if the stream is closed and we have buffered requests
		// or we have a request and enough data to serve it
		void StreamBuffers::fulfillRequest(GetRequest& gr){
// 			CkPrintf("+++++++++++++\n");
			CkPrintf("PE #%d: in fulfillRequest: gr.requested_bytes=%zu, _get_buffer_size=%zu, _buffered_gets.size():%zu\n", CkMyPe(), gr.requested_bytes, _get_buffer_size, _buffered_gets.size());
			size_t num_bytes_to_copy = std::min(gr.requested_bytes, _get_buffer_size);
// 			CkPrintf("about to fulfill a request where num_bytes_to_copy=%zu\n", num_bytes_to_copy);
			size_t num_bytes_copied = 0;
			// we already know that we have enough data to satisfy stuff
			StreamDeliveryMsg* res = new (num_bytes_to_copy) StreamDeliveryMsg(_stream_id);
			if(_get_buffer.empty()){
				CkPrintf("FUCK SHIT BALLZ\n");
			}
			while(!_get_buffer.empty()) {
				InData& front = _get_buffer.front();
				if(front.num_bytes_rem <= num_bytes_to_copy){
						CkPrintf("fulfillRequest true branch: front.num_bytes_rem=%zu, num_bytes_to_copy=%zu\n", front.num_bytes_rem, num_bytes_to_copy);
						// CkPrintf("--- PE #%d: printing out remaining contents fo front (InData):", CkMyPe());
						for (int i = 0; i < front.num_bytes_rem / sizeof(size_t); ++i) {
// 							CkPrintf("%zu, ", ((size_t*)front.curr)[i]);
						}
// 						CkPrintf("-----\n");
						std::memcpy(res -> data + num_bytes_copied, front.curr, front.num_bytes_rem);
						num_bytes_copied += front.num_bytes_rem;
						num_bytes_to_copy -= front.num_bytes_rem;
						_get_buffer.pop_front();
				} else {
					CkPrintf("fulfillRequest else branch: front.num_bytes_rem=%zu, num_bytes_to_copy=%zu\n", front.num_bytes_rem, num_bytes_to_copy);
// 					CkPrintf("--- PE #%d: printing out remaining contents fo front (InData):", CkMyPe());
					for (int i = 0; i < front.num_bytes_rem / sizeof(size_t); ++i) {
// 						CkPrintf("%zu, ", ((size_t*)front.curr)[i]);
					}
// 					CkPrintf("-----\n");
					std::memcpy(res -> data + num_bytes_copied, front.curr, num_bytes_to_copy);
// 					CkPrintf("fulfill else branch before: num_bytes_copied=%zu\n", num_bytes_copied);
					num_bytes_copied += num_bytes_to_copy;
// 					CkPrintf("fulfill else branch after: num_bytes_copied=%zu\n", num_bytes_copied);
					front.curr += num_bytes_to_copy;
					front.num_bytes_rem -= num_bytes_to_copy;
					num_bytes_to_copy -= num_bytes_to_copy;
					break;
				}
			}
			CkPrintf("After: _buffered_gets.size():%zu\n", _buffered_gets.size());
			// we now have all the data, now we send it
			_get_buffer_size -= num_bytes_copied;
			res -> num_bytes = num_bytes_copied;
			// if we know no more data is coming in, received all the data we should, and nothing left in buffer, mark stream as closed in the message
			CkPrintf("From PE[%d], num_bytes_copied=%d, res -> num_bytes=%d, _counter.receivedAllData()=%d, _get_buffer_size=%d\n", CkMyPe(), num_bytes_copied, res -> num_bytes, _counter.receivedAllData(), _get_buffer_size);
			if(_counter.receivedAllData() && !_get_buffer_size){
				res -> status = StreamStatus::STREAM_CLOSED;
			} else {
				res -> status = StreamStatus::STREAM_OK;
			}
			gr.cb.send(res);
		}

		void StreamBuffers::clearBufferedGetRequests(){
			// clear all of the buffered get requests when enough data comes in to serve the head of queue
			while(!_buffered_gets.empty()){
				GetRequest& fr = _buffered_gets.front();
				if(!_counter.receivedAllData() && _get_buffer_size < fr.requested_bytes){// not enough bytes to satisfy front of queue
// 					CkPrintf("clearBufferedGetRequeests on PE[%d]: returning early from clearBufferedGetRequsts: _counter.receivedAllData()=%d, _get_buffer_size=%zu\n", CkMyPe(), _counter.receivedAllData(), _get_buffer_size);
					return;
				}
// 				CkPrintf("on PE[%d], fulfilling a get request..\n");
				_buffered_gets.pop_front();
				fulfillRequest(fr);
			}
		}

		void StreamBuffers::pushBackRegisteredPE(size_t pe){
			_registered_pes.push_back(pe);
		}

		size_t StreamBuffers::numBufferedDeliveryMsg(){
			return _buffered_msg_to_deliver.size();
		}

		void StreamBuffers::setCloseFlag(){
			_counter.setCloseFlag();
		}

		StreamCoordinator::StreamCoordinator() = default;
		
		StreamCoordinator::StreamCoordinator(StreamToken stream) : _stream(stream) {}

		void StreamCoordinator::registerThisPE(size_t pe){
				_meta_data._registered_pes.push_back(pe);
		}

		StreamMessageCounter::StreamMessageCounter() = default;

		StreamMessageCounter::StreamMessageCounter(StreamToken stream) : _stream(stream) {}

		bool StreamMessageCounter::isCloseFlagSet(){
			return _close_initiated;
		}

		size_t StreamMessageCounter::getNumberOfExpectedReceives() {
			return _number_of_expected_receives;
		}

		void StreamMessageCounter::setCloseFlag(){
			_close_initiated = true;
		}

		void StreamMessageCounter::setExpectedReceives(size_t num_expected_receives){
			CkPrintf("Called setExpectedReceives with value %d\n", num_expected_receives);
			_number_of_expected_receives = num_expected_receives;
		}


		bool StreamMessageCounter::receivedAllData(){
			size_t total_received_messages = totalReceivedMessages();
			CkPrintf("receivedAllData queried, total_received_messages: %d\n", total_received_messages);
			return _close_initiated && (total_received_messages == _number_of_expected_receives);
		}

		size_t StreamMessageCounter::totalReceivedMessages(){
			size_t sum = 0;
			// CkPrintf("--- sta totalReceivedMessager map: \n");
			// for(auto& p: _received_counter){
			// 	CkPrintf("PE %d: %d\n", p.first, p.second);
			// 	sum += p.second;
			// }
			// CkPrintf("--- end totalReceivedMessager map: \n");

			return sum;
		}

		// This will depend on if we reall
		u_long* StreamMessageCounter::getSentCounterArray() {
			u_long* sent_arr = new u_long[CkNumPes()];
			// loop through every PE
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
				// TODO:
		}

		void StreamMessageCounter::processIncomingMessage(size_t num_bytes, size_t src_pe){
			if(_received_counter.count(src_pe)){
				_received_counter[src_pe] += num_bytes;
			} else {
				_received_counter[src_pe] = num_bytes;
			}
		}

		void StreamMessageCounter::processOutgoingMessage(size_t num_bytes, size_t dest_pe){
			if(_sent_counter.count(dest_pe)){
				_sent_counter[dest_pe] = _sent_counter[dest_pe] + num_bytes;
			} else {
				_sent_counter[dest_pe] = num_bytes;
			}
			CkPrintf("processOutgoingMessage map:");
			for(auto& p: _sent_counter){
				CkPrintf("_sent_counter[%zu]=%zu\n", p.first, p.second);
			}
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
// 			CkPrintf("called for a close to the stream...\n");
			// if we have no registered readers, we should buffer the close request
			CkpvAccess(stream_manager) -> startWriteStreamClose(stream);
		}
	}

	void createNewStream(CkCallback cb){
		impl::starter.createNewStream(cb);
	}

	void put(StreamToken stream, void* data, size_t elem_size, size_t num_elems){
		impl::impl_put(stream, data, elem_size, num_elems);
	}

	void putRecord(StreamToken stream, void* data, size_t data_size) {
// 		CkPrintf("-=-=-=-=-=-=-ODFHUSEIRVSEOIRUYSOEIRUVIOEPir=-=\n");

		size_t total_size = sizeof(size_t) + data_size;
		char* record_buffer = new char[total_size];
		std::memcpy(record_buffer, &data_size, sizeof(size_t));
		std::memcpy(record_buffer + sizeof(size_t), data, data_size);
		size_t* tmp = (size_t*)record_buffer;
// 		CkPrintf("record buffer size: %zu\n", tmp[0]);
		
// 		CkPrintf("record buffer: %s\n", record_buffer + sizeof(size_t));

		impl::impl_put(stream, record_buffer, sizeof(char), total_size);
	}

	void flushLocalStream(StreamToken stream){
		CkPrintf("START FLUSH INVOKE\n");
		impl::impl_flushLocalStream(stream);
		CkPrintf("END FLUSH INVOKE\n");
	}

	void get(StreamToken stream, size_t elem_size, size_t num_elems, CkCallback cb){
		impl::impl_get(stream, elem_size, num_elems, cb);
	}

	// This function must be invoked by a threaded entry method
	void getRecord(StreamToken stream, CkCallback cb) {
		StreamDeliveryMsg* msg;
		impl::impl_get(stream, sizeof(size_t), 1, CkCallbackResumeThread((void*&)msg));
		size_t* record_size = (size_t*)msg->data;

		impl::impl_get(stream, sizeof(char), record_size[0], cb);
	}

	void closeWriteStream(StreamToken stream){
		impl::impl_closeWriteStream(stream);
		return;
	}


}
}

#include "CkStream.def.h"
#include "CkStream_impl.def.h"
