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
				
				void addRegisteredPE(StreamToken token, size_t pe){
					StreamBuffers& sb = _stream_table[token];
					if(pe != CkMyPe()) 
						sb.pushBackRegisteredPE(pe);
					CkPrintf("just received a addRegisteredPE call to add %d on PE=%d\n", pe, CkMyPe());
					if(sb.numBufferedDeliveryMsg()){
						CkPrintf("Time to clear out the buffered DeliverStreamBytesMsg to be delivered from PE=%d\n", CkMyPe());
						thisProxy[CkMyPe()].clearBufferedDeliveryMsg(token);	
					}
				}

				void clearBufferedDeliveryMsg(StreamToken token){
					StreamBuffers& sb = _stream_table[token];	
					sb.popFrontMsgOutBuffer();
					if(sb.numBufferedDeliveryMsg()){
						CkPrintf("Scheduling another clearBufferDeliverMsg because number of buffered outgoing delivery messages is %d\n", sb.numBufferedDeliveryMsg());
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

				void recvData(DeliverStreamBytesMsg* in_msg){
					CkPrintf("Stream Manager %d received data for stream %d\n", CkMyPe(), in_msg -> stream_id);
					StreamBuffers& buff = _stream_table[in_msg -> stream_id];
					char* temp_data = in_msg -> data;
					CkPrintf("Just before delete\n");
					delete in_msg;
					buff.addToRecvBuffer(in_msg);
				}

				void flushLocalStream(StreamToken stream){
					StreamBuffers& local_buffer = _stream_table[stream];
					local_buffer.flushOutBuffer();
				}

				inline void sendDeliverMsg(DeliverStreamBytesMsg* in_msg, ssize_t target_pe){
					thisProxy[target_pe].recvData(in_msg);
				}

		};

		void dummyImplFunction(){
			CkpvAccess(stream_manager)-> sayHello();	
		}
		StreamBuffers::StreamBuffers() = default;

		StreamBuffers::StreamBuffers(size_t stream_id){
			_stream_id = stream_id;
			_in_buffer = new char[_in_buffer_capacity];
		}

		StreamBuffers::StreamBuffers(size_t stream_id, size_t coordinator_pe){
			_stream_id = stream_id;
			_coordinator_pe = coordinator_pe;
			_in_buffer = new char[_in_buffer_capacity];
		}

		StreamBuffers::StreamBuffers(size_t stream_id, size_t in_buffer_capacity, size_t out_buffer_capacity){
			_stream_id = stream_id;
			_in_buffer_capacity = in_buffer_capacity;
			_out_buffer_capacity = out_buffer_capacity;
			_in_buffer = new char[_in_buffer_capacity];
		}

		StreamBuffers::StreamBuffers(size_t stream_id, size_t coordinator_pe, size_t in_buffer_capacity, size_t out_buffer_capacity){
			_stream_id = stream_id;
			_coordinator_pe = coordinator_pe;
			_in_buffer_capacity = in_buffer_capacity;
			_out_buffer_capacity = out_buffer_capacity;
			_in_buffer = new char[_in_buffer_capacity];
		}

		void StreamBuffers::flushOutBuffer(){
			if(!_in_buffer_size) return;
			_sendOutBuffer(_in_buffer, _in_buffer_size);	
		}

		void StreamBuffers::popFrontMsgOutBuffer(){
			if(_msg_out_buffer.empty()){
				CkPrintf("The msg out buffer is empty; not doing anything\n");
				return;
			}
			CkPrintf("Number of elements in _msg_out_buffer: %d\n", _msg_out_buffer.size());
			DeliverStreamBytesMsg* msg = _msg_out_buffer.front();
			_msg_out_buffer.pop_front();
			ssize_t target_pe = _pickTargetPE();
			CkPrintf("Sending to %d from %d in popFrontMsgOutBuffer\n", target_pe, CkMyPe());
			CkpvAccess(stream_manager) -> sendDeliverMsg(msg, target_pe);
		}

		void StreamBuffers::flushOutBuffer(char* extra_data, size_t extra_bytes){
			if(!_in_buffer_size) return;
			char* bigger_buffer = new char[_in_buffer_size + extra_bytes];
			std::memcpy(bigger_buffer, _in_buffer, _in_buffer_size);
			std::memcpy(bigger_buffer + _in_buffer_size, extra_data, extra_bytes);
			_sendOutBuffer(bigger_buffer, _in_buffer_size + extra_bytes);
			delete[] bigger_buffer;
		}

		void StreamBuffers::_sendOutBuffer(char* data, size_t size){
			DeliverStreamBytesMsg* msg = new (size) DeliverStreamBytesMsg(data,size);
			msg -> stream_id = _stream_id;
			ssize_t target_pe = _pickTargetPE();
			if(target_pe == -1){
				CkPrintf("no valid pes to deliver message to, buffering send\n");
				_msg_out_buffer.push_back(msg);
				_in_buffer_size = 0;
				return;
			}
			// insert sending code here
			CkpvAccess(stream_manager) -> sendDeliverMsg(msg, target_pe);
			_in_buffer_size = 0;
			_num_sent_messages++;
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
				_buffered_reqs.push_back(gr);
			} else {
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
			if(!_out_buffer_capacity || ((_out_buffer_size + num_bytes) <= _out_buffer_capacity)){
				_out_buffer.push_back(InData(data, num_bytes));
				_out_buffer_size += num_bytes;
				CkPrintf("Just added %zu bytes to the recv stream of token=%zu on pe=%d\n", num_bytes, _stream_id, CkMyPe());
			} else {
				CkPrintf("capacity has been reached on the recv buffer, so dropping incoming message\n");
			}
			while(!_out_buffer.empty() && !_buffered_reqs.empty()){ // keep fulfilling buffered requests in FIFO order
				GetRequest& gr = _buffered_reqs.front();
				if(gr.requested_bytes > _out_buffer_size){
					break;
				}
				GetRequest curr_req = gr;
				_buffered_reqs.pop_front();
				fulfillRequest(curr_req);
			}
		}

		void StreamBuffers::fulfillRequest(GetRequest& gr){
			size_t num_bytes_copied = 0;
			size_t num_bytes_requested = std::min(gr.requested_bytes, _out_buffer_size);
			StreamDeliveryMsg* res = new (num_bytes_requested) StreamDeliveryMsg(_stream_id);

			while(!_out_buffer.empty() && (num_bytes_copied != num_bytes_requested)){
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
			// copied all of the data we could
			res -> num_bytes = num_bytes_copied;
			gr.cb.send(res);
			return;
		}

		void StreamBuffers::pushBackRegisteredPE(size_t pe){
			_registered_pes.push_back(pe);
		}

		size_t StreamBuffers::numBufferedDeliveryMsg(){
			return _msg_out_buffer.size();
		}

		StreamCoordinator::StreamCoordinator() = default;
		
		StreamCoordinator::StreamCoordinator(StreamToken stream) : _stream(stream) {}

		void StreamCoordinator::registerThisPE(size_t pe){
				_meta_data._registered_pes.push_back(pe);
				CkpvAccess(stream_manager) -> addRegisteredPE(_stream, pe);
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

	void flushLocalStream(StreamToken stream){
		impl::impl_flushLocalStream(stream);
	}

	void get(StreamToken stream, size_t elem_size, size_t num_elems, CkCallback cb){
		impl::impl_get(stream, elem_size, num_elems, cb);
	}


}
}

#include "CkStream.def.h"
#include "CkStream_impl.def.h"
