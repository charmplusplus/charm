#include "stream.h"

#include <iostream>
#include <unordered_map>
#include <cstring>

namespace Ck { namespace Stream {
	namespace impl {
		CProxy_Starter starter;
		CkpvDeclare(StreamManager*, stream_manager);
	}
	namespace impl {

		class Starter : public CBase_Starter {
			CProxy_StreamManager stream_managers;
			size_t _curr_stream_token_id = 0;
			Starter_SDAG_CODE
		public:
			Starter(CkArgMsg* m){
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

		};

		class StreamManager : public CBase_StreamManager {
			std::unordered_map<StreamToken, StreamBuffers> _stream_table;
			StreamManager_SDAG_CODE
		public:
				StreamManager(){
					CkPrintf("Stream Manager created on PE=%d\n", CkMyPe());
					CkpvInitialize(StreamManager*, stream_manager);
					CkpvAccess(stream_manager) = this;
				}

				StreamManager(CkMigrateMessage* m) : CBase_StreamManager(m) {}
  				void pup(PUP::er& p){

				}

				void sayHello(){
					CkPrintf("StreamManager %d says hello!\n", CkMyPe());
				}

				void initializeStream(int id){
					_stream_table[id] = StreamBuffers(id);
					CkPrintf("On %d, we created stream %d\n", CkMyPe(), id);
					contribute(sizeof(id), &id, CkReduction::max_int, CkCallback(CkReductionTarget(Starter, streamCreated), starter));
				}

				void putToStream(StreamToken stream, void* data, size_t elem_size, size_t num_elems){
					StreamBuffers& buffer = _stream_table[stream];	
					buffer.insertToStream((char*) data, elem_size * num_elems);
				}

				void recvData(DeliverStreamBytesMsg* in_msg){
					CkPrintf("Stream Manager %d received data for stream %d\n", CkMyPe(), in_msg -> stream_id);
				}

				void flushLocalStream(StreamToken stream){
					StreamBuffers& local_buffer = _stream_table[stream];
					local_buffer.flushOutBuffer();
				}

				inline void sendDeliverMsg(DeliverStreamBytesMsg* in_msg, size_t target_pe){
					thisProxy[target_pe].recvData(in_msg);
				}

		};

		void dummyImplFunction(){
			CkpvAccess(stream_manager)-> sayHello();	
		}
		StreamBuffers::StreamBuffers() = default;

		StreamBuffers::StreamBuffers(size_t stream_id){
			_stream_id = stream_id;
			_in_buffer = new char[_in_buffer_size];
			_out_buffer = new char[_out_buffer_size];
		}

		StreamBuffers::StreamBuffers(size_t stream_id, size_t in_buffer_capacity, size_t out_buffer_capacity){
			_stream_id = stream_id;
			_in_buffer_capacity = in_buffer_capacity;
			_out_buffer_capacity = out_buffer_capacity;
			_in_buffer = new char[_in_buffer_capacity];
			_out_buffer = new char[_out_buffer_capacity];
		}

		void StreamBuffers::flushOutBuffer(){
			if(!_in_buffer_size) return;
			_sendOutBuffer(_in_buffer, _in_buffer_size);	
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
			size_t target_pe = _pickTargetPE();
			// insert sending code here
			//
			CkpvAccess(stream_manager) -> sendDeliverMsg(msg, target_pe);
			_in_buffer_size = 0;
		}

		size_t StreamBuffers::_pickTargetPE(){
			// insert some random picking logic
			return 0;
		}

		void StreamBuffers::insertToStream(char* data, size_t num_bytes){
			if(num_bytes + _in_buffer_size > _in_buffer_capacity){
				flushOutBuffer(data, num_bytes);
				return;
			}
			std::memcpy(_in_buffer + _in_buffer_size, data, num_bytes);
			_in_buffer_size += num_bytes;
			if(_in_buffer_size == _in_buffer_capacity){
				flushOutBuffer();
			}
		}

		inline void impl_put(StreamToken stream, void* data, size_t elem_size, size_t num_elems){
			CkpvAccess(stream_manager) -> putToStream(stream, data, elem_size, num_elems);		
		}
		inline void impl_flushLocalStream(StreamToken stream){
			CkpvAccess(stream_manager) -> flushLocalStream(stream);
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

}
}





#include "CkStream.def.h"
#include "CkStream_impl.def.h"
