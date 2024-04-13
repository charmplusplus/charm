#include "CkStream.decl.h"
#include "CkStream_impl.decl.h"
#include <iostream>

namespace Ck { namespace Stream {
	namespace impl {
		CProxy_Starter starter;
		CkpvDeclare(StreamManager*, stream_manager);
	}
	namespace impl {
		class Starter : public CBase_Starter {
			CProxy_StreamManager stream_managers;
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
			}

		};

		class StreamManager : public CBase_StreamManager {
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
		};

		void dummyImplFunction(){
			CkpvAccess(stream_manager)-> sayHello();	
		}
	}
	void dummyFunction(){
		impl::dummyImplFunction();
		std::cout << "about to try execute starterHello()\n";
		impl::starter.starterHello();
		std::cout << "should have finished starterHello()\n";
	}

}
}





#include "CkStream.def.h"
#include "CkStream_impl.def.h"
