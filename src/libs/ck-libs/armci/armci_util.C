#include "armci_impl.h"

// malloc reduction client
void mallocClient(void *param, int datasize, void *data) {
  // get the array proxy from the supplied aid value (at setup)
  CProxy_ArmciVirtualProcessor vpProxy = 
    CProxy_ArmciVirtualProcessor(*(CkArrayID *)param);
  int numBlocks = datasize/sizeof(addressPair);
  addressPair *dataBlocks = (addressPair *)data;

  AddressMessage *msg = new(numBlocks, 0) AddressMessage;
  // constructing the ordered set of shared pointers
  for (int i=0; i<numBlocks; i++) {
    msg->addresses[dataBlocks[i].pe] = dataBlocks[i].ptr;
    /*
    ckout << "dest" << i << ": " << dataBlocks[i].pe 
	  << "->" << dataBlocks[i].ptr << endl;
    */
  }
  // broadcast the results to everyone.
  vpProxy.getAddresses(msg);
}
