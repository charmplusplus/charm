#include "armci_impl.h"

void *stridedCopy(void *base, void *buffer_ptr,
		  int *stride, int *count, 
		  int dim_id, bool flatten) {
  if (dim_id == 0) {
    if (flatten) {
      memcpy(buffer_ptr, base, count[dim_id]);
    } else {
      memcpy(base, buffer_ptr, count[dim_id]);
    }
    return (void *)((char *)buffer_ptr + count[dim_id]);
  } else {
    for (int i=0; i<count[dim_id]; i++) {
      buffer_ptr = stridedCopy(base, buffer_ptr, stride, 
			       count, dim_id-1, flatten);
      base = (void *)((char *)base + stride[dim_id-1]);
    }
    return buffer_ptr;
  }
}

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
