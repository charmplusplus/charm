#ifndef DATA_ITEM_TYPES_H
#define DATA_ITEM_TYPES_H

#include <utility>
#include "ckarrayindex.h" // For CkArrayIndex in DataItemHandle

#define CHUNK_SIZE 256

template<class dtype, class itype>
class ArrayDataItem {

public:
  itype arrayIndex;
  int sourcePe;
  dtype dataItem;

  ArrayDataItem(){}

  ArrayDataItem(itype i, int srcPe, const dtype d)
    : arrayIndex(i), sourcePe(srcPe), dataItem(d) {}

  void pup(PUP::er &p) {
    p|arrayIndex;
    p|sourcePe;
    p|dataItem;
  }
};

template <typename dtype, typename itype>
void operator|(PUP::er &p, ArrayDataItem<dtype, itype>& obj) {
   obj.pup(p);
}

class ChunkDataItem {

public:
  short chunkSize;
  int bufferNumber;
  int sourcePe; 
  int chunkNumber; 
  int numChunks;  
  int numItems;
  char rawData[CHUNK_SIZE];
  
  ChunkDataItem& operator=(const ChunkDataItem &rhs) {
    
    if (this != &rhs) {      
      chunkSize = rhs.chunkSize; 
      bufferNumber = rhs.bufferNumber;
      sourcePe = rhs.sourcePe;
      chunkNumber = rhs.chunkNumber; 
      numChunks = rhs.numChunks;
      numItems = rhs.numItems;
      memcpy(rawData, rhs.rawData, CHUNK_SIZE);
    }

    return *this;
  }
  
};

template <class dtype>
struct DataItemHandle {
  CkArrayIndex arrayIndex;
  const dtype *dataItem;

  DataItemHandle(dtype* _ptr, CkArrayIndex _idx = CkArrayIndex()) : dataItem(_ptr), arrayIndex(_idx) {}
};

template <class dtype, class ClientType>
inline int defaultMeshStreamerDeliver(char *data, void *clientObj_)
{
  ((ClientType *) clientObj_)->process(*((dtype *) data));
  return 0;
}

#endif
