#ifndef DATA_ITEM_TYPES_H
#define DATA_ITEM_TYPES_H

#define CHUNK_SIZE 256

template<class dtype, class itype>
class ArrayDataItem {

public:
  itype arrayIndex;
  int sourcePe;
  dtype dataItem;

  ArrayDataItem(itype i, int srcPe, const dtype d)
    : arrayIndex(i), sourcePe(srcPe), dataItem(d) {}
};

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

template <class dtype, class ClientType>
inline int defaultMeshStreamerDeliver(char *data, void *clientObj_)
{
  ((ClientType *) clientObj_)->process(*((dtype *) data));
  return 0;
}

#endif
