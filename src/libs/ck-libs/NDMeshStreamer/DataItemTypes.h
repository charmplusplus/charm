#ifndef DATA_ITEM_TYPES_H
#define DATA_ITEM_TYPES_H

#define CHUNK_SIZE 256

template<class dtype, class itype>
class ArrayDataItem {

public:
  itype arrayIndex;
  dtype dataItem;

  ArrayDataItem(itype i, const dtype d) : arrayIndex(i), dataItem(d) {}
};

class ChunkDataItem {

public:
  short chunkSize;
  int sourcePe; 
  int chunkNumber; 
  int numChunks;  
  int numItems;
  char rawData[CHUNK_SIZE];
  
  ChunkDataItem& operator=(const ChunkDataItem &rhs) {
    
    if (this != &rhs) {      
      chunkSize = rhs.chunkSize; 
      sourcePe = rhs.sourcePe;
      chunkNumber = rhs.chunkNumber; 
      numChunks = rhs.numChunks;
      numItems = rhs.numItems;
      memcpy(rawData, rhs.rawData, CHUNK_SIZE);
    }

    return *this;
  }
  
};

#endif
