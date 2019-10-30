#ifndef ATA_DATATYPE
#define ATA_DATATYPE

#define DATA_ITEM_SIZE 32

struct DataItem {
public:
  char data[DATA_ITEM_SIZE];
};
PUPbytes(DataItem)

#endif
