/*
  This file defines various reducers for sparse contiguous 2D arrays. 
  It also defines some functions to parse the result of a reduction 
  on sparse contiguous 2D arrays.

                              Vikas Mehta, vmehta1@uiuc.edu, 06/03/03
*/

#include "cksparsecontiguousreducer.h"

/* 
   various reducers supported for sparse contiguous 2D arrays
*/
CkReduction::reducerType sparse_sum_int;
CkReduction::reducerType sparse_sum_float;
CkReduction::reducerType sparse_sum_double;
CkReduction::reducerType sparse_sum_TwoFloats;
CkReduction::reducerType sparse_sum_TwoDoubles;

CkReduction::reducerType sparse_product_int;
CkReduction::reducerType sparse_product_float;
CkReduction::reducerType sparse_product_double;

CkReduction::reducerType sparse_max_int;
CkReduction::reducerType sparse_max_float;
CkReduction::reducerType sparse_max_double;

CkReduction::reducerType sparse_min_int;
CkReduction::reducerType sparse_min_float;
CkReduction::reducerType sparse_min_double;

/*
  This function returns the index of jth data segment header in the 
  array of headers pointed to by 'ptr'.
*/
//int getIndex(CkDataSegHeader *r, const unsigned char *ptr, int j);

/*
  macro defining various reducer functions
*/
#define SIMPLE_SPARSE_CONTIGUOUS_REDUCTION(name,dataType,typeStr,loop) \
static CkReductionMsg *name(int nMsg, CkReductionMsg ** msg){\
  if(nMsg==1){\
    CkReductionMsg *ret = msg[0];\
    msg[0] = NULL;\
    return ret;\
  }\
 \
  int               count          = 0;\
  CkDataSegHeader*  finalHeaderArr = NULL; /* size [count] */\
  CkDataSegHeader** msgHeaderArr   = NULL; /* size [nMsg] */\
  int*              size           = NULL; /* size [nMsg] */\
  int*              pos            = NULL; /* size [nMsg] */\
  int*              off            = NULL; /* size [nMsg] */\
  int*              arrPos         = NULL; /* size [count] */\
  int*              dataPos        = NULL; /* size [count] */\
  int*              flag           = NULL; /* size [count] */\
  int               i              = 0;\
\
  /* find the total data segments in n input msgs */\
  for(i=0; i<nMsg; i++) {\
    count += numDataSegs((unsigned char*)(msg[i]->getData()));\
  }\
\
  int nm  = ((nMsg*sizeof(int)) + \
            sizeof(CkDataSegHeader))/sizeof(CkDataSegHeader);\
  int nmp = ((nMsg*sizeof(CkDataSegHeader*)) + \
            sizeof(CkDataSegHeader))/sizeof(CkDataSegHeader);\
  int nc  = ((count*sizeof(int)) + \
            sizeof(CkDataSegHeader))/sizeof(CkDataSegHeader);\
  int numHeaderToAllocate = count + nmp + 3*nm + 3*nc;\
\
  finalHeaderArr = new CkDataSegHeader[numHeaderToAllocate];\
\
  msgHeaderArr   = (CkDataSegHeader**)(finalHeaderArr + count);\
  size           = (int*)(((CkDataSegHeader*)msgHeaderArr) + nmp);\
  pos            = (int*)(((CkDataSegHeader*)size) + nm);\
  off            = (int*)(((CkDataSegHeader*)pos) + nm);\
  arrPos         = (int*)(((CkDataSegHeader*)off) + nm);\
  dataPos        = (int*)(((CkDataSegHeader*)arrPos) + nc);\
  flag           = (int*)(((CkDataSegHeader*)dataPos) + nc);\
\
  /* put all the unique data headers from input messages to the header-array */\
\
  /* initialize temporary variables */\
  pos [0]          = 0;\
  size [0]         = numDataSegs((unsigned char*)(msg[0]->getData()));\
  off [0]          = 0;\
  msgHeaderArr [0] = getDataSegHeaderPtr((unsigned char*)(msg[0]->getData()));\
\
  for(i=1; i<nMsg; i++){\
    unsigned char * data = (unsigned char*)(msg[i]->getData());\
    pos  [i]             = 0;\
    size [i]             = numDataSegs(data);\
    off [i]              = off [i-1] + size [i-1];\
    msgHeaderArr [i]     = getDataSegHeaderPtr(data);\
  }\
\
  for(i=0; i<count; i++){\
    arrPos  [i]  = -1;\
    dataPos [i]  = -1;\
  }\
\
  /* n-way merge */\
  int nScanned    = 0;\
  int currMsg     = 0;\
  /* number of unique merged headers */\
  int totalHeader = 0;\
\
  /* find the first header */\
  int index = 0;\
  int currDataOff = 0;\
  CkDataSegHeader minHdr = msgHeaderArr [currMsg][pos[currMsg]];\
\
  for (index=1; index < nMsg; index++) {\
    if (msgHeaderArr [index][pos[index]] < minHdr)  {\
      currMsg = index;\
      minHdr = msgHeaderArr [index][pos[index]];\
    }\
  }\
\
  arrPos [off [currMsg] + pos [currMsg]] = totalHeader;\
  dataPos [off [currMsg] + pos [currMsg]] = currDataOff;\
  finalHeaderArr [totalHeader++] = msgHeaderArr [currMsg][pos[currMsg]++];\
  nScanned ++;\
\
  while (nScanned != count) {\
    if (pos [currMsg] != size [currMsg]) {\
      minHdr = msgHeaderArr [currMsg][pos[currMsg]];\
    } else {\
      currMsg = (currMsg+1)%nMsg;\
      continue;\
    }\
    for (index=0; index < nMsg; index++) {\
      if ((pos [index] != size [index])&&\
          (msgHeaderArr [index][pos[index]] < minHdr))  {\
        currMsg = index;\
        minHdr = msgHeaderArr [index][pos[index]];\
      }\
    }\
\
    if (!(finalHeaderArr [totalHeader-1] == msgHeaderArr [currMsg][pos[currMsg]])) {\
      currDataOff += sizeof(dataType)*finalHeaderArr [totalHeader-1].getNumElements();\
      arrPos [off [currMsg] + pos [currMsg]] = totalHeader;\
      dataPos [off [currMsg] + pos [currMsg]] = currDataOff;\
      finalHeaderArr [totalHeader++] = msgHeaderArr [currMsg][pos[currMsg]++];\
    }\
    else {\
      arrPos [off [currMsg] + pos [currMsg]] = totalHeader-1;\
      dataPos [off [currMsg] + pos [currMsg]] = currDataOff;\
      pos[currMsg]++;\
    }\
    nScanned ++;\
  }\
\
 /* number of non-null data blocks and total number of elements in them is\
    known. Now pack the input data into one buffer resolving the overlap.*/\
\
  int alignedHdrSize = 2*sizeof(int) + sizeof(CkDataSegHeader)*totalHeader;\
  if (alignedHdrSize%sizeof(double))\
    alignedHdrSize += sizeof(double) - (alignedHdrSize%sizeof(double));\
  int dataSize = alignedHdrSize + \
                 currDataOff + \
                 sizeof(dataType)*finalHeaderArr [totalHeader-1].getNumElements();\
\
  CkReductionMsg* m = CkReductionMsg::buildNew(dataSize, NULL); \
  unsigned char *data = (unsigned char*)(m->getData ());\
  memset (flag, 0, totalHeader*sizeof(int));\
\
  *(int*)data = totalHeader;\
  data += sizeof(int);\
  *(int*)data = alignedHdrSize;\
  data += sizeof(int);\
\
  int dataOffset = alignedHdrSize - 2*sizeof (int);\
  int numElems;\
  CkDataSegHeader* hdrPtr = (CkDataSegHeader*)data;\
  dataType*        dataptr = (dataType*)(data + dataOffset);\
\
  for (i=0; i<nMsg; i++) {\
    dataType* msgDataptr = (dataType*)getDataPtr((unsigned char*)(msg[i]->getData()));\
    for (index=0; index<size[i]; index++) {\
      numElems = msgHeaderArr [i][index].getNumElements();\
\
      hdrPtr [arrPos [off [i] + index]] = msgHeaderArr [i][index];\
      dataptr = (dataType*)(data + dataOffset + dataPos [off [i] + index]);\
      if (!flag [arrPos [off [i] + index]]) {\
        flag [arrPos [off [i] + index]] = 1;\
        memcpy(dataptr, msgDataptr, sizeof(dataType)*numElems);\
        msgDataptr += numElems;\
      } else {\
        for (int k=0; k<numElems; k++) {\
          loop\
          msgDataptr++;\
        }\
      }\
    }\
  }\
\
  delete[] finalHeaderArr;\
\
  return m;\
}

//Use this macro for reductions that have the same type for all inputs
#define SIMPLE_POLYMORPH_SPARSE_CONTIGUOUS_REDUCTION(nameBase,loop) \
  SIMPLE_SPARSE_CONTIGUOUS_REDUCTION(nameBase##_int,int,"%d",loop) \
  SIMPLE_SPARSE_CONTIGUOUS_REDUCTION(nameBase##_float,float,"%f",loop) \
  SIMPLE_SPARSE_CONTIGUOUS_REDUCTION(nameBase##_double,double,"%f",loop)

//Use this macro for reductions that have the same type for all inputs
#define POLYMORPH_SPARSE_CONTIGUOUS_REDUCTION(nameBase,loop) \
  SIMPLE_SPARSE_CONTIGUOUS_REDUCTION(nameBase##_TwoFloats,CkTwoFloats,"%d",loop) \
  SIMPLE_SPARSE_CONTIGUOUS_REDUCTION(nameBase##_TwoDoubles,CkTwoDoubles,"%f",loop) \

// Merge the sparse arrays passed by elements, summing the elements with same
// indices.
POLYMORPH_SPARSE_CONTIGUOUS_REDUCTION(_sparse_sum, *(dataptr + k) +=
  *msgDataptr;)

// Merge the sparse arrays passed by elements, summing the elements with same
// indices.
SIMPLE_POLYMORPH_SPARSE_CONTIGUOUS_REDUCTION(_sparse_sum, *(dataptr + k) +=
  *msgDataptr;)

// Merge the sparse arrays passed by elements, multiplying the elements with
// same indices.
SIMPLE_POLYMORPH_SPARSE_CONTIGUOUS_REDUCTION(_sparse_product, *(dataptr + k) *=
  *msgDataptr;)

// Merge the sparse arrays passed by elements, keeping the largest of the
// elements with same indices.
SIMPLE_POLYMORPH_SPARSE_CONTIGUOUS_REDUCTION(_sparse_max,
  if(*(dataptr+k)<*msgDataptr) *(dataptr+k)=*msgDataptr;)

// Merge the sparse arrays passed by elements, keeping the smallest of the
// elements with same indices.
SIMPLE_POLYMORPH_SPARSE_CONTIGUOUS_REDUCTION(_sparse_min,
  if(*(dataptr+k)>*msgDataptr) *(dataptr+k)=*msgDataptr;)

/*
   register supported reducers
*/
void registerReducers(void)
{
  sparse_sum_int = CkReduction::addReducer(_sparse_sum_int);
  sparse_sum_float = CkReduction::addReducer(_sparse_sum_float);
  sparse_sum_double = CkReduction::addReducer(_sparse_sum_double);
  sparse_sum_TwoFloats = CkReduction::addReducer(_sparse_sum_TwoFloats);
  sparse_sum_TwoDoubles = CkReduction::addReducer(_sparse_sum_TwoDoubles);

  sparse_product_int = CkReduction::addReducer(_sparse_product_int);
  sparse_product_float= CkReduction::addReducer(_sparse_product_float);
  sparse_product_double = CkReduction::addReducer(_sparse_product_double);

  sparse_max_int = CkReduction::addReducer(_sparse_max_int);
  sparse_max_float = CkReduction::addReducer(_sparse_max_float);
  sparse_max_double = CkReduction::addReducer(_sparse_max_double);

  sparse_min_int = CkReduction::addReducer(_sparse_min_int);
  sparse_min_float= CkReduction::addReducer(_sparse_min_float);
  sparse_min_double = CkReduction::addReducer(_sparse_min_double);
}

int numDataSegs(const unsigned char *data){
  return *(int*)data;
}

CkDataSegHeader getDataSegHeader(int index, const unsigned char *data){
  int size=numDataSegs(data);
  CkDataSegHeader r;
  if(index >= size)
    CkAbort("Error!!!\n");

  memcpy(&r, data+2*sizeof(int)+sizeof(CkDataSegHeader)*index, sizeof(CkDataSegHeader));
  return r;
}

/*int getIndex(CkDataSegHeader *r, const unsigned char *ptr, int j){
  int i=0;
  CkDataSegHeader header = getDataSegHeader(j, ptr);
  while(!(header == r[i]))
    i++;
  return i;
}*/

unsigned char * getDataPtr(unsigned char *ptr){
/*
  int size;
  size = numDataSegs(ptr);
  return (unsigned char*)(ptr + sizeof(int) + size*sizeof(CkDataSegHeader));
*/
  return (ptr + *(((int*)ptr)+1));
}

CkDataSegHeader* getDataSegHeaderPtr(const unsigned char *ptr) {
  return (CkDataSegHeader*)(ptr + 2*sizeof(int));
}

CkDataSegHeader getDecompressedDataHdr(const unsigned char *msg){
  CkDataSegHeader retHead(0, 0, 0, 0);
  CkDataSegHeader h;
  int numSegs = numDataSegs(msg);

  for(int i=0; i<numSegs; i++){
    h = getDataSegHeader(i, msg);
    if(retHead.sx > h.sx)
      retHead.sx = h.sx;
    if(retHead.sy > h.sy)
      retHead.sy = h.sy;
    if(retHead.ex < h.ex)
      retHead.ex = h.ex;
    if(retHead.ey < h.ey)
      retHead.ey = h.ey;
  }
  return retHead;
}

#define SIMPLE_DECOMPRESSOR(dataType)\
dataType *decompressMsg(CkReductionMsg *m, CkDataSegHeader &h, dataType nullVal){\
  unsigned char *msg = (unsigned char*)m->getData();\
  h = getDecompressedDataHdr(msg);\
  CkDataSegHeader head;\
  dataType *data;\
  int sizeX = h.ex - h.sx + 1;\
  int sizeY = h.ey - h.sy + 1;\
  int numSegs = numDataSegs(msg);\
\
  data = new dataType[sizeX*sizeY];\
\
  int i;	\
  for(i=0; i<sizeX*sizeY; i++)\
      data[i] = nullVal;\
\
  dataType *msgDataptr = (dataType *)getDataPtr(msg);\
  for(i=0; i<numSegs; i++){\
    head = getDataSegHeader(i, msg);\
    int startY = head.sy - h.sy;\
    int endY   = head.ey - h.sy;\
    int startX = head.sx - h.sx;\
    int endX   = head.ex - h.sx;\
    for(int y=startY; y<=endY; y++){\
      int inc = y*sizeX;\
      for(int x=startX; x<=endX; x++)\
        data[x+inc] = *msgDataptr++;\
    }\
  }\
  return data;\
}

// define decompressor for 'int' data 
SIMPLE_DECOMPRESSOR(int)

// define decompressor for 'float' data
SIMPLE_DECOMPRESSOR(float)

// define decompressor for 'double' data
SIMPLE_DECOMPRESSOR(double)

// define decompressor for 'CkTwoFloats' data
SIMPLE_DECOMPRESSOR(CkTwoFloats)

// define decompressor for 'CkTwoDoubles' data
SIMPLE_DECOMPRESSOR(CkTwoDoubles)
#include "CkSparseContiguousReducer.def.h"
