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
int getIndex(CkDataSegHeader *r, unsigned char *ptr, int j);

/*
  macro defining various reducer functions
*/
#define SIMPLE_SPARSE_CONTIGUOUS_REDUCTION(name,dataType,typeStr,loop) \
static CkReductionMsg *name(int nMsg, CkReductionMsg ** msg){\
  int count = 0;\
  int numElements = 0;\
  CkDataSegHeader *headerArray;\
  int *size;\
  unsigned char *flag;\
\
  /* find the total data segments in n input msgs */\
  for(int i=0; i<nMsg; i++)\
    count += numDataSegs((unsigned char*)(msg[i]->getData()));\
\
  headerArray = new CkDataSegHeader[count];\
  size = new int[count];\
  flag = new (unsigned char)[count];\
\
  count = 0;\
\
  /* put all the unique data headers from input messages to the header-array */\
  for(int i=0; i<nMsg; i++){\
\
    unsigned char * data = (unsigned char*)(msg[i]->getData());\
    int numSegs = numDataSegs(data);\
\
    for(int j=0; j<numSegs; j++){\
\
      int index = count;\
      CkDataSegHeader node = getDataSegHeader(j, data);\
\
      /* to maintain x-sorted header array */\
      for(int k=count-1; k >= 0; k--)\
	if(node < headerArray[k])\
          index = k;\
	else\
	  break;\
\
      if((index != 0) && (node == headerArray[index-1]))\
	continue; /* overlap */\
\
      for(int k=count-1; k>=index; k--){\
	headerArray[k+1] = headerArray[k];\
	size[k+1] = size[k];\
      }\
\
      headerArray[index] = node;\
      size[index] = node.getNumElements();\
      count++;\
      numElements += size[index];\
    }\
  }\
\
 /* number of non-null data blocks and total number of elements in them is known.
    Now pack the input data into one buffer resolving the overlap.*/\
\
  unsigned char *data = new (unsigned char)[sizeof(int) +\
       sizeof(CkDataSegHeader)*count + sizeof(dataType)*numElements];\
\
  memset(flag, 0,count*sizeof(unsigned char));\
  memcpy(data, &count, sizeof(int));\
\
  unsigned char *ptr = data + sizeof(int);\
  /* copy the data segment headers to packing buffer */\
  for(int i=0; i<count; i++){\
    memcpy(ptr, &(headerArray[i]), sizeof(CkDataSegHeader));\
    ptr += sizeof(CkDataSegHeader);\
    if(i != 0)\
      size[i] += size[i-1];\
  }\
\
 /* copy data from n-input messages to packing buffer */\
  for(int i=0; i<nMsg; i++){\
    unsigned char *msgptr = (unsigned char*)(msg[i]->getData());\
    dataType *msgDataptr = (dataType *)getDataPtr(msgptr);\
    dataType *dataptr = (dataType *)ptr;\
    int numSegs = numDataSegs(msgptr);\
    int index = 0;\
    int startInd = 0;\
\
    for(int j=0; j<numSegs; j++){\
      index = getIndex(headerArray, msgptr, j);\
      if(index != 0)\
        startInd = size[index-1];\
      else\
        startInd = 0;\
\
      if(flag[index] != 0){\
        for(int k=startInd; k<size[index]; k++){\
          loop /* operation to be performed on overlapping data */\
          msgDataptr++;\
        }\
      }\
      else{\
        for(int k=startInd; k<size[index]; k++)\
          *(dataptr + k) = *msgDataptr++;\
      }\
      flag[index] = 1;\
    }\
  }\
\
  CkReductionMsg* m = CkReductionMsg::buildNew(sizeof(int) + \
    sizeof(CkDataSegHeader)*count + sizeof(dataType)*numElements, (void*)data);\
\
  delete[] headerArray;\
  delete[] size;\
  delete[] data;\
  delete[] flag;\
\
  return m;\
}

//Use this macro for reductions that have the same type for all inputs
#define SIMPLE_POLYMORPH_SPARSE_CONTIGUOUS_REDUCTION(nameBase,loop) \
  SIMPLE_SPARSE_CONTIGUOUS_REDUCTION(nameBase##_int,int,"%d",loop) \
  SIMPLE_SPARSE_CONTIGUOUS_REDUCTION(nameBase##_float,float,"%f",loop) \
  SIMPLE_SPARSE_CONTIGUOUS_REDUCTION(nameBase##_double,double,"%f",loop)

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
  int size=0;
  memcpy(&size, data, sizeof(int));
  return size;
}

CkDataSegHeader getDataSegHeader(int index, unsigned char *data){
  int size=numDataSegs(data);
  CkDataSegHeader r;
  if(index >= size)
    CkAbort("Error!!!\n");

  unsigned char *ptr = data;
  ptr += sizeof(int);
  ptr += sizeof(CkDataSegHeader)*index;
  memcpy(&r, ptr, sizeof(CkDataSegHeader));
  return r;
}

int getIndex(CkDataSegHeader *r, unsigned char *ptr, int j){
  int i=0;
  CkDataSegHeader header = getDataSegHeader(j, ptr);
  while(!(header == r[i]))
    i++;
  return i;
}

unsigned char * getDataPtr(const unsigned char *ptr){
  int size;
  size = numDataSegs(ptr);
  return (unsigned char*)(ptr + sizeof(int) + size*sizeof(CkDataSegHeader));
}

#include "CkSparseContiguousReducer.def.h"
