#include "cksparsereducer.h"

/*
** For 1D-sparse reduction, all the reducer functions return a CkReductionMsg,
** which contains an array of  sparseRec1D<dataType>.
**
** sparseRec1D<T> has following structure:
** template <class T>
** struct sparseRec1D
** {
**	int x; // index of the element
**	T data; // actual data
** };
** The array in the message returned by reducers is sorted on 'x'.
**
** Similarly for 2D & 3D sparse reduction, all the reducer functions return a
** CkReductionMsg, which contains an array of  sparseRec2D<dataType> and
** sparseRec3D<dataType> respectively.
*/

CkReduction::reducerType sparse1D_sum_int;
CkReduction::reducerType sparse1D_sum_float;
CkReduction::reducerType sparse1D_sum_double;

CkReduction::reducerType sparse1D_product_int;
CkReduction::reducerType sparse1D_product_float;
CkReduction::reducerType sparse1D_product_double;

CkReduction::reducerType sparse1D_max_int;
CkReduction::reducerType sparse1D_max_float;
CkReduction::reducerType sparse1D_max_double;

CkReduction::reducerType sparse1D_min_int;
CkReduction::reducerType sparse1D_min_float;
CkReduction::reducerType sparse1D_min_double;

CkReduction::reducerType sparse2D_sum_int;
CkReduction::reducerType sparse2D_sum_float;
CkReduction::reducerType sparse2D_sum_double;

CkReduction::reducerType sparse2D_product_int;
CkReduction::reducerType sparse2D_product_float;
CkReduction::reducerType sparse2D_product_double;

CkReduction::reducerType sparse2D_max_int;
CkReduction::reducerType sparse2D_max_float;
CkReduction::reducerType sparse2D_max_double;

CkReduction::reducerType sparse2D_min_int;
CkReduction::reducerType sparse2D_min_float;
CkReduction::reducerType sparse2D_min_double;

CkReduction::reducerType sparse3D_sum_int;
CkReduction::reducerType sparse3D_sum_float;
CkReduction::reducerType sparse3D_sum_double;

CkReduction::reducerType sparse3D_product_int;
CkReduction::reducerType sparse3D_product_float;
CkReduction::reducerType sparse3D_product_double;

CkReduction::reducerType sparse3D_max_int;
CkReduction::reducerType sparse3D_max_float;
CkReduction::reducerType sparse3D_max_double;

CkReduction::reducerType sparse3D_min_int;
CkReduction::reducerType sparse3D_min_float;
CkReduction::reducerType sparse3D_min_double;

#define SIMPLE_SPARSE1D_REDUCTION(name,dataType,typeStr,loop) \
static CkReductionMsg *name(int nMsg,CkReductionMsg **msg)\
{\
  int mergedDataSize=0, outDataSize=0;\
  int sizeofRec = sizeof(sparseRec1D<dataType>);\
  sparseRec1D<dataType> *mergedData, *mergedDataEnd;\
  sparseRec1D<dataType> *out, *mergeMsg, *mergeMsgEnd, *buff1, *buff2;\
\
  /*find the maximum possible size of final data array*/\
  for(int i=0; i<nMsg; i++)\
    outDataSize += msg[i]->getSize()/sizeofRec;\
\
  buff1 = new sparseRec1D<dataType>[outDataSize];\
  buff2 = new sparseRec1D<dataType>[outDataSize];\
\
  outDataSize = 0;\
  /*merge the n-sparse arrays. This is an inefficient algorithm, merges input
    sparse-arrays one by one to the final array*/\
  for(int i=0; i<nMsg; i++)\
  {\
\
    mergedData = buff2;\
    mergedDataEnd = mergedData + outDataSize;\
    mergedDataSize = outDataSize;\
\
    mergeMsg = (sparseRec1D<dataType> *)(msg[i]->getData());\
    mergeMsgEnd = mergeMsg + (msg[i]->getSize()/sizeofRec);\
\
    out = buff1;\
    outDataSize = 0;\
\
    buff1 = mergedData;\
    buff2 = out;\
\
    /*merge mergedData and data in msg to out*/\
    while((mergedData != mergedDataEnd)&&(mergeMsg != mergeMsgEnd))\
    {\
      if(mergedData->x < mergeMsg->x)\
        *out++ = *mergedData++;\
      else\
      if(mergedData->x > mergeMsg->x)\
        *out++ = *mergeMsg++;\
      else\
      {\
        *out = *mergedData;\
        loop\
        out++; mergedData++; mergeMsg++;\
      }\
    }\
\
    while(mergedData != mergedDataEnd)\
      *out++ = *mergedData++;\
\
    while(mergeMsg != mergeMsgEnd)\
      *out++ = *mergeMsg++;\
\
    outDataSize = out - buff2;\
  }\
\
  CkReductionMsg* m = CkReductionMsg::buildNew(outDataSize*sizeofRec, \
(void*)buff2);\
\
  delete[] buff1;\
  delete[] buff2;\
\
  return m;\
}

#define SIMPLE_SPARSE2D_REDUCTION(name,dataType,typeStr,loop) \
static CkReductionMsg *name(int nMsg,CkReductionMsg **msg)\
{\
  int mergedDataSize=0, outDataSize=0;\
  int sizeofRec = sizeof(sparseRec2D<dataType>);\
  sparseRec2D<dataType> *mergedData, *mergedDataEnd;\
  sparseRec2D<dataType> *out, *mergeMsg, *mergeMsgEnd, *buff1, *buff2;\
\
  /*find the maximum possible size of final data array*/\
  for(int i=0; i<nMsg; i++)\
    outDataSize += msg[i]->getSize()/sizeofRec;\
\
  buff1 = new sparseRec2D<dataType>[outDataSize];\
  buff2 = new sparseRec2D<dataType>[outDataSize];\
\
  outDataSize = 0;\
  /*merge the n-sparse arrays. This is an inefficient algorithm, merges input
    sparse-arrays one by one to the final array*/\
  for(int i=0; i<nMsg; i++)\
  {\
\
    mergedData = buff2;\
    mergedDataEnd = mergedData + outDataSize;\
    mergedDataSize = outDataSize;\
\
    mergeMsg = (sparseRec2D<dataType> *)(msg[i]->getData());\
    mergeMsgEnd = mergeMsg + (msg[i]->getSize()/sizeofRec);\
\
    out = buff1;\
    outDataSize = 0;\
\
    buff1 = mergedData;\
    buff2 = out;\
\
    /*merge mergedData and data in msg to out*/\
    while((mergedData != mergedDataEnd)&&(mergeMsg != mergeMsgEnd))\
    {\
      if((mergedData->y < mergeMsg->y) || ((mergedData->y == mergeMsg->y) &&\
        (mergedData->x < mergeMsg->x)))\
        *out++ = *mergedData++;\
      else\
      if((mergedData->y > mergeMsg->y) || ((mergedData->y == mergeMsg->y) &&\
        (mergedData->x > mergeMsg->x)))\
        *out++ = *mergeMsg++;\
      else\
      {\
        *out = *mergedData;\
        loop\
        out++; mergedData++; mergeMsg++;\
      }\
    }\
\
    while(mergedData != mergedDataEnd)\
      *out++ = *mergedData++;\
\
    while(mergeMsg != mergeMsgEnd)\
      *out++ = *mergeMsg++;\
\
    outDataSize = out - buff2;\
  }\
\
  CkReductionMsg* m = CkReductionMsg::buildNew(outDataSize*sizeofRec, \
(void*)buff2);\
\
  delete[] buff1;\
  delete[] buff2;\
\
  return m;\
}

#define SIMPLE_SPARSE3D_REDUCTION(name,dataType,typeStr,loop) \
static CkReductionMsg *name(int nMsg,CkReductionMsg **msg)\
{\
  int mergedDataSize=0, outDataSize=0;\
  int sizeofRec = sizeof(sparseRec3D<dataType>);\
  sparseRec3D<dataType> *mergedData, *mergedDataEnd;\
  sparseRec3D<dataType> *out, *mergeMsg, *mergeMsgEnd, *buff1, *buff2;\
\
  /*find the maximum possible size of final data array*/\
  for(int i=0; i<nMsg; i++)\
    outDataSize += msg[i]->getSize()/sizeofRec;\
\
  buff1 = new sparseRec3D<dataType>[outDataSize];\
  buff2 = new sparseRec3D<dataType>[outDataSize];\
\
  outDataSize = 0;\
  /*merge the n-sparse arrays. This is an inefficient algorithm, merges input
    sparse-arrays one by one to the final array*/\
  for(int i=0; i<nMsg; i++)\
  {\
\
    mergedData = buff2;\
    mergedDataEnd = mergedData + outDataSize;\
    mergedDataSize = outDataSize;\
\
    mergeMsg = (sparseRec3D<dataType> *)(msg[i]->getData());\
    mergeMsgEnd = mergeMsg + (msg[i]->getSize()/sizeofRec);\
\
    out = buff1;\
    outDataSize = 0;\
\
    buff1 = mergedData;\
    buff2 = out;\
\
    /*merge mergedData and data in msg to out*/\
    while((mergedData != mergedDataEnd)&&(mergeMsg != mergeMsgEnd))\
    {\
      if((mergedData->z < mergeMsg->z) || ((mergedData->z == mergeMsg->z) &&\
        (mergedData->y < mergeMsg->y)) || ((mergedData->z == mergeMsg->z) &&\
        (mergedData->y == mergeMsg->y) && (mergedData->x < mergeMsg->x)))\
        *out++ = *mergedData++;\
      else\
      if((mergedData->z > mergeMsg->z) || ((mergedData->z == mergeMsg->z) &&\
        (mergedData->y > mergeMsg->y)) || ((mergedData->z == mergeMsg->z) &&\
        (mergedData->y == mergeMsg->y) && (mergedData->x > mergeMsg->x)))\
        *out++ = *mergeMsg++;\
      else\
      {\
        *out = *mergedData;\
        loop\
        out++; mergedData++; mergeMsg++;\
      }\
    }\
\
    while(mergedData != mergedDataEnd)\
      *out++ = *mergedData++;\
\
    while(mergeMsg != mergeMsgEnd)\
      *out++ = *mergeMsg++;\
\
    outDataSize = out - buff2;\
  }\
\
  CkReductionMsg* m = CkReductionMsg::buildNew(outDataSize*sizeofRec, \
(void*)buff2);\
\
  delete[] buff1;\
  delete[] buff2;\
\
  return m;\
}

//Use this macro for reductions that have the same type for all inputs
#define SIMPLE_POLYMORPH_SPARSE1D_REDUCTION(nameBase,loop) \
  SIMPLE_SPARSE1D_REDUCTION(nameBase##_int,int,"%d",loop) \
  SIMPLE_SPARSE1D_REDUCTION(nameBase##_float,float,"%f",loop) \
  SIMPLE_SPARSE1D_REDUCTION(nameBase##_double,double,"%f",loop)

#define SIMPLE_POLYMORPH_SPARSE2D_REDUCTION(nameBase,loop) \
  SIMPLE_SPARSE2D_REDUCTION(nameBase##_int,int,"%d",loop) \
  SIMPLE_SPARSE2D_REDUCTION(nameBase##_float,float,"%f",loop) \
  SIMPLE_SPARSE2D_REDUCTION(nameBase##_double,double,"%f",loop)

#define SIMPLE_POLYMORPH_SPARSE3D_REDUCTION(nameBase,loop) \
  SIMPLE_SPARSE3D_REDUCTION(nameBase##_int,int,"%d",loop) \
  SIMPLE_SPARSE3D_REDUCTION(nameBase##_float,float,"%f",loop) \
  SIMPLE_SPARSE3D_REDUCTION(nameBase##_double,double,"%f",loop)

// Merge the sparse arrays passed by elements, summing the elements with same
// indices.
SIMPLE_POLYMORPH_SPARSE1D_REDUCTION(_sparse1D_sum, out->data += mergeMsg->data;)

// Merge the sparse arrays passed by elements, multiplying the elements with
// same indices.
SIMPLE_POLYMORPH_SPARSE1D_REDUCTION(_sparse1D_product, out->data *=
 mergeMsg->data;)

// Merge the sparse arrays passed by elements, keeping the largest of the
// elements with same indices.
SIMPLE_POLYMORPH_SPARSE1D_REDUCTION(_sparse1D_max, if(out->data<mergeMsg->data)
 out->data=mergeMsg->data;)

// Merge the sparse arrays passed by elements, keeping the smallest of the
// elements with same indices.
SIMPLE_POLYMORPH_SPARSE1D_REDUCTION(_sparse1D_min,if(out->data>mergeMsg->data)
 out->data=mergeMsg->data;)

// Merge the sparse arrays passed by elements, summing the elements with same
// indices.
SIMPLE_POLYMORPH_SPARSE2D_REDUCTION(_sparse2D_sum, out->data += mergeMsg->data;)

// Merge the sparse arrays passed by elements, multiplying the elements with
// same indices.
SIMPLE_POLYMORPH_SPARSE2D_REDUCTION(_sparse2D_product, out->data *=
 mergeMsg->data;)

// Merge the sparse arrays passed by elements, keeping the largest of the
// elements with same indices.
SIMPLE_POLYMORPH_SPARSE2D_REDUCTION(_sparse2D_max, if(out->data<mergeMsg->data)
 out->data=mergeMsg->data;)

// Merge the sparse arrays passed by elements, keeping the smallest of the
// elements with same indices.
SIMPLE_POLYMORPH_SPARSE2D_REDUCTION(_sparse2D_min,if(out->data>mergeMsg->data)
 out->data=mergeMsg->data;)

// Merge the sparse arrays passed by elements, summing the elements with same
// indices.
SIMPLE_POLYMORPH_SPARSE3D_REDUCTION(_sparse3D_sum, out->data += mergeMsg->data;)

// Merge the sparse arrays passed by elements, multiplying the elements with
// same indices.
SIMPLE_POLYMORPH_SPARSE3D_REDUCTION(_sparse3D_product, out->data *=
 mergeMsg->data;)

// Merge the sparse arrays passed by elements, keeping the largest of the
// elements with same indices.
SIMPLE_POLYMORPH_SPARSE3D_REDUCTION(_sparse3D_max, if(out->data<mergeMsg->data)
 out->data=mergeMsg->data;)

// Merge the sparse arrays passed by elements, keeping the smallest of the
// elements with same indices.
SIMPLE_POLYMORPH_SPARSE3D_REDUCTION(_sparse3D_min,if(out->data>mergeMsg->data)
 out->data=mergeMsg->data;)

// register simple reducers for sparse arrays.
void registerReducers(void)
{
  sparse1D_sum_int = CkReduction::addReducer(_sparse1D_sum_int);
  sparse1D_sum_float = CkReduction::addReducer(_sparse1D_sum_float);
  sparse1D_sum_double = CkReduction::addReducer(_sparse1D_sum_double);

  sparse1D_product_int = CkReduction::addReducer(_sparse1D_product_int);
  sparse1D_product_float = CkReduction::addReducer(_sparse1D_product_float);
  sparse1D_product_double = CkReduction::addReducer(_sparse1D_product_double);

  sparse1D_max_int = CkReduction::addReducer(_sparse1D_max_int);
  sparse1D_max_float = CkReduction::addReducer(_sparse1D_max_float);
  sparse1D_max_double = CkReduction::addReducer(_sparse1D_max_double);

  sparse1D_min_int = CkReduction::addReducer(_sparse1D_min_int);
  sparse1D_min_float = CkReduction::addReducer(_sparse1D_min_float);
  sparse1D_min_double = CkReduction::addReducer(_sparse1D_min_double);

  sparse2D_sum_int = CkReduction::addReducer(_sparse2D_sum_int);
  sparse2D_sum_float = CkReduction::addReducer(_sparse2D_sum_float);
  sparse2D_sum_double = CkReduction::addReducer(_sparse2D_sum_double);

  sparse2D_product_int = CkReduction::addReducer(_sparse2D_product_int);
  sparse2D_product_float = CkReduction::addReducer(_sparse2D_product_float);
  sparse2D_product_double = CkReduction::addReducer(_sparse2D_product_double);

  sparse2D_max_int = CkReduction::addReducer(_sparse2D_max_int);
  sparse2D_max_float = CkReduction::addReducer(_sparse2D_max_float);
  sparse2D_max_double = CkReduction::addReducer(_sparse2D_max_double);

  sparse2D_min_int = CkReduction::addReducer(_sparse2D_min_int);
  sparse2D_min_float = CkReduction::addReducer(_sparse2D_min_float);
  sparse2D_min_double = CkReduction::addReducer(_sparse2D_min_double);

  sparse3D_sum_int = CkReduction::addReducer(_sparse3D_sum_int);
  sparse3D_sum_float = CkReduction::addReducer(_sparse3D_sum_float);
  sparse3D_sum_double = CkReduction::addReducer(_sparse3D_sum_double);

  sparse3D_product_int = CkReduction::addReducer(_sparse3D_product_int);
  sparse3D_product_float = CkReduction::addReducer(_sparse3D_product_float);
  sparse3D_product_double = CkReduction::addReducer(_sparse3D_product_double);

  sparse3D_max_int = CkReduction::addReducer(_sparse3D_max_int);
  sparse3D_max_float = CkReduction::addReducer(_sparse3D_max_float);
  sparse3D_max_double = CkReduction::addReducer(_sparse3D_max_double);

  sparse3D_min_int = CkReduction::addReducer(_sparse3D_min_int);
  sparse3D_min_float = CkReduction::addReducer(_sparse3D_min_float);
  sparse3D_min_double = CkReduction::addReducer(_sparse3D_min_double);
}

#include "CkSparseReducer.def.h"
