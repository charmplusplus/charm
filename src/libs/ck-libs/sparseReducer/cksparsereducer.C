#include "cksparsereducer.h"
/*
** All the reducer functions return a CkReductionMsg, which contains an array of
 sparseRec1D<dataType>.
** For instance, if the reducer used was sparse_sum_int, then each record in the
 array will be of type sparseRec1D<int>
** sparseRec1D<T> has following structure:
** template <class T>
** struct sparseRec1D
** {
**	int i; // index of the element
**	T data; // actual data
** };
** The array in the message returned by reducers is sorted on 'i'.
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

#define SIMPLE_SPARSE_REDUCTION(name,dataType,typeStr,loop) \
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
      if(mergedData->index < mergeMsg->index)\
        *out++ = *mergedData++;\
      else\
      if(mergedData->index > mergeMsg->index)\
        *out++ = *mergeMsg++;\
      else\
      {\
        out->index = mergedData->index;\
        out->data = mergedData->data;\
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
#define SIMPLE_POLYMORPH_SPARSE_REDUCTION(nameBase,loop) \
  SIMPLE_SPARSE_REDUCTION(nameBase##_int,int,"%d",loop) \
  SIMPLE_SPARSE_REDUCTION(nameBase##_float,float,"%f",loop) \
  SIMPLE_SPARSE_REDUCTION(nameBase##_double,double,"%f",loop)


// Merge the sparse arrays passed by elements, summing the elements with same
// indices.
SIMPLE_POLYMORPH_SPARSE_REDUCTION(_sparse_sum, out->data += mergeMsg->data;)

// Merge the sparse arrays passed by elements, multiplying the elements with
// same indices.
SIMPLE_POLYMORPH_SPARSE_REDUCTION(_sparse_product, out->data *= mergeMsg->data;)

// Merge the sparse arrays passed by elements, keeping the largest of the
// elements with same indices.
SIMPLE_POLYMORPH_SPARSE_REDUCTION(_sparse_max, if((out->data)<(mergeMsg->data))
 out->data=mergeMsg->data;)

// Merge the sparse arrays passed by elements, keeping the smallest of the
// elements with same indices.
SIMPLE_POLYMORPH_SPARSE_REDUCTION(_sparse_min,if((out->data)>(mergeMsg->data))
 out->data=mergeMsg->data;)

// register simple reducers for sparse arrays.
void registerReducers(void)
{
	sparse_sum_int = CkReduction::addReducer(_sparse_sum_int);
	sparse_sum_float = CkReduction::addReducer(_sparse_sum_float);
	sparse_sum_double = CkReduction::addReducer(_sparse_sum_double);

	sparse_product_int = CkReduction::addReducer(_sparse_product_int);
	sparse_product_float = CkReduction::addReducer(_sparse_product_float);
	sparse_product_double = CkReduction::addReducer(_sparse_product_double);

	sparse_max_int = CkReduction::addReducer(_sparse_max_int);
	sparse_max_float = CkReduction::addReducer(_sparse_max_float);
	sparse_max_double = CkReduction::addReducer(_sparse_max_double);

	sparse_min_int = CkReduction::addReducer(_sparse_min_int);
	sparse_min_float = CkReduction::addReducer(_sparse_min_float);
	sparse_min_double = CkReduction::addReducer(_sparse_min_double);
}

#include "CkSparseReducer.def.h"
