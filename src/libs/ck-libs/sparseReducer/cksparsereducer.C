#include "cksparsereducer.h"
/*
** All the reducer functions return a CkReductionMsg, which contains an array of sparseRec1D<dataType>.
** For instance, if the reducer used was sparse_sum_int, then each record in the array will be of type sparseRec1D<int>
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
	int mergedDataSize=0, size2=0;\
	int sizeofRec = sizeof(sparseRec1D<dataType>);\
	sparseRec1D<dataType> *mergedData, *arr2;\
	/*find the maximum possible size of final data array*/\
	for(int i=0; i<nMsg; i++)\
		mergedDataSize += msg[i]->getSize()/sizeofRec;\
\
	mergedData = new sparseRec1D<dataType>[mergedDataSize];\
	arr2 = new sparseRec1D<dataType>[mergedDataSize];\
\
	mergedDataSize = 0;\
	/*merge the n-sparse arrays. This is an inefficient algorithm, merges input sparse-arrays one by one to the final array*/\
	for(int i=0; i<nMsg; i++)\
	{	/*swap the array pointers, so that 'arr2' holds the merged data for 1 to i-1 input arrays.*/\
		/*mergedData is used to merge data from 'msg[i]' and 'arr2'*/\
		sparseRec1D<dataType> *tempArr = arr2;\
\
		arr2 = mergedData;\
		size2 = mergedDataSize;\
\
		mergedData = tempArr;\
		mergedDataSize = 0;\
\
		int size = msg[i]->getSize()/sizeofRec;\
		sparseRec1D<dataType> *temp = (sparseRec1D<dataType> *)(msg[i]->getData());\
		/*merge arr2 and data in msg[i] to mergedData*/\
		int index2=0, index=0;\
		while((size!=0)&&(size2!=0))\
		{\
			if(arr2[index2].i < temp[index].i)\
			{\
				mergedData[mergedDataSize].i = arr2[index2].i;\
				mergedData[mergedDataSize].data = arr2[index2].data;\
\
				mergedDataSize++;\
				size2--;\
				index2++;\
			}\
			else\
			if(arr2[index2].i > temp[index].i)\
			{\
				mergedData[mergedDataSize].i = temp[index].i;\
				mergedData[mergedDataSize].data = temp[index].data;\
\
				mergedDataSize++;\
				size--;\
				index++;\
			}\
			else\
			{\
				mergedData[mergedDataSize].i = arr2[index2].i;\
				mergedData[mergedDataSize].data = arr2[index2].data;\
				loop\
\
				mergedDataSize++;\
				size2--;\
				size--;\
				index2++;\
				index++;\
			}\
		}\
\
		while(size!=0)\
		{\
			mergedData[mergedDataSize].i = temp[index].i;\
			mergedData[mergedDataSize].data = temp[index].data;\
\
			mergedDataSize++;\
			size--;\
			index++;\
		}\
\
		while(size2!=0)\
		{\
			mergedData[mergedDataSize].i = arr2[index2].i;\
			mergedData[mergedDataSize].data = arr2[index2].data;\
\
			mergedDataSize++;\
			size2--;\
			index2++;\
		}\
	}\
\
	CkReductionMsg* m = CkReductionMsg::buildNew(mergedDataSize*sizeofRec, (void *)mergedData);\
\
	delete[] mergedData;\
	delete[] arr2;\
\
	return m;\
}

//Use this macro for reductions that have the same type for all inputs
#define SIMPLE_POLYMORPH_SPARSE_REDUCTION(nameBase,loop) \
  SIMPLE_SPARSE_REDUCTION(nameBase##_int,int,"%d",loop) \
  SIMPLE_SPARSE_REDUCTION(nameBase##_float,float,"%f",loop) \
  SIMPLE_SPARSE_REDUCTION(nameBase##_double,double,"%f",loop)


// Merge the sparse arrays passed by elements, summing the elements with same indices.
SIMPLE_POLYMORPH_SPARSE_REDUCTION(_sparse_sum, mergedData[mergedDataSize].data += temp[index].data;)

// Merge the sparse arrays passed by elements, multiplying the elements with same indices.
SIMPLE_POLYMORPH_SPARSE_REDUCTION(_sparse_product, mergedData[mergedDataSize].data *= temp[index].data;)

// Merge the sparse arrays passed by elements, keeping the largest of the elements with same indices.
SIMPLE_POLYMORPH_SPARSE_REDUCTION(_sparse_max,if (mergedData[mergedDataSize].data<temp[index].data) mergedData[mergedDataSize].data=temp[index].data;)

// Merge the sparse arrays passed by elements, keeping the smallest of the elements with same indices.
SIMPLE_POLYMORPH_SPARSE_REDUCTION(_sparse_min,if (mergedData[mergedDataSize].data>temp[index].data) mergedData[mergedDataSize].data=temp[index].data;)

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
