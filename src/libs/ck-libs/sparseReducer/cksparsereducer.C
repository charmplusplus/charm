#include "cksparsereducer.h"

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
	int size1=0, size2=0;\
	int sizeofRec = sizeof(sparseRec1D<dataType>);\
	sparseRec1D<dataType> *arr1, *arr2;\
\
	for(int i=0; i<nMsg; i++)\
		size1 += msg[i]->getSize()/sizeofRec;\
\
	arr1 = new sparseRec1D<dataType>[size1];\
	arr2 = new sparseRec1D<dataType>[size1];\
\
	size1 = 0;\
\
	for(int i=0; i<nMsg; i++)\
	{\
		sparseRec1D<dataType> *tempArr = arr2;\
\
		arr2 = arr1;\
		size2 = size1;\
\
		arr1 = tempArr;\
		size1 = 0;\
\
		int size = msg[i]->getSize()/sizeofRec;\
		sparseRec1D<dataType> *temp = (sparseRec1D<dataType> *)(msg[i]->getData());\
\
		int index2=0, index=0;\
		while((size!=0)&&(size2!=0))\
		{\
			if(arr2[index2].i < temp[index].i)\
			{\
				arr1[size1].i = arr2[index2].i;\
				arr1[size1].data = arr2[index2].data;\
\
				size1++;\
				size2--;\
				index2++;\
			}\
			else\
			if(arr2[index2].i > temp[index].i)\
			{\
				arr1[size1].i = temp[index].i;\
				arr1[size1].data = temp[index].data;\
\
				size1++;\
				size--;\
				index++;\
			}\
			else\
			{\
				arr1[size1].i = arr2[index2].i;\
				arr1[size1].data = arr2[index2].data;\
				loop\
\
				size1++;\
				size2--;\
				size--;\
				index2++;\
				index++;\
			}\
		}\
\
		while(size!=0)\
		{\
			arr1[size1].i = temp[index].i;\
			arr1[size1].data = temp[index].data;\
\
			size1++;\
			size--;\
			index++;\
		}\
\
		while(size2!=0)\
		{\
			arr1[size1].i = arr2[index2].i;\
			arr1[size1].data = arr2[index2].data;\
\
			size1++;\
			size2--;\
			index2++;\
		}\
	}\
\
	CkReductionMsg* m = CkReductionMsg::buildNew(size1*sizeofRec, (void *)arr1);\
\
	delete[] arr1;\
	delete[] arr2;\
\
	return m;\
}

//Use this macro for reductions that have the same type for all inputs
#define SIMPLE_POLYMORPH_SPARSE_REDUCTION(nameBase,loop) \
  SIMPLE_SPARSE_REDUCTION(nameBase##_int,int,"%d",loop) \
  SIMPLE_SPARSE_REDUCTION(nameBase##_float,float,"%f",loop) \
  SIMPLE_SPARSE_REDUCTION(nameBase##_double,double,"%f",loop)


//Compute the sum the numbers passed by each element.
SIMPLE_POLYMORPH_SPARSE_REDUCTION(_sparse_sum, arr1[size1].data += temp[index].data;)

//Compute the product of the numbers passed by each element.
SIMPLE_POLYMORPH_SPARSE_REDUCTION(_sparse_product, arr1[size1].data *= temp[index].data;)

//Compute the largest number passed by any element.
SIMPLE_POLYMORPH_SPARSE_REDUCTION(_sparse_max,if (arr1[size1].data<temp[index].data) arr1[size1].data=temp[index].data;)

//Compute the smallest integer passed by any element.
SIMPLE_POLYMORPH_SPARSE_REDUCTION(_sparse_min,if (arr1[size1].data>temp[index].data) arr1[size1].data=temp[index].data;)

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
