#ifndef _CKSPARSEREDUCER_H
#define _CKSPARSEREDUCER_H

#include "CkSparseReducer.decl.h"

template <class T>
struct sparseRec1D
{
	sparseRec1D(int _index, T _data)
	{
		index = _index;
		data = _data;
	}

	sparseRec1D()
	{
	}

	int index; // index of element
	T data; // actual data
};

extern CkReduction::reducerType sparse_sum_int;
extern CkReduction::reducerType sparse_sum_float;
extern CkReduction::reducerType sparse_sum_double;

extern CkReduction::reducerType sparse_product_int;
extern CkReduction::reducerType sparse_product_float;
extern CkReduction::reducerType sparse_product_double;

extern CkReduction::reducerType sparse_max_int;
extern CkReduction::reducerType sparse_max_float;
extern CkReduction::reducerType sparse_max_double;
extern CkReduction::reducerType sparse_min_int;
extern CkReduction::reducerType sparse_min_float;
extern CkReduction::reducerType sparse_min_double;

/*
** To contribute a sparse array,
**    create an object of CkSparseReducer1D<T> r(numOfElements). Here 'numOfElements' is the # elements to contribute.
**    call r.add(index, data) 'numOfElements' times, to add all the elements to the object r.
**    call r.contribute(...)
**  NOTE that sparse reduction library expects data contributed (added to r) to be sorted on index.
*/

template <class T>
class CkSparseReducer1D
{
	public:

		CkSparseReducer1D(int numOfElements)
		{
			size = numOfElements;
			index = 0;
			if(size != 0)
				records = new rec[size];
			else
				records = NULL;
		}

		~CkSparseReducer1D()
		{
			if(records != NULL)
				delete[] records;
		}

		void add(int i, T data)
		{
			records[index].index = i;
			records[index].data = data;
			index++;
		}

                void contribute(ArrayElement *elem, CkReduction::reducerType type)
		{
			elem->contribute(size*sizeof(rec), records, type);
                }

		void contribute(ArrayElement *elem, CkReduction::reducerType type, const CkCallback &cb)
		{
			elem->contribute(size*sizeof(rec), records, type, cb);
		}

	protected:

		typedef sparseRec1D<T> rec;
		rec *records;
		int size;
		int index;

	private:
		CkSparseReducer1D(){} // should not use the default constructor
};

#endif
