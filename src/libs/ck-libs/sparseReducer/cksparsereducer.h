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
**    call r.contribute[Sum | Product | Max | Min](ArrayElement *) e.g. to using 'sum' operation call contributeSum(this)
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
                	int ind = index;
                        // simple insertion sort
                       	while((ind != 0)&&(records[ind-1].index > i))
                        {
                        	records[ind] = records[ind-1];
                                ind--;
                        }
			records[ind].index = i;
			records[ind].data = data;
			index++;
		}

                // contribute to sum reducers
                void contributeSum(ArrayElement *elem, const CkCallback &cb)
                {
			T dummy; // to resolve the function to be called
                        contributeSum(elem, cb, dummy);
                }

                void contributeSum(ArrayElement *elem, const CkCallback &cb, int dummy)
		{
			elem->contribute(size*sizeof(rec), records, sparse_sum_int, cb);
		}

                void contributeSum(ArrayElement *elem, const CkCallback &cb, float dummy)
		{
			elem->contribute(size*sizeof(rec), records, sparse_sum_float, cb);
		}

                void contributeSum(ArrayElement *elem, const CkCallback &cb, double dummy)
		{
			elem->contribute(size*sizeof(rec), records, sparse_sum_double, cb);
		}

                // contribute to product reducers
                void contributeProduct(ArrayElement *elem, const CkCallback &cb)
                {
			T dummy;
                        contributeProduct(elem, cb, dummy);
                }

                void contributeProduct(ArrayElement *elem, const CkCallback &cb, int dummy)
		{
			elem->contribute(size*sizeof(rec), records, sparse_product_int, cb);
		}

                void contributeProduct(ArrayElement *elem, const CkCallback &cb, float dummy)
		{
			elem->contribute(size*sizeof(rec), records, sparse_product_float, cb);
		}

                void contributeProduct(ArrayElement *elem, const CkCallback &cb, double dummy)
		{
			elem->contribute(size*sizeof(rec), records, sparse_product_double, cb);
		}

                // contribute to max reducers
                void contributeMax(ArrayElement *elem, const CkCallback &cb)
                {
			T dummy;
                        contributeMax(elem, cb, dummy);
                }

                void contributeMax(ArrayElement *elem, const CkCallback &cb, int dummy)
		{
			elem->contribute(size*sizeof(rec), records, sparse_max_int, cb);
		}

                void contributeMax(ArrayElement *elem, const CkCallback &cb, float dummy)
		{
			elem->contribute(size*sizeof(rec), records, sparse_max_float, cb);
		}

                void contributeMax(ArrayElement *elem, const CkCallback &cb, double dummy)
		{
			elem->contribute(size*sizeof(rec), records, sparse_max_double, cb);
		}

                // contribute to min reducers
                void contributeMin(ArrayElement *elem, const CkCallback &cb)
                {
			T dummy;
                        contributeMin(elem, cb, dummy);
                }

                void contributeMin(ArrayElement *elem, const CkCallback &cb, int dummy)
		{
			elem->contribute(size*sizeof(rec), records, sparse_min_int, cb);
		}

                void contributeMin(ArrayElement *elem, const CkCallback &cb, float dummy)
		{
			elem->contribute(size*sizeof(rec), records, sparse_min_float, cb);
		}

                void contributeMin(ArrayElement *elem, const CkCallback &cb, double dummy)
		{
			elem->contribute(size*sizeof(rec), records, sparse_min_double, cb);
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
