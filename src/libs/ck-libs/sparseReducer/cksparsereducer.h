#ifndef _CKSPARSEREDUCER_H
#define _CKSPARSEREDUCER_H

#include "CkSparseReducer.decl.h"

/*
** To contribute a 1D-sparse array,
**    - create an object of CkSparseReducer1D<T> r(numOfElements).
**           Here 'numOfElements' is the # elements to contribute.
**    - call r.add(x, data) 'numOfElements' times,
**           to add all the elements to the object r. Here 'x' is the actual
**           index of 'data' in the 1D array.
**    - call r.contribute[Sum | Product | Max | Min](ArrayElement *, CkCallback)
**           e.g. to using 'sum' operation call r.contributeSum(this, cb)
**
** To contribute a 2D-sparse array,
**    - create an object of CkSparseReducer2D<T> r(numOfElements).
**           Here 'numOfElements' is the # elements to contribute.
**    - call r.add(x, y, data) 'numOfElements' times,
**           to add all the elements to the object r. Here 'x' and 'y' are the
**           actual 2-indices of 'data' in the 2D array.
**    - call r.contribute[Sum | Product | Max | Min](ArrayElement *, CkCallback)
**           e.g. to using 'sum' operation call r.contributeSum(this, cb)
**
** To contribute a 3D-sparse array,
**    - create an object of CkSparseReducer3D<T> r(numOfElements).
**           Here 'numOfElements' is the # elements to contribute.
**    - call r.add(x, y, z, data) 'numOfElements' times,
**           to add all the elements to the object r. Here 'x', 'y' and 'z' are
**           the actual indices of 'data' in the 3D array.
**    - call r.contribute[Sum | Product | Max | Min](ArrayElement *, CkCallback)
**           e.g. to using 'sum' operation call r.contributeSum(this, cb)
*/

template <class T>
struct sparseRec1D
{
  sparseRec1D(int _x, T _data)
  {
    x = _x;
    data = _data;
  }

  sparseRec1D()
  {
  }

  int x; // index of element
  T data; // actual data
};

template <class T>
struct sparseRec2D
{
  sparseRec2D(int _x, int _y, T _data)
  {
    x = _x;
    y = _y;
    data = _data;
  }

  sparseRec2D()
  {
  }

  int x,y; // index of element
  T data;  // actual data
};

template <class T>
struct sparseRec3D
{
  sparseRec3D(int _x, int _y, int _z, T _data)
  {
    x = _x;
    y = _y;
    z = _z;
    data = _data;
  }

  sparseRec3D()
  {
  }

  int x,y,z; // index of element
  T data;    // actual data
};


extern CkReduction::reducerType sparse1D_sum_int;
extern CkReduction::reducerType sparse1D_sum_float;
extern CkReduction::reducerType sparse1D_sum_double;

extern CkReduction::reducerType sparse1D_product_int;
extern CkReduction::reducerType sparse1D_product_float;
extern CkReduction::reducerType sparse1D_product_double;

extern CkReduction::reducerType sparse1D_max_int;
extern CkReduction::reducerType sparse1D_max_float;
extern CkReduction::reducerType sparse1D_max_double;

extern CkReduction::reducerType sparse1D_min_int;
extern CkReduction::reducerType sparse1D_min_float;
extern CkReduction::reducerType sparse1D_min_double;

extern CkReduction::reducerType sparse2D_sum_int;
extern CkReduction::reducerType sparse2D_sum_float;
extern CkReduction::reducerType sparse2D_sum_double;

extern CkReduction::reducerType sparse2D_product_int;
extern CkReduction::reducerType sparse2D_product_float;
extern CkReduction::reducerType sparse2D_product_double;

extern CkReduction::reducerType sparse2D_max_int;
extern CkReduction::reducerType sparse2D_max_float;
extern CkReduction::reducerType sparse2D_max_double;

extern CkReduction::reducerType sparse2D_min_int;
extern CkReduction::reducerType sparse2D_min_float;
extern CkReduction::reducerType sparse2D_min_double;

extern CkReduction::reducerType sparse3D_sum_int;
extern CkReduction::reducerType sparse3D_sum_float;
extern CkReduction::reducerType sparse3D_sum_double;

extern CkReduction::reducerType sparse3D_product_int;
extern CkReduction::reducerType sparse3D_product_float;
extern CkReduction::reducerType sparse3D_product_double;

extern CkReduction::reducerType sparse3D_max_int;
extern CkReduction::reducerType sparse3D_max_float;
extern CkReduction::reducerType sparse3D_max_double;

extern CkReduction::reducerType sparse3D_min_int;
extern CkReduction::reducerType sparse3D_min_float;
extern CkReduction::reducerType sparse3D_min_double;

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

    void add(int x, T data)
    {
      int ind = index;
      // simple insertion sort
      while((ind != 0)&&(records[ind-1].x > x))
      {
        records[ind] = records[ind-1];
        ind--;
      }
      records[ind].x = x;
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
      elem->contribute(size*sizeof(rec), records, sparse1D_sum_int, cb);
    }

    void contributeSum(ArrayElement *elem, const CkCallback &cb, float dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse1D_sum_float, cb);
    }

    void contributeSum(ArrayElement *elem, const CkCallback &cb, double dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse1D_sum_double, cb);
    }

    // contribute to product reducers
    void contributeProduct(ArrayElement *elem, const CkCallback &cb)
    {
      T dummy;
      contributeProduct(elem, cb, dummy);
    }

    void contributeProduct(ArrayElement *elem, const CkCallback &cb, int dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse1D_product_int, cb);
    }

    void contributeProduct(ArrayElement *elem, const CkCallback &cb, float
 dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse1D_product_float, cb);
    }

    void contributeProduct(ArrayElement *elem, const CkCallback &cb, double
 dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse1D_product_double, cb);
    }

    // contribute to max reducers
    void contributeMax(ArrayElement *elem, const CkCallback &cb)
    {
      T dummy;
      contributeMax(elem, cb, dummy);
    }

    void contributeMax(ArrayElement *elem, const CkCallback &cb, int dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse1D_max_int, cb);
    }

    void contributeMax(ArrayElement *elem, const CkCallback &cb, float dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse1D_max_float, cb);
    }

    void contributeMax(ArrayElement *elem, const CkCallback &cb, double dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse1D_max_double, cb);
    }

    // contribute to min reducers
    void contributeMin(ArrayElement *elem, const CkCallback &cb)
    {
      T dummy;
      contributeMin(elem, cb, dummy);
    }

    void contributeMin(ArrayElement *elem, const CkCallback &cb, int dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse1D_min_int, cb);
    }

    void contributeMin(ArrayElement *elem, const CkCallback &cb, float dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse1D_min_float, cb);
    }

    void contributeMin(ArrayElement *elem, const CkCallback &cb, double dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse1D_min_double, cb);
    }

  protected:

    typedef sparseRec1D<T> rec;
    rec *records;
    int size;
    int index;

  private:
    CkSparseReducer1D(){} // should not use the default constructor
};

template <class T>
class CkSparseReducer2D
{
  public:

    CkSparseReducer2D(int numOfElements)
    {
      size = numOfElements;
      index = 0;
      if(size != 0)
        records = new rec[size];
      else
        records = NULL;
    }

    ~CkSparseReducer2D()
    {
      if(records != NULL)
        delete[] records;
    }

    void add(int x, int y, T data)
    {
      int ind = index;
      // simple insertion sort
      while((ind != 0)&&((records[ind-1].y > y) || ((records[ind-1].y == y) &&
            (records[ind-1].x > x))))
      {
        records[ind] = records[ind-1];
        ind--;
      }
      records[ind].x = x;
      records[ind].y = y;
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
      elem->contribute(size*sizeof(rec), records, sparse2D_sum_int, cb);
    }

    void contributeSum(ArrayElement *elem, const CkCallback &cb, float dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse2D_sum_float, cb);
    }

    void contributeSum(ArrayElement *elem, const CkCallback &cb, double dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse2D_sum_double, cb);
    }

    // contribute to product reducers
    void contributeProduct(ArrayElement *elem, const CkCallback &cb)
    {
      T dummy;
      contributeProduct(elem, cb, dummy);
    }

    void contributeProduct(ArrayElement *elem, const CkCallback &cb, int dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse2D_product_int, cb);
    }

    void contributeProduct(ArrayElement *elem, const CkCallback &cb, float
 dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse2D_product_float, cb);
    }

    void contributeProduct(ArrayElement *elem, const CkCallback &cb, double
 dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse2D_product_double, cb);
    }

    // contribute to max reducers
    void contributeMax(ArrayElement *elem, const CkCallback &cb)
    {
      T dummy;
      contributeMax(elem, cb, dummy);
    }

    void contributeMax(ArrayElement *elem, const CkCallback &cb, int dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse2D_max_int, cb);
    }

    void contributeMax(ArrayElement *elem, const CkCallback &cb, float dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse2D_max_float, cb);
    }

    void contributeMax(ArrayElement *elem, const CkCallback &cb, double dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse2D_max_double, cb);
    }

    // contribute to min reducers
    void contributeMin(ArrayElement *elem, const CkCallback &cb)
    {
      T dummy;
      contributeMin(elem, cb, dummy);
    }

    void contributeMin(ArrayElement *elem, const CkCallback &cb, int dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse2D_min_int, cb);
    }

    void contributeMin(ArrayElement *elem, const CkCallback &cb, float dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse2D_min_float, cb);
    }

    void contributeMin(ArrayElement *elem, const CkCallback &cb, double dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse2D_min_double, cb);
    }

  protected:

    typedef sparseRec2D<T> rec;
    rec *records;
    int size;
    int index;

  private:
    CkSparseReducer2D(){} // should not use the default constructor
};

template <class T>
class CkSparseReducer3D
{
  public:

    CkSparseReducer3D(int numOfElements)
    {
      size = numOfElements;
      index = 0;
      if(size != 0)
        records = new rec[size];
      else
        records = NULL;
    }

    ~CkSparseReducer3D()
    {
      if(records != NULL)
        delete[] records;
    }

    void add(int x, int y, int z, T data)
    {
      int ind = index;
      // simple insertion sort
      while((ind != 0) && ((records[ind-1].z > z) || ((records[ind-1].z == z) &&
            (records[ind-1].y > y)) || ((records[ind-1].z == z) &&
            (records[ind-1].y == y) && (records[ind-1].x > x))))
      {
        records[ind] = records[ind-1];
        ind--;
      }
      records[ind].x = x;
      records[ind].y = y;
      records[ind].z = z;
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
      elem->contribute(size*sizeof(rec), records, sparse3D_sum_int, cb);
    }

    void contributeSum(ArrayElement *elem, const CkCallback &cb, float dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse3D_sum_float, cb);
    }

    void contributeSum(ArrayElement *elem, const CkCallback &cb, double dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse3D_sum_double, cb);
    }

    // contribute to product reducers
    void contributeProduct(ArrayElement *elem, const CkCallback &cb)
    {
      T dummy;
      contributeProduct(elem, cb, dummy);
    }

    void contributeProduct(ArrayElement *elem, const CkCallback &cb, int dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse3D_product_int, cb);
    }

    void contributeProduct(ArrayElement *elem, const CkCallback &cb, float
 dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse3D_product_float, cb);
    }

    void contributeProduct(ArrayElement *elem, const CkCallback &cb, double
 dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse3D_product_double, cb);
    }

    // contribute to max reducers
    void contributeMax(ArrayElement *elem, const CkCallback &cb)
    {
      T dummy;
      contributeMax(elem, cb, dummy);
    }

    void contributeMax(ArrayElement *elem, const CkCallback &cb, int dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse3D_max_int, cb);
    }

    void contributeMax(ArrayElement *elem, const CkCallback &cb, float dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse3D_max_float, cb);
    }

    void contributeMax(ArrayElement *elem, const CkCallback &cb, double dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse3D_max_double, cb);
    }

    // contribute to min reducers
    void contributeMin(ArrayElement *elem, const CkCallback &cb)
    {
      T dummy;
      contributeMin(elem, cb, dummy);
    }

    void contributeMin(ArrayElement *elem, const CkCallback &cb, int dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse3D_min_int, cb);
    }

    void contributeMin(ArrayElement *elem, const CkCallback &cb, float dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse3D_min_float, cb);
    }

    void contributeMin(ArrayElement *elem, const CkCallback &cb, double dummy)
    {
      elem->contribute(size*sizeof(rec), records, sparse3D_min_double, cb);
    }

  protected:

    typedef sparseRec3D<T> rec;
    rec *records;
    int size;
    int index;

  private:
    CkSparseReducer3D(){} // should not use the default constructor
};

#endif
