/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _TStrideArray_H_
#define _TStrideArray_H_

template <class T> 
class TStrideArray {
  private:
    T *array;
    int start, end, stride;
  public:
    TStrideArray<T>(int s, int e, int st=1) {
      int size = (e-s+1)/st;
      array = new T[size];
      start = s; end = e; stride = st;
    }
    T& operator[](int idx) {
      return array[(idx-start)/stride];
    }
    ~TStrideArray<T>(void) {
      delete[] array;
    }
};

#endif
