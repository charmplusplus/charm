/*
   This file defines interface for sparse contiguous 2D reducers.
   These reducers handle the parts of 2D arrays contributed by 
   different chare elements. Right now, the reducers assume that 
   contributed sections of arrays either do not overlap or completely
   overlap.
                              Vikas Mehta, vmehta1@uiuc.edu, 06/03/03
*/

#ifndef _CKSPARSECONTIGUOUSREDUCER_H
#define _CKSPARSECONTIGUOUSREDUCER_H

#include "CkSparseContiguousReducer.decl.h"

/*
   types of reducers supported
*/
extern CkReduction::reducerType sparse_sum_int;
extern CkReduction::reducerType sparse_sum_float;
extern CkReduction::reducerType sparse_sum_double;
extern CkReduction::reducerType sparse_sum_TwoFloats;
extern CkReduction::reducerType sparse_sum_TwoDoubles;

extern CkReduction::reducerType sparse_product_int;
extern CkReduction::reducerType sparse_product_float;
extern CkReduction::reducerType sparse_product_double;

extern CkReduction::reducerType sparse_max_int;
extern CkReduction::reducerType sparse_max_float;
extern CkReduction::reducerType sparse_max_double;

extern CkReduction::reducerType sparse_min_int;
extern CkReduction::reducerType sparse_min_float;
extern CkReduction::reducerType sparse_min_double;

class CkTwoDoubles{
 public:
  double d1, d2;
  CkTwoDoubles(double _d1=0, double _d2=0) :d1(_d1), d2(_d2){}
  void operator+=(CkTwoDoubles d){d1+=d.d1; d2+=d.d2;}
};

class CkTwoFloats{
 public:
  float f1, f2;
  CkTwoFloats(float _f1=0, float _f2=0) :f1(_f1), f2(_f2){}
  void operator+=(CkTwoFloats f){f1+=f.f1; f2+=f.f2;}
};

class CkDataSegHeader{
 public:
  /* 
     for a 2D Array, sx,  sy, ex, ey are start x and y and end x and y indices
     respectively.
  */
  int sx, sy, ex, ey;

  CkDataSegHeader(){
  }

  CkDataSegHeader(int _sx, int _sy, int _ex, int _ey): sx(_sx), sy(_sy), ex(_ex), ey(_ey) {}

  CkDataSegHeader(const CkDataSegHeader &r) : sx(r.sx), sy(r.sy), ex(r.ex), ey(r.ey) {}


  inline bool operator==(const CkDataSegHeader &r){
    if((r.sx == sx)&&(r.sy == sy)&&(r.ex == ex)&&(r.ey == ey))
      return true;

    return false;
  }

  /* 
     overloaded '<' and '>' operators to keep the headers x-aligned
  */
  inline bool operator<(const CkDataSegHeader &r){
    if(sx < r.sx)
      return true;
    else
      if((sx == r.sx)&&(ex < r.ex))
	return true;
      else
	if((sx == r.sx)&&(ex == r.ex)&&(sy < r.sy))
	  return true;
	else
	  if((sx == r.sx)&&(ex == r.ex)&&(sy == r.sy)&&(ey < r.ey))
	    return true;

    return false;
  }

  /*inline bool operator>(const CkDataSegHeader &r){
    if(sx > r.sx)
      return true;
    else
      if((sx == r.sx)&&(ex > r.ex))
	return true;

    return false;
  }*/

  /* 
     returns the number of elements in this data segment 
  */
  inline int getNumElements(){
    return (ex-sx+1)*(ey-sy+1);
  }
};

/* 
   To contribute a 2D array of data, use an object of 
   CkSparseContiguousReducer<T>
*/
template <class T>
class CkSparseContiguousReducer
{
  CkDataSegHeader r;
  T *data;

 public:
  CkSparseContiguousReducer(){}
  ~CkSparseContiguousReducer(){} // delete all the nodes in list

  /* 
     Accepts starting x,y and end x,y indices of the 2D data.
     the data buffer accepted by this function must be on row 
     major order. This function must be called once.
  */
  inline void add(int _sx, int _sy, int _ex, int _ey, T *_data){
    r.sx = _sx;
    r.sy = _sy;
    r.ex = _ex;
    r.ey = _ey;
    data = _data;
  }
   
  /* 
     This function is same as contribute. The only difference is that, 
     it contributes data "added" to this object
  */
  void contribute(ArrayElement *elem, CkReduction::reducerType type, const
		  CkCallback &cb){
    int dataAlignSize = sizeof(int)*2 + sizeof(CkDataSegHeader);
    if(dataAlignSize%sizeof(double)) {
      dataAlignSize += sizeof(double) - (dataAlignSize%sizeof(double));
    }

    int size = r.getNumElements()*sizeof(T) + dataAlignSize;
    CkReductionMsg* msg = CkReductionMsg::buildNew (size, NULL, type);
    msg->setCallback(cb);

    unsigned char *ptr = (unsigned char*)(msg->getData());
    int count = 1;
    /* pack data */
    memcpy(ptr, &count, sizeof(int));
    memcpy(ptr+sizeof(int), &dataAlignSize, sizeof(int));
    memcpy(ptr + 2*sizeof(int), &r, sizeof(CkDataSegHeader));
    memcpy(ptr + 2*sizeof(int) + sizeof(CkDataSegHeader), data,
	   r.getNumElements()*sizeof(T));
    /* contribute on behalf of chare calling this function */
    elem->contribute(msg);
  }

  /* 
     This function is same as contribute. The only difference is that, 
     it contributes data "added" to this object
  */
  void contribute(ArrayElement *elem, CkReduction::reducerType type){
    CkCallback cb;
    contribute (elem, type, cb);
  }
};

/* 
   returns number of non null data segments in the buffer 
*/
int numDataSegs(const unsigned char *data);

/* 
   returns an instance of CkDataSegHeader having indices for index(th) data 
   segment 
*/
CkDataSegHeader getDataSegHeader(int index, const unsigned char *data);

/* 
   returns an instance of CkDataSegHeader having indices for index(th) data 
   segment 
*/
CkDataSegHeader* getDataSegHeaderPtr(const unsigned char *data);

/* 
   returns pointer to data portion in the message. All the data segments are 
   arranged in row major order in the data portion of the buffer, starting at 
   the returned pointer
*/
unsigned char * getDataPtr(unsigned char *ptr);

/*
   expands the message returned by reduction, into a 2D array returned as a 1D
   array in row major order. It returns the size of array implicitly in 
   'CkDataSegHeader' input to the function.

   NOTE: while passing the 'nullValue' to this function, ensure to type cast it
   to pass proper data type as these are overloaded functions and output will
   be defined based to on the function which is called.
*/
int *decompressMsg(CkReductionMsg *m, CkDataSegHeader &h, int nullVal);

float *decompressMsg(CkReductionMsg *m, CkDataSegHeader &h, float nullVal);

double *decompressMsg(CkReductionMsg *m, CkDataSegHeader &h, double nullVal);

CkTwoDoubles *decompressMsg(CkReductionMsg *m, CkDataSegHeader &h, CkTwoDoubles nullVal);

CkTwoFloats *decompressMsg(CkReductionMsg *m, CkDataSegHeader &h, CkTwoFloats nullVal);

#endif
