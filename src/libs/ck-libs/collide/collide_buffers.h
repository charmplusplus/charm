/*
Ancient, hideous custom memory management classes.
FIXME: replace these with something well-tested,
standard, and modern, like std::vector.

Orion Sky Lawlor, olawlor@acm.org, 2001/2/5
*/
#ifndef __UIUC_CHARM_COLLIDE_BUFFERS_H
#define __UIUC_CHARM_COLLIDE_BUFFERS_H

#include <stdlib.h> /* for size_t */

//A simple extensible untyped chunk of memory.
//  Lengths are in bytes
class memoryBuffer {
 public:
  typedef unsigned int size_t;
 private:
  void *data;
  size_t len;//Length of data array above
  void setData(const void *toData,size_t toLen);//Reallocate and copy
 public:
  memoryBuffer();//Empty initial buffer
  memoryBuffer(size_t initLen);//Initial capacity specified
  ~memoryBuffer();//Deletes heap-allocated buffer
  memoryBuffer(const memoryBuffer &in) {data=NULL;setData(in.data,in.len);}
  memoryBuffer &operator=(const memoryBuffer &in) {setData(in.data,in.len); return *this;}
  
  size_t length(void) const {return len;}
  void *getData(void) {return data;}
  const void *getData(void) const {return data;}
  void detachBuffer(void) {data=NULL;len=0;}
  
  void resize(size_t newlen);//Reallocate, preserving old data
  void reallocate(size_t newlen);//Free old data, allocate new
};

//Superclass for simple flat memory containers
template <class T> class bufferT {
  T *data; //Data items in container
  int len; //Index of last valid member + 1
 protected:
  bufferT() :data(NULL),len(0) {}
  bufferT(T *data_,int len_) :data(data_),len(len_) {}
  void set(T *data_,int len_) {data=data_;len=len_;}
  void setData(T *data_) {data=data_;}
  void setLength(int len_) {len=len_;}
  int &getLength(void) {return len;}
 public:
  int length(void) const {return len;}
  
  T &operator[](int t) {return data[t];}
  const T &operator[](int t) const {return data[t];}
  T &at(int t) {return data[t];}
  const T &at(int t) const {return data[t];}
  
  T *getData(void) {return (T *)data;}
  const T *getData(void) const {return (T *)data;}
};

//For preallocated data
template <class T> class fixedBufferT : public bufferT<T> {
 public:
  fixedBufferT(T *data_,int len_) :bufferT<T>(data_,len_) {}
};

//Variable size buffer
//T's constructors/destructors are not called by this (use std::vector)
//Copying the memory of a T must be equivalent to copying a T.
template <class T> class growableBufferT : public bufferT<T> {
  typedef bufferT<T> super;
  enum { sT=sizeof(T) };
  memoryBuffer buf;//Data storage
  int max;//Length of storage buffer
  //Don't use these:
  growableBufferT<T>(const growableBufferT<T> &in);
  growableBufferT<T> &operator=(const growableBufferT<T> &in);
 public:
  growableBufferT<T>() :buf() {max=0;}
  growableBufferT<T>(size_t Len) :buf(Len*sT) {this->set((T*)buf.getData(),Len);max=Len;}
    
  inline int length(void) const {return super::length();}
  inline int size(void) const {return super::length();}
  inline int &length(void) {return this->getLength();}
  inline int capacity(void) const {return max;}
  
  T *detachBuffer(void) {
    T *ret=(T *)buf.getData();
    buf.detachBuffer();
    this->set(NULL,0);
    max=0;
    return ret;
  }
  void empty(void) {reallocate(0);}
  void push_back(const T& v) {
    grow(length()+1);
    this->at(this->getLength()++)=v;
  }
  //Push without checking bounds
  void push_fast(const T& v) {
    grow(length()+1);
    this->at(this->getLength()++)=v;
  }
  void grow(int min) {
    if (min>max) {
      if (min > 1000000) {
	//printf("COLLIDE: Buffer size %d is getting out of hand, switching to conservative mechanism!\n", min);
	atLeast(min+(min/4));
      }
      else {
	resize(min+max+8);
      }
    }
  }
  void atLeast(int min) {//More conservative version of grow
    if (min>max) resize(min);
  }
  void resize(int Len) {
    buf.resize(Len*sT);
    this->setData((T*)buf.getData());
    max=Len;
  }
  void reallocate(int Len) {
    buf.reallocate(Len*sT);
    setData((T*)buf.getData());
    this->setLength(0);
    max=Len;
  }
};

#endif
