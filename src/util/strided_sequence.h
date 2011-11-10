#ifndef STRIDED_SEQUENCE_H_
#define STRIDED_SEQUENCE_H_

#include "cksequence_internal.h"
#include "pup.h"

#include <vector>
#include <iostream>

template <typename T>
class StridedIterator : public CkSequenceIteratorInternal<T> {
 public:
  T element_;
  T stride_;

  StridedIterator() {
  }

  StridedIterator(T element) : element_(element), stride_(0) {
  }

  StridedIterator(T element, T stride) : element_(element), stride_(stride) {
  }

  T operator*() {
    return element_;
  }

  void operator++() {
    element_ += stride_;
  }

  void operator++(int) {
    element_ += stride_;
  }

  void operator--() {
    element_ -= stride_;
  }

  void operator--(int) {
    element_ -= stride_;
  }

  bool operator==(const CkSequenceIteratorInternal<T>& rhs) const {
    return (this->element_ == ((StridedIterator*)&rhs)->element_);
  }

  bool operator!=(const CkSequenceIteratorInternal<T>& rhs) const {
    return (this->element_ != ((StridedIterator*)&rhs)->element_);
  }

  

};

template <typename T>
class StridedSequence : public CkSequenceInternal<T> {
 public:
  StridedSequence() : start_element_(1), stride_(1), last_element_(1) {
  }

  StridedSequence(T start_element, T stride, int end_element) :
      start_element_(start_element),
      stride_(stride),
      last_element_(end_element) {
  }

  void Insert(const T& element);

  void Remove(const T& element);
  
  int num_elements() const;

  T min() const {
    return start_element_;
  }

  T max() const {
    return last_element_;
  }

  T stride() const {
    return stride_;
  }

  Type type() const { 
    return STRIDE;
  }

  int mem_size() const;

  CkSequenceIteratorInternal<T>* begin() {
    return new StridedIterator<T>(start_element_, stride_);
  }

  CkSequenceIteratorInternal<T>* end() {
    return new StridedIterator<T>(last_element_ + stride_);
  }

  void pup(PUP::er &p) {
    p|start_element_;
    p|stride_;
    p|last_element_;
  }

 private:
  T start_element_;
  T stride_;
  T last_element_;
};

template <class T>
inline void StridedSequence<T>::Insert(const T& element) {
}

template <class T>
inline void StridedSequence<T>::Remove(const T& element) {
}

template <class T>
inline int StridedSequence<T>::num_elements() const {
  return (((last_element_ - start_element_) / stride_)  + 1);
}

template <class T>
inline int StridedSequence<T>::mem_size() const {
  return sizeof(T)*3;
}

#endif   // STRIDED_SEQUENCE_H_
