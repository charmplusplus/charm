#ifndef SEQUENCE_INTERNAL_H
#define SEQUENCE_INTERNAL_H

#include <vector>
#include "pup.h"

  /**
  * There are two types of sequences, RANDOM and STRIDE.
  */
  enum Type {RANDOM, STRIDE};
template <typename T>
struct StrideInfo {
  T start_element;
  T stride;
  T end_element;
};

/**
* Interface(Abstract Class) for the Internal implementation of the Iterator
*
* @tparam T
*/
template <typename T>
class CkSequenceIteratorInternal {
 public:


  virtual T operator*() = 0;
  virtual void operator++() = 0;
  virtual void operator++(int) = 0;
  virtual void operator--() = 0;
  virtual void operator--(int) = 0;
  virtual bool operator==(const CkSequenceIteratorInternal& rhs) const = 0;
  virtual bool operator!=(const CkSequenceIteratorInternal& rhs) const = 0;
};

/**
* Interface(Abstract class) for the Internal implementation of the CkSequence
*
* @tparam T
*/
template <typename T>
class CkSequenceInternal {
 public:


  virtual void Insert(const T& element) = 0;

  virtual void Remove(const T& element) = 0;

  virtual int num_elements() const = 0;

  virtual T min() const = 0;

  virtual T max() const = 0;

  virtual Type type() const = 0;

  virtual CkSequenceIteratorInternal<T>* begin() = 0;

  virtual CkSequenceIteratorInternal<T>* end() = 0;

  // Returns the size of the memory for internal representation of the elements
  virtual int mem_size() const = 0;
  virtual void pup(PUP::er &p) = 0;
};

#endif  // SEQUENCE_INTERNAL_H
