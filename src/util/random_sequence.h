#ifndef RANDOM_SEQUENCE_H_
#define RANDOM_SEQUENCE_H_

#include "cksequence_internal.h"
#include "pup.h"

#include <vector>
#include <algorithm>
#include <string.h>
#include <iostream>

#define Set(a, ind) a[(ind/8)] = (a[(ind/8)] | (1<<(ind%8)))
#define Reset(a, ind) a[(ind/8)] = (a[(ind/8)] & (~(1<<(ind%8))))
#define IsSet(a, ind) (a[(ind/8)] & (1<<(ind%8)))

/**
* Iterator for the RandomSequence   
*
* @tparam T
*/
template <typename T>
class RandomIterator : public CkSequenceIteratorInternal<T> {
 public:
  typename std::vector<T>::iterator it_;

  RandomIterator() {}

  RandomIterator(typename std::vector<T>::iterator it) : it_(it) {}

  T& operator*() {
    return *it_;
  }

  void operator++() {
    ++it_;
  }

  void operator++(int) {
    ++it_;
  }

  void operator--() {
    --it_;
  }

  void operator--(int) {
    --it_;
  }

  bool operator==(const CkSequenceIteratorInternal<T>& rhs) const {
    return (this->it_ == ((RandomIterator *)&rhs)->it_);
  }

  bool operator!=(const CkSequenceIteratorInternal<T>& rhs) const {
    return (this->it_ != ((RandomIterator *)&rhs)->it_);
  }
};

template <typename T>
class BitVectorIterator : public CkSequenceIteratorInternal<T> {
 public:
  BitVectorIterator() {}

  BitVectorIterator(char*& bit_vector, int start, int index, int max) :
      bit_vector_(bit_vector), start_(start), index_(index), max_(max) {
    while ((index_ < max_ + 1) && !IsSet(bit_vector_, index_)) {
      index_++;
    }
  }

  T operator*() {
    return (start_ + index_);
  }

  void operator++() {
    while ((++index_ < max_+1) && !IsSet(bit_vector_, index_)) {
    }
  }

  void operator++(int) {
    while ((++index_ < max_+1) && !IsSet(bit_vector_, index_)) {
    }
  }

  void operator--() {
  }

  void operator--(int) {
  }

  bool operator==(const CkSequenceIteratorInternal<T>& rhs) const {
    return (this->bit_vector_ == ((BitVectorIterator *)&rhs)->bit_vector_ &&
        this->index_ == ((BitVectorIterator *)&rhs)->index_);
  }

  bool operator!=(const CkSequenceIteratorInternal<T>& rhs) const {
    return (this->bit_vector_ != ((BitVectorIterator *)&rhs)->bit_vector_ ||
        this->index_ != ((BitVectorIterator *)&rhs)->index_);
  }

 private:
  char* bit_vector_;
  int start_;
  int index_;
  int max_;
};


/**
*
* @tparam T
*/
template <typename T>
class RandomSequence : public CkSequenceInternal<T> {

 public:
  
  RandomSequence() {
  }

  RandomSequence(char*& bit_vector, int start, int end) {
    min_ = start % 8;
    start_ = start - min_;
    max_ = min_ + (end - start);
    std::cout << "starting element " << start_ << " ending ele " << end << " max " << max_ << " min " << min_ << std::endl;
    bit_vector_ = (char *) malloc ((max_+1)/8 + 1);
    memcpy(bit_vector_, &bit_vector[(start/8)], (max_+1)/8 + 1);
  }

  template <typename GenericIterator>
  RandomSequence(const GenericIterator& begin, const GenericIterator& end) {
    num_elements_ = 0;
    max_ = 0;
    if (begin == end) {
      return;
    }
    min_ = *begin;
    for (GenericIterator it = begin; it != end; ++it) {
      num_elements_++;
      if (max_ < *it) {
        max_ = *it;
      }
      if (*it < min_) {
        min_ = *it;
      }
    }
    max_;
    std::cout << "max " << max_ << std::endl;
    bit_vector_ = (char *) malloc ((max_+1)/8 + 1);
    memset(bit_vector_, 0, (max_+1)/8 + 1);

    for (GenericIterator it = begin; it != end; ++it) {
      Set(bit_vector_, (*it));
    }
    std::cout << "Malloc bits " << ((max_+1)/8 + 1) << std::endl;
  }

  ~RandomSequence() {
  }

  void Insert(const T& element);

  void Remove(const T& element);

  int num_elements() const;

  int mem_size() const;

  T min() const {
    return start_;
  }

  T max() const {
    return start_ + max_;
  }

  Type type() const {
    return RANDOM;
  }

  CkSequenceIteratorInternal<T>* begin() {
    return new BitVectorIterator<T>(bit_vector_, start_, min_, max_);
  }

  CkSequenceIteratorInternal<T>* end() {
    return new BitVectorIterator<T>(bit_vector_, start_, max_+1, max_);
  }

  void pup(PUP::er &p) {
    p|num_elements_;
    p|start_;
    p|min_;
    p|max_;
    if (p.isUnpacking()) {
      bit_vector_ = (char *) malloc ((max_+1)/8 + 1);
    }
    PUParray(p, bit_vector_, ((max_+1)/8 + 1));
  }

 private:
  int num_elements_;
  T start_;
  T min_;
  T max_;
  char* bit_vector_;
};

template <typename T>
inline void RandomSequence<T>::Insert(const T& element) {
  int ele_ind = element - start_;
  if (ele_ind/8 > (max_+1)/8) {
    int diff = ((ele_ind + 1) / 8) - (( max_ + 1) / 8);
    bit_vector_ = (char *) realloc(bit_vector_, (ele_ind+1)/8 + 1);
    memset(&bit_vector_[((max_+1)/8) + 1], 0, diff);
  }
  Set(bit_vector_, ele_ind);
  if (ele_ind > max_) { 
    max_ = ele_ind;
  }
}

template <typename T>
inline void RandomSequence<T>::Remove(const T& element) {
}

template <typename T>
inline int RandomSequence<T>::num_elements() const {
  return num_elements_;
}

template <typename T>
inline int RandomSequence<T>::mem_size() const {
//  return sizeof(T) * container_.size();
  return ((max_+1)/8 + 3);
}


#endif  // RANDOM_SEQUENCE_H_
