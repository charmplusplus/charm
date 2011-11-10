#ifndef SEQUENCE_FACTORY_H_
#define SEQUENCE_FACTORY_H_

#include "random_sequence.h"
#include "cksequence_internal.h"
#include "strided_sequence.h"

/**
* Factory that creates different kinds of CkSequences. Currently, there are two
* types of CkSequences, namely, RANDOM and STRIDE
*
* @tparam T
*/
template <typename T>
class CkSequenceFactory {
 public:
  static CkSequenceInternal<T>* CreateRandomSequence() {
    return new RandomSequence<T>();
  }

  template <typename GenericIterator>
  static CkSequenceInternal<T>* CreateRandomSequence(const GenericIterator& begin,
      const GenericIterator& end) {
    return new RandomSequence<T>(begin, end);
  }

  static CkSequenceInternal<T>* CreateRandomSequence(char* bit_vector, int
      start_ele, int end_ele) {
    return new RandomSequence<T>(bit_vector, start_ele, end_ele);
  }

  static CkSequenceInternal<T>* CreateStridedSequence() {
    return new StridedSequence<T>();
  }

  static CkSequenceInternal<T>* CreateStridedSequence(T start_element, T stride,
      T end_element) {
    return new StridedSequence<T>(start_element, stride, end_element);
  }
};

#endif   // SEQUENCE_FACTORY_H_
