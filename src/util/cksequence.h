#ifndef SEQUENCE_H_
#define SEQUENCE_H_

#include "cksequence_factory.h"
#include "cksequence_internal.h"
#include "pup.h"

#include <list>
#include <math.h>

#define CHAR_SIZE 8

/**
* Iterates over the elements in the CkSequence. Consists of the standard
* functionalities of an iterator.
* Sample usage:
*   CkSequence<int>::iterator it  = s.begin();
*
* @tparam T
*/
template <typename T>
class CkSequenceIterator {
 public:

  CkSequenceIterator() {}
  
  CkSequenceIterator(CkSequenceIteratorInternal<T>* it) : it_(it) {}

  /**
  * Constructs the CkSequenceIterator given an internal implementation of the
  * CkSequenceIterator
  *
  * @Param it internal implementation of the Iterator
  */
  CkSequenceIterator(typename std::list<CkSequenceInternal<T> *>::iterator
       subsequence_list_it,
       typename std::list<CkSequenceInternal<T> *>::iterator
       subsequence_list_it_end) {
    subsequence_list_it_ = subsequence_list_it;
    subsequence_list_it_end_ = subsequence_list_it_end;
    if (subsequence_list_it_ == subsequence_list_it_end_) {
      it_ = NULL;
      it_end_ = NULL;
    } else {
      it_ = (*subsequence_list_it_)->begin();
      it_end_ = (*subsequence_list_it_)->end();
    }
  }

  ~CkSequenceIterator() {
//    delete it_;
  }

  T operator*() {
    return **it_;
  }

  void operator++() {
    ++(*it_);
    if (*it_ == *it_end_) {
      if (++subsequence_list_it_ == subsequence_list_it_end_) {
        it_ = NULL;
        it_end_ = NULL;
      } else {
        it_ = (*subsequence_list_it_)->begin();
        it_end_ = (*subsequence_list_it_)->end();
      }
    }
  }

  void operator++(int) {
    ++(*it_);
    if (*it_ == *it_end_) {
      if (++subsequence_list_it_ == subsequence_list_it_end_) {
        it_ = NULL;
        it_end_ = NULL;
      } else {
        it_ = (*subsequence_list_it_)->begin();
        it_end_ = (*subsequence_list_it_)->end();
      }
    }

  }

  bool operator==(const CkSequenceIterator& rhs) const {
    if (it_ == NULL || rhs.it_ == NULL) {
      return (this->it_ == rhs.it_);
    }
    return (*(this->it_) == *(rhs.it_));
  }

  bool operator!=(const CkSequenceIterator& rhs) const {
    if (it_ == NULL || rhs.it_ == NULL) {
      return (this->it_ != rhs.it_);
    }
    return (*(this->it_) != *(rhs.it_));
  }

 private:
  CkSequenceIteratorInternal<T>* it_;
  CkSequenceIteratorInternal<T>* it_end_;
  typename std::list<CkSequenceInternal<T>* >::iterator subsequence_list_it_;
  typename std::list<CkSequenceInternal<T>* >::iterator subsequence_list_it_end_;
  int index_;
};

/**
* Data Structure to store a sequence of any type, typically int, short, long
* etc. Two types of CkSequences are currently supported, RANDOM and STRIDE. By
* default, a RandomCkSequence is created. This class Delegates the calls to the
* internal implementation of CkSequence
*
* Sample Usage:
*   CkSequence<int> s_default;
*   CkSequence<int> seq_random(CkSequence<int>::RANDOM);
*
* @tparam T
*/
template <typename T>
class CkSequence {

 public:


  /**
  * Creates a RandomSequence by default
  */
  CkSequence() : bit_vector_(NULL), compact_(false) {}

  /**
  * Creates Sequence object based on the vector passed in. The default sequence
  * created is RandomSequence.
  *
  * @Param l containing the elements to be stored
  */
  template <typename GenericIterator>
  CkSequence(const GenericIterator begin, const GenericIterator end) {
    compact_ = false;
    min_ = 0;
    max_ = 0;
    num_elements_ = 0;
    if (begin == end) {
      return;
    }

    // Find the min and the max element
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

    // Allocate memory for the bit vector
    bit_vector_ = (char *) malloc ((max_+1)/CHAR_SIZE + 1);
    memset(bit_vector_, 0, (max_+1)/CHAR_SIZE + 1);

    for (GenericIterator it = begin; it != end; ++it) {
      Set(bit_vector_, (*it));
    }
  }

  template <typename GenericIterator>
  void Insert(const GenericIterator begin, const GenericIterator end) {
    compact_ = false;
    min_ = 0;
    max_ = 0;
    num_elements_ = 0;
    if (begin == end) {
      return;
    }

    // Find the min and the max element
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

    // Allocate memory for the bit vector
    bit_vector_ = (char *) malloc ((max_+1)/CHAR_SIZE + 1);
    memset(bit_vector_, 0, (max_+1)/CHAR_SIZE + 1);

    for (GenericIterator it = begin; it != end; ++it) {
      Set(bit_vector_, (*it));
    }
  }


  ~CkSequence() {
    if (bit_vector_ != NULL) {
      delete bit_vector_;
      bit_vector_ = NULL;
    }
    for (int i = 0; i < subsequence_list_.size(); i++) {
      //delete subsequence_list_[i];
    }
  }

  /**
  * Inserts the element in to the CkSequence Data Structure.
  *
  * @Param element to be inserted
  */
  void Insert(const T& element);

  void InsertIntoStrided(typename std::list<CkSequenceInternal<T> *>::iterator
      iter, const T& element);

  /**
  * Removes the element from the CkSequence Data Structure
  *
  * @Param element to be removed
  */
  void Remove(const T& element);

  /**
  * Called when the elements have been inserted into the sequence and no further
  * modification would happen
  */
  void DoneInserting();

  /**
  * Identifies if the sequence has a stride pattern and if so returns true and
  * sets stride to the identified stride.
  *
  * @Param stride sets it to be the stride for the given sequence
  *
  * @Returns true if there is a stride pattern else false
  */

  int num_elements() const;

  int mem_size() const;

  typedef CkSequenceIterator<T> iterator;

  /**
  * Returns the begin of the CkSequence.
  * Sample Usage:
  *   CkSequence<int>::iterator it = s.begin();
  *
  * @Returns  the iterator pointing to the begin
  */
  iterator begin() {
    if (compact_) {
      return iterator(subsequence_list_.begin(), subsequence_list_.end());
    } else {
      return iterator(new BitVectorIterator<T>(bit_vector_, 0, min_, max_));
    }
  }

  /**
   * Returns the end of the CkSequence
   * Sample Usage:
   *   CkSequence<int>::iterator it = s.end();
   *
   * @Returns   the iterator to the end
   */
  iterator end() {
    if (compact_) {
      return iterator(NULL);
    } else {
      return iterator(new BitVectorIterator<T>(bit_vector_, 0, max_+1, max_));
    }
  }

  void pup(PUP::er &p) {
    p|min_;
    p|max_;
    p|num_elements_;
    p|compact_;
    if (p.isUnpacking() && !compact_) {
      bit_vector_ = (char *) malloc((max_+1)/CHAR_SIZE + 1);
    }
    if (!compact_) {
      PUParray(p, bit_vector_, (max_+1)/CHAR_SIZE + 1);
    }
    if (!p.isUnpacking()) {
      int size = subsequence_list_.size();
      p|size;
      int type;
      for (typename std::list<CkSequenceInternal<T>*>::iterator it =
          subsequence_list_.begin(); it != subsequence_list_.end(); it++) {
        type = (*it)->type();
        p|type;
        p|(**it);
      }
    } else {
      int size;
      int type;
      p|size;

      for (int i=0; i < size; i++) {
        p|type;
        CkSequenceInternal<T>* seq_internal;
        if (type == STRIDE) {
          seq_internal = new StridedSequence<T>();
        } else {
          seq_internal = new RandomSequence<T>();
        }
        p|(*seq_internal);
        subsequence_list_.push_back(seq_internal);
      }
    }
  }

 private:

  void Compact();

  T min_;
  T max_;
  T num_elements_;

  // Storing the sequence 
  bool compact_;
  char* bit_vector_;

  // Contains the combination of different types of sequences namely Random and
  // Strided. 
  std::list<CkSequenceInternal<T>*> subsequence_list_;
};


template <class T>
inline void CkSequence<T>::Insert(const T& element) {
  if (compact_) {
    CkAbort("Cannot insert after DoneInserting() is called\n");
  }
  if ((element/CHAR_SIZE) > ((max_+1)/CHAR_SIZE)) {
    bit_vector_ = (char *) realloc(bit_vector_, (element+1)/CHAR_SIZE + 1);
    int diff = ((element + 1) / CHAR_SIZE) - ((max_+1) / CHAR_SIZE);
    memset(&bit_vector_[((max_+1)/CHAR_SIZE) + 1], 0, diff);
  }

  Set(bit_vector_, element);

  if (element > max_) {
    max_ = element;
  } else if (element < min_) {
    min_ = element;
  }
}

template <class T>
inline void CkSequence<T>::Remove(const T& element) {
  if (compact_) {
    CkAbort("Cannot insert after DoneInserting() is called\n");
  }
  Reset(bit_vector_, element);
}

template <class T>
inline int CkSequence<T>::num_elements() const {
  if (subsequence_list_.size() <= 0) {
    return 0;
  }
//  return subsequence_list_[0]->num_elements();
}

template <class T>
inline int CkSequence<T>::mem_size() const {
  int sum = 0;
  for (int i = 0; i < subsequence_list_.size(); ++i) {
 //   sum += subsequence_list_[i]->mem_size();
  }
  return sum;
}

template <typename T>
inline void CkSequence<T>::Compact() {
  compact_ = true;
  std::cout << "Compacting!!!\n";
  int seq_begin_ele = min_;
  int prev_ele = min_;
  int prev_prev_ele = min_;
  int prev_prev_prev_ele = min_;

  int end_stride = min_;

  int start_random = min_;
  int end_random = min_;

  int stride = -1;
  std::cout << "Current size " << ((max_+1)/CHAR_SIZE + 1) << std::endl;


  int tmp_stride = 0;
  bool is_strided = false;

  for (T i = min_+ 1; i <= max_; i++) {
    if (IsSet(bit_vector_,i)) {
      tmp_stride = i - prev_ele;

      if (tmp_stride == stride) {
        // Start of strided pattern
        if (!is_strided && seq_begin_ele != prev_prev_ele) {
          std::cout << "Create random " << seq_begin_ele  << " end " << prev_prev_prev_ele<< std::endl;
          CkSequenceInternal<T>* sequence_internal =
            CkSequenceFactory<T>::CreateRandomSequence(bit_vector_,
                seq_begin_ele, prev_prev_prev_ele);
          subsequence_list_.push_back(sequence_internal);
          seq_begin_ele = prev_prev_ele;
        }

        is_strided = true;
        end_stride = i;
      } else if (tmp_stride != stride) {
        // End of stride pattern
        if (is_strided) {
          std::cout << "Create stride " << seq_begin_ele<< " stride " << stride
            << " end stride " << end_stride  << " new seq end " << i << " count " << (end_stride - seq_begin_ele)/stride + 1<< std::endl;

          CkSequenceInternal<T>* sequence_internal =
            CkSequenceFactory<T>::CreateStridedSequence(seq_begin_ele, stride,
                end_stride);
          subsequence_list_.push_back(sequence_internal);
          seq_begin_ele = i;
          prev_prev_prev_ele = i;
          prev_prev_ele = i;
          tmp_stride = -1;
        }

        start_random = i;
        is_strided = false;
      }
      prev_prev_prev_ele = prev_prev_ele;
      prev_prev_ele = prev_ele;
      stride = tmp_stride;
      prev_ele = i;
    }
  }

  if (is_strided) {
    std::cout << "Create stride " << seq_begin_ele<< " stride " << stride
      << " end stride " << end_stride  << " count " << (end_stride - seq_begin_ele)/stride + 1<< std::endl;
    CkSequenceInternal<T>* sequence_internal =
      CkSequenceFactory<T>::CreateStridedSequence(seq_begin_ele, stride, end_stride);
    subsequence_list_.push_back(sequence_internal);
  } else {
    CkSequenceInternal<T>* sequence_internal =
      CkSequenceFactory<T>::CreateRandomSequence(bit_vector_, seq_begin_ele,start_random);
    subsequence_list_.push_back(sequence_internal);
  }
  delete bit_vector_;
  bit_vector_ = NULL;
}

template <class T>
inline void CkSequence<T>::DoneInserting() {
  std::cout << "Done inserting\n";
  Compact();
}

#endif // SEQUENCE_H_
