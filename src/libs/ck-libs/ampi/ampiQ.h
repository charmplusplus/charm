#ifndef __AMPIQ_H
#define __AMPIQ_H

#include "charm++.h"

#define MAX_QUEUE_LEN 1024 // must be a power of 2

/*
   AmpiQ is a fixed length queue,which supports "peek at pos" and
   "insert at pos". This queue is optimized to make use of fact that
   its size is always a power of 2.
*/
template <class T>
class AmpiQ {
  T   m_data [MAX_QUEUE_LEN];
  int m_first;
  int m_len;

public:

  AmpiQ () {
    m_first = 0;
    m_len = 0;
  }

  int length (void) { return m_len; }

  // return (length () == 0)
  bool isEmpty (void) { return  (0 == m_len); }

  // peek at the nth item from queue
  T& operator[] (size_t n) {
    n = ((n+m_first) & (MAX_QUEUE_LEN-1));
    return m_data [n];
  }

  T deq (void) {
    if (m_len>0) {
      T &ret = m_data [m_first];
      m_first = ((m_first+1) & (MAX_QUEUE_LEN-1));
      m_len --;
      return ret;
    } 
    else 
      return T(); //For builtin types like int, void*, this is equivalent to T(0)
  }

  void insert (int pos, T& elt) {

    if (m_len==MAX_QUEUE_LEN || pos>=MAX_QUEUE_LEN) {
      CkAbort ("error: queue overflow\n"); // error
    }

    for (int i=m_len; i>pos; i--)
      m_data[(i+m_first)&(MAX_QUEUE_LEN-1)]=m_data[(i-1+m_first)&(MAX_QUEUE_LEN-1)];
    m_data [(pos+m_first)&(MAX_QUEUE_LEN-1)] = elt;
    if (pos > m_len) m_len = pos+1;
      else m_len++;
  }

  void enq (const T& elt) {
    if(m_len==MAX_QUEUE_LEN)
      CkAbort ("error: queue overflow\n");
    m_data [(m_first+m_len)&(MAX_QUEUE_LEN-1)] = elt;
    m_len++;
  }
};
#endif
