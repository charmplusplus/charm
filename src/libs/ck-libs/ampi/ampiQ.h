#ifndef __AMPIQ_H
#define __AMPIQ_H

#include "charm++.h"

/*
   AmpiQ is a dynamic length queue, however length must always be a power of 2.
   It supports "peek at pos" and "insert at pos". This queue is optimized to 
   make use of fact that its size is always a power of 2.
*/
template <class T>
class AmpiQ : private CkSTLHelper<T>, private CkNoncopyable{
  T   *m_data;
  int m_first;
  int m_len;
  int m_blklen;

  void _expand(void) {
    int newlen=m_blklen*2;
    T *newblk = new T[newlen];
    elementCopy(newblk,m_data+m_first,m_blklen-m_first);
    elementCopy(newblk+m_blklen-m_first,m_data,m_first);
    delete[] m_data; m_data = newblk;
    m_blklen = newlen; m_first = 0;
  }

public:

  AmpiQ () {
    m_first  = 0;
    m_len    = 0;
    m_blklen = 1024; // default initial size
    m_data   = new T [m_blklen];
  }

  // sz must be a power of 2 for correct behaviour of this queue.
  AmpiQ (int sz) {
    m_first  = 0;
    m_len    = 0;
    m_blklen = sz;
    m_data   = new T [m_blklen];
  }

  ~AmpiQ () { delete [] m_data; }

  int length (void) { return m_len; }

  // return (length () == 0)
  bool isEmpty (void) { return  (0 == m_len); }

  // peek at the nth item from queue
  T& operator[] (size_t n) {
    n = ((n+m_first) & (m_blklen-1));
    return m_data [n];
  }

  T deq (void) {
    if (m_len>0) {
      T &ret = m_data [m_first];
      m_first = ((m_first+1) & (m_blklen-1));
      m_len --;
      return ret;
    } 
    else 
      return T(); //For builtin types like int, void*, this is equivalent to T(0)
  }

  void insert (int pos, T& elt) {
    while (m_len==m_blklen || pos>=m_blklen) _expand ();

    for (int i=m_len; i>pos; i--)
      m_data[(i+m_first)&(m_blklen-1)]=m_data[(i-1+m_first)&(m_blklen-1)];
    m_data [(pos+m_first)&(m_blklen-1)] = elt;
    if (pos > m_len) m_len = pos+1;
      else m_len++;
  }

  void enq (const T& elt) {
    if(m_len==m_blklen) _expand ();
    m_data [(m_first+m_len)&(m_blklen-1)] = elt;
    m_len++;
  }
};
#endif
