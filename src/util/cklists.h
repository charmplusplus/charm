#ifndef _CKLISTS_H
#define _CKLISTS_H

#include <converse.h> // for size_t
#include <stdlib.h> // for size_t
#include <string.h> // for memcpy

template <class T>
class CkQ {
    T *block;
    int blklen;
    int first;
    int len;
    void _expand(void) {
      int newlen=len*2;
      T *newblk = new T[newlen];
      memcpy(newblk, block+first, sizeof(T)*(blklen-first));
      memcpy(newblk+blklen-first, block, sizeof(T)*first);
      delete[] block; block = newblk;
      blklen = newlen; first = 0;
    }
  public:
    CkQ() :first(0),len(0) {
      block = new T[blklen=16];
    }
    CkQ(int sz) :first(0),len(0) {
      block = new T[blklen=sz];
    }
    ~CkQ() { delete[] block; }
    int length(void) { return len; }
    CmiBool isEmpty(void) { return (CmiBool)(len==0); }
    T deq(void) {
      if(len>0) {
        T &ret = block[first];
        first = (first+1)%blklen;
        len--;
      	return ret;
      } else return T(0);
    }
    void enq(const T &elt) {
      if(len==blklen) _expand();
      block[(first+len)%blklen] = elt;
      len++;
    }
    // stack semantics, needed to replace FIFO_QUEUE of converse
    void push(const T &elt) {
      if(len==blklen) _expand();
      first = (first-1+blklen)%blklen;
      block[first] = elt;
      len++;
    }
    //Peek at the n'th item from the queue
    T& operator[](size_t n) 
    {
    	n=(n+first)%blklen;
    	return block[n];
    }
    // needed to replace FIFO_Enumerate
    T* getArray(void) {
      T *newblk = new T[len];
      int i,j;
      for(i=0,j=first;i<len;i++){
        newblk[i] = block[j];
        j = (j+1)%blklen;
      }
      return newblk;
    }
};

template <class T>
class CkVec {
    T *block;
    int blklen,len;
  public:
    CkVec() {block=NULL;blklen=len=0;}
    ~CkVec() { delete[] block; }
    int &length(void) { return len; }
    T *getVec(void) { return block; }
    T& operator[](size_t n) { return block[n]; }
    const T& operator[](size_t n) const { return block[n]; }
    void insert(int pos, const T &elt) {
      if(pos>=blklen) {
      	int newlen=pos*2+16;//Double length at each step
        T *newblk = new T[newlen];
        if (block!=NULL)
        	memcpy(newblk, block, sizeof(T)*blklen);
        for(int i=blklen; i<newlen; i++) newblk[i] = T(0);
        delete[] block; block = newblk;
        blklen = newlen;
      }
      if (pos>=len) len=pos+1;
      block[pos] = elt;
    }
    void insertAtEnd(const T &elt) {insert(length(),elt);}
    void push_back(const T &elt) {insert(length(),elt);}
};
#endif
