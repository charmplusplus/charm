#ifndef _READONLY_H
#define _READONLY_H

#include <stdlib.h>
#include "charm.h"

extern int _mainDone;

template <class dtype> class readonly
{
  private:
    dtype data;
    // operator new is private, because we dont want anyone to
    // dynamically allocate readonly variables
    // cannot return 0 in operator new
    void *operator new(size_t sz) { return malloc(sz); }
  public:
    readonly()
    {
      CkRegisterReadonly(sizeof(dtype), (void*) &data);
    }
    readonly<dtype>& operator=(dtype d)
    {
      if(_mainDone)
        CkAbort("Cannot set readonly variables after main::main\n");
      data = d;
      return *this;
    }
    operator dtype() { return data; }
    ~readonly()
    {
      if(_mainDone==0) {
        CkAbort("Destructor for readonly variable called.\n"
                "Cannot have local readonly variables\n");
      }
    }
};

template <class dtype> class _roatom 
{
  private:
    dtype data;
  public:
    _roatom<dtype>& operator=(dtype d)
    {
      if(_mainDone)
        CkAbort("Cannot set readonly variables after main::main\n");
      data = d;
      return *this;
    }
    operator dtype() { return data; }
};

template <class dtype, int len> class roarray 
{
  private:
    _roatom<dtype> data[len];
    // operator new is private, because we dont want anyone to
    // dynamically allocate readonly variables
    // cannot return 0 in operator new
    void *operator new(size_t sz) { return malloc(sz); }
  public:
    roarray()
    {
      CkRegisterReadonly(len*sizeof(_roatom<dtype>), (void*) &data[0]);
    }
    _roatom<dtype> &operator[](int idx) { 
      if(idx <0 || idx>=len)
        CkAbort("Readonly array access bounds violation.\n");
      return data[idx]; 
    }
    ~roarray()
    {
      if(_mainDone==0) {
        CkAbort("Destructor for readonly variable called.\n"
                "Cannot have local readonly variables\n");
      }
    }
};

template <class dtype> class romsg
{
  private:
    dtype *msg;
    // operator new is private, because we dont want anyone to
    // dynamically allocate readonly variables
    // cannot return 0 in operator new
    void *operator new(size_t sz) { return malloc(sz); }
  public:
    romsg()
    {
      CkRegisterReadonlyMsg((void**) &msg);
    }
    romsg<dtype>& operator=(dtype *d)
    {
      if(_mainDone)
        CkAbort("Cannot set readonly variables after main::main\n");
      msg = d;
      return *this;
    }
    operator dtype*() { return msg; }
    dtype& operator*() { return *msg; }
    dtype* operator->() { return msg; }
    ~romsg()
    {
      if(_mainDone==0) {
        CkAbort("Destructor for readonly variable called.\n"
                "Cannot have local readonly variables\n");
      }
    }
};

#endif
