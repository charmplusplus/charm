#ifndef _READONLY_H
#define _READONLY_H

#if CMK_HAS_TYPEINFO
# include <typeinfo>
#endif

#include <stdlib.h>
#include "charm.h"

extern int _mainDone;

class readonlybase {
private:
	//Don't use operator new
	void *operator new(size_t sz) { return malloc(sz);}

	//Don't use copy constructor, assign. operator, or destructor:
	readonlybase(const readonlybase &);
	readonlybase &operator=(const readonlybase &);
public:
	readonlybase() {}
	~readonlybase()	{ }
};


template <class dtype> class readonly : public readonlybase
{
  private:
    dtype data;

  public:
    readonly()
    {
      CkRegisterReadonly(
#if 0 /*CMK_HAS_TYPEINFO*/
	typeid(*this).name(),typeid(dtype).name()
#else
	"readonly<>","unknown"
#endif
	,sizeof(dtype), (void*)&data,NULL);
    }
    readonly<dtype>& operator=(dtype d)
    {
      if(_mainDone)
        CkAbort("Cannot set readonly variables after main::main\n");
      data = d;
      return *this;
    }
    operator dtype() { return data; }
};

template <class dtype> class _roatom : public readonlybase
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

template <class dtype, int len> class roarray : public readonlybase 
{
  private:
    _roatom<dtype> data[len];
  public:
    roarray()
    {
      CkRegisterReadonly(
#if 0 /*CMK_HAS_TYPEINFO*/
	typeid(*this).name(),typeid(data).name()
#else
	"roarray<>","unknown"
#endif
	,len*sizeof(_roatom<dtype>), (void*)&data[0],NULL);
    }
    _roatom<dtype> &operator[](int idx) { 
      if(idx <0 || idx>=len)
        CkAbort("Readonly array access bounds violation.\n");
      return data[idx]; 
    }
};

template <class dtype> class romsg : public readonlybase
{
  private:
    dtype *msg;
  public:
    romsg()
    {
      CkRegisterReadonlyMsg(
#if 0 /*CMK_HAS_TYPEINFO */
	typeid(*this).name(),typeid(dtype).name()
#else
	"romsg<>","unknown"
#endif
	,(void**) &msg);
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
};

#endif
