#ifndef ROTEST_H
#define ROTEST_H

#include "megatest.h"
#include "rotest.decl.h"

#define ROTEST_SIZE 100

extern roarray<int,10> rotest_iarray_num;
extern roarray<int,ROTEST_SIZE> rotest_iarray_sz;

class rotest_msg : public CMessage_rotest_msg
{
    int datasz;
    int *data;
  public:
    rotest_msg(void *m)
    {
      int *im = (int*) m;
      datasz = im[0];
      data = new int[datasz];
      for(int i=0;i<datasz;i++)
        data[i] = im[i+1];
    }
    rotest_msg(int sz)
    {
      data = new int[datasz=sz];
      for(int i=0; i<datasz; i++)
        data[i] = i*32-1;
    }
    ~rotest_msg()
    {
      delete[] data;
    }
    int check(void)
    {
      for(int i=0; i<datasz; i++)
        if(data[i] != (i*32-1))
          return 1;
      return 0;
    }
    static void *pack(rotest_msg *m)
    {
      char *pbuf = (char*) CkAllocBuffer(m, (m->datasz+1)*sizeof(int));
      *((int*)pbuf) = m->datasz;
      memcpy((void*)(pbuf+sizeof(int)), m->data, m->datasz*sizeof(int));
      return (void*) pbuf;
    }
    static rotest_msg *unpack(void *m)
    {
      rotest_msg *msg = (rotest_msg*) CkAllocBuffer(m, sizeof(rotest_msg));
      msg = new ((void*) msg) rotest_msg(m);
      CkFreeMsg(m);
      return msg;
    }
};

class rotest_group : public CBase_rotest_group {
  int numdone;
  public:
    rotest_group(void){ numdone = 0; }
    void start(void);
    void done(void);
    rotest_group(CkMigrateMessage *m) {}
};

#endif 
