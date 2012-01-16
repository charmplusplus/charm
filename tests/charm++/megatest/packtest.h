#ifndef _PACKTEST_H
#define _PACKTEST_H

#include "megatest.h"
#include "packtest.decl.h"

class packtest_Msg : public CMessage_packtest_Msg
{
  public:
    int value;
    int hop;
    int *list1;
    int listsize;
    static void *pack(packtest_Msg *);
    static packtest_Msg  *unpack(void *);
    packtest_Msg(void) { list1=0; }
    ~packtest_Msg(void) { if(list1) delete[] list1; }
};

class packtest_Btest : public CBase_packtest_Btest
{
  private:
    int sentval;
  public:
    packtest_Btest(void);
    packtest_Btest(CkMigrateMessage *m) {}
    void recv_msg(packtest_Msg *);
};

#endif
