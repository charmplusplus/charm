#include "converse.h"
#include "packtest.h"

void packtest_init(void)
{ 
  CProxy_packtest_Btest::ckNew();
}

void packtest_moduleinit(void) {}

void *packtest_Msg::pack(packtest_Msg* m)
{
  int *p = (int *) CkAllocBuffer(m, (m->listsize+3)*sizeof(int));  
  int *t = p;

  *t = m->value; t++;
  *t = m->hop; t++;
  *t = m->listsize; t++;
  for(int i=0;i<m->listsize; i++, t++)
    *t = m->list1[i];
  delete m;
  return(p);
}

packtest_Msg * packtest_Msg::unpack(void *buf)
{
   int *in = (int *) buf;
   packtest_Msg *t = new (CkAllocBuffer(in, sizeof(packtest_Msg)))packtest_Msg;
   t->value = in[0];
   t->hop = in[1];
   t->listsize = in[2];
   t->list1 = new int[t->listsize];
   for(int i=0;i<t->listsize;i++)
     t->list1[i] = in[i+3];
   CkFreeMsg(buf);
   return t;
}

packtest_Btest::packtest_Btest(void)
{
/*
  static CrnStream str;
  static int flag = 0;
  if (0 == flag) {
    CrnInitStream(&str, (int)this, 0);
    flag = 1;
  }
  // seed needs to be set only once
*/

  if(CkMyPe()==0) {
    packtest_Msg *msg = new packtest_Msg;

    // msg->value = sentval = CrnInt(&str);
    msg->value = sentval = CrnRand();
    msg->hop=1;
    msg->listsize=10;
    msg->list1=new int[msg->listsize];
    for (int i=0;i<msg->listsize;i++) 
      msg->list1[i]=i;

    CProxy_packtest_Btest btest(thisgroup);
    btest[(CkMyPe()+1)%CkNumPes()].recv_msg(msg);
  }
}

void 
packtest_Btest::recv_msg(packtest_Msg * m)
{
  if (CkMyPe() == 0) {
    for (int i=0;i<m->listsize;i++)
      if(m->list1[i]!=i)
        CkAbort("packtest: message corrupted!\n");
    if(sentval != m->value)
      CkAbort("packtest: message corrupted!\n");
    delete m;
    megatest_finish();
  } else {
    CProxy_packtest_Btest btest(thisgroup);
    btest[(CkMyPe()+1)%CkNumPes()].recv_msg(m);
  }
}

MEGATEST_REGISTER_TEST(packtest,"fang",1)
#include "packtest.def.h"
