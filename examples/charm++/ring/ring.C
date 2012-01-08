#include <string.h>
#include "ring.h"

void *
Msg::pack(Msg* m)
{
  int *p = (int *) CkAllocBuffer(m, (m->listsize+3)*sizeof(int));  
  int *t = p;

  *t = m->value; t++;
  *t = m->hop; t++;
  *t = m->listsize; t++;
  for(int i=0;i<m->listsize; i++, t++)
    *t = m->list1[i];
  delete [] m->list1;
  CkFreeMsg(m);
  return(p);
}

Msg *
Msg::unpack(void *buf)
{
   int *in = (int *) buf;
   Msg *t = new (CkAllocBuffer(in, sizeof(Msg))) Msg;
   t->value = in[0];
   t->hop = in[1];
   t->listsize = in[2];
   t->list1 = new int[t->listsize];
   for(int i=0;i<t->listsize;i++)
     t->list1[i] = in[i+3];
   CkFreeMsg(buf);
   return t;
}

Btest::Btest(Msg *m)
{
  CkPrintf("branch created on %d\n", CkMyPe());

// Uses the hop information from the message.
  nexthop=(CkMyPe()+(m->hop)) % CkNumPes();

// Takes care of negative hops
  if (nexthop < 0) nexthop += CkNumPes();

  if (CkMyPe()==0) {
    CProxy_Btest btest(thisgroup);
    btest[nexthop].recv_msg(m);
  } else  {
    delete m;
  }
}

void 
Btest::recv_msg(Msg * m)
{
  CkPrintf("Message received on  %d for ring %d\n", CkMyPe(), m->value);

// BOC at Processor 0 initiates and terminates a ring.
  if (CkMyPe() == 0) {
     CkPrintf("Message got back! %d \n", m->value);
     CProxy_main mainchare(main::mainhandle);
     mainchare.quit_when_done(m);
  } else {
    CProxy_Btest btest(thisgroup);
    btest[nexthop].recv_msg(m);
  }
}

CkChareID main::mainhandle;

main::main(CkArgMsg* m)
{
  Msg *msg1 = new Msg;
  mainhandle=thishandle;

  msg1->value = 1;
  msg1->hop=1;
  msg1->listsize=10;
  msg1->list1=new int[msg1->listsize];
  for (int i=0;i<msg1->listsize;i++) 
    msg1->list1[i]=i;
  CProxy_Btest::ckNew(msg1);
  delete m;
}

// Keeps track of the number of rings that terminated. An alternative 
// would be to use quiscence detection.
void 
main::quit_when_done(Msg *m)
{
   CkPrintf("The ring %d terminated\n", m->value);
   for (int i=0;i<m->listsize;i++)
     CkPrintf("%d\n", m->list1[i]);
   delete m;
   CkExit();
}

#include "ring.def.h"

