/* Charm/Charm++ future related functions... Could be in a different file. */
#include "ckdefs.h"
#include "chare.h"
#include "c++interface.h"
#include "charmFutures.top.h"

int futureBocNum;

extern "C" void*  CharmBlockingCall(int entry, void * m, int g, int p);
extern "C" int createFuture ();
extern "C" void* waitFuture (int key, int free);
extern "C" void setFuture(int k, void * p);

extern "C" SetRefNumber(void *, int);
extern "C" GetRefNumber(void *);

class FutureInitMsg : public comm_object {
public:		int x ;
		} ;

void* CharmBlockingCall( int ep, void *m, int group, int processor)
{ 
  void * result;
  ENVELOPE *env = ENVELOPE_UPTR(m);
  int i = createFuture();
  SetRefNumber(m,i);
  CmiPrintf("GetRefnum=%d\n",   GetRefNumber(m));
SetEnv_pe(env, CmiMyPe());
  GeneralSendMsgBranch(ep, m, processor, -1, group);
  CkPrintf("[%d]: waiting in blocking call for future# %d\n",CmiMyPe(), i);
  result = waitFuture(i, 1);
//   CkPrintf("[%d]: waited in blocking call\n",CmiMyPe());
  return (result);
}

class Future: public groupmember {

public:

  Future(FutureInitMsg *m) 
  {
    //    CPrintf("[%d]Starting Future handler BOC\n", CmiMyPe());
  }

  SetFuture(FutureInitMsg * m)
  { int key;

    key = GetRefNumber(m);
    CPrintf("[%d]:Got new furture in : Future handler BOC. Key=%d\n",  CmiMyPe(),key);
    setFuture( key, m);
  }
};

extern "C" 
InitCharmFutures()
{
    FutureInitMsg *message2 = new (MsgIndex(FutureInitMsg)) FutureInitMsg ;	
    futureBocNum = new_group (Future, message2);
}

extern "C" 
CSendToFuture(void *m, int processor)
{

	CSendMsgBranch(Future, SetFuture, m, futureBocNum, processor);

}
#include "charmFutures.bot.h"
