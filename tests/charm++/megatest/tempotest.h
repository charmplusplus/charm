#ifndef _TEMPOTEST_H
#define _TEMPOTEST_H

#include "megatest.h"
#include "tempotest.decl.h"
#include "tempo.h"

class IdMsg : public CMessage_IdMsg
{
  public :
    CkChareID id;
    IdMsg(CkChareID _id) : id(_id) {}
};

class tempotest_UserClass : public TempoChare
{
  CkChareID mainid;
  public :
    tempotest_UserClass(IdMsg *);
    tempotest_UserClass(CkMigrateMessage *m) {}
    void doSendRecv(void);
};

class tempotest_UserGroup : public TempoGroup
{
  CkChareID mainid;
  public :
    tempotest_UserGroup(IdMsg *);
    tempotest_UserGroup(CkMigrateMessage *m) {}
    void doSendRecv(void);
};
 
class tempotest_main : public Chare
{
  IdMsg *id1, *id2;
  void sendids(void);
  int recvd;

  public :
    tempotest_main(void);
    tempotest_main(CkMigrateMessage *m) {}
    void Finish(void);
    void getid(IdMsg *);
};

#endif
