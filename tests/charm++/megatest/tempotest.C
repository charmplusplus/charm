#include "tempotest.h"

void tempotest_init(void)
{
  CProxy_tempotest_main::ckNew(0);
}

void tempotest_moduleinit(void) {}

tempotest_UserClass::tempotest_UserClass(IdMsg *msg)
{
  mainid = msg->id;
  delete msg;
  CProxy_tempotest_UserClass uc(thishandle);
  uc.doSendRecv();
}

void
tempotest_UserClass::doSendRecv(void)
{
  CkChareID otherid;
  char inbuf[11], outbuf[11];
  
  IdMsg *idmsg = new IdMsg(thishandle);
  CProxy_tempotest_main mainproxy(mainid);
  mainproxy.getid(idmsg);

  ckTempoRecv(1, &otherid, sizeof(CkChareID));
  
  for (int i=0; i<10; i++) {
    sprintf(outbuf, "UserClass!");
    sprintf(inbuf, "");
    ckTempoSend(i+2, outbuf, strlen(outbuf)+1, otherid);
    ckTempoRecv(i+2, inbuf, 11);
    if(strcmp(inbuf, "UserClass!")) {
      CkAbort("tempotest: Message corrupted!\n");
      mainproxy.Finish();
      return;
    }
  }
  mainproxy.Finish();
  delete this;
}

tempotest_UserGroup::tempotest_UserGroup(IdMsg *msg)
{
  mainid = msg->id;
  CProxy_tempotest_UserGroup ug(thisgroup);
  ug[CkMyPe()].doSendRecv();
  delete msg;
}

void
tempotest_UserGroup::doSendRecv(void)
{
  char *outbuf = new char[1024000];
  if(CkMyPe()==0)
    sprintf(outbuf, "UserGroup!");
  ckTempoBcast(CkMyPe()==0, 1001, outbuf, 11);
  if(strcmp(outbuf, "UserGroup!"))
    CkAbort("tempotest: Message corrupted!\n");
  CProxy_tempotest_main mainproxy(mainid);
  mainproxy.Finish();
  delete [] outbuf;
  delete this;
}

void 
tempotest_main::sendids(void)
{
  TempoChare::ckTempoSend(1, &(id2->id), sizeof(CkChareID), id1->id); 
  TempoChare::ckTempoSend(1, &(id1->id), sizeof(CkChareID), id2->id); 
  delete id1;
  delete id2;
}

tempotest_main::tempotest_main(void)
{
  id1 = id2 = 0;
  recvd = 0;
  CProxy_tempotest_UserGroup::ckNew(new IdMsg(thishandle));
  CProxy_tempotest_UserClass::ckNew(new IdMsg(thishandle));
  CProxy_tempotest_UserClass::ckNew(new IdMsg(thishandle));
}

void tempotest_main::Finish(void)
{
  recvd++;
  if(recvd==(CkNumPes()+2))
    megatest_finish();
}

void tempotest_main::getid(IdMsg *idmsg)
{
  if (id1 == 0)
    id1 = idmsg;
  else {
    id2 = idmsg;
    sendids();
  }
}

MEGATEST_REGISTER_TEST(tempotest,"fang",1)
#include "tempotest.def.h"
