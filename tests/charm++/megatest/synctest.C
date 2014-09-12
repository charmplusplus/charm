#include "synctest.h"

void synctest_moduleinit(void) {}

void synctest_init(void) 
{
  // Construct this chare on PE 0, so that the array it creates does not need
  // the asynchronous API.
  CProxy_synctest_main::ckNew(0);
}

synctest_main::synctest_main(void) 
{
  count = 0;
  for (int i=0;i<NUMCHILDREN;i++) {
    synctest_InitMsg *s_im = new synctest_InitMsg;
    s_im->initValue = i+1;
    s_im->myMain = thishandle;
    CProxy_synctest_chare::ckNew(s_im);
  }
    synctest_InitMsg *s_im = new synctest_InitMsg;
    s_im->initValue = 23;
    s_im->myMain = thishandle;
    arr=CProxy_synctest_arr::ckNew(s_im,NUMCHILDREN);
}

void synctest_main::reply(synctest_ReplyMsg *s_rm) 
{
  children[count] = s_rm->childID;
  delete s_rm;
  count++;
  if (count == NUMCHILDREN) {
    CProxy_synctest_main mainproxy(thishandle);
    mainproxy.doTest();
  } 
}

void synctest_main::doTest(void)
{
  int i,sum = 0;
  synctest_SyncRecvMsg *childValue;

  for (i=0;i<NUMCHILDREN;i++) {
    CProxy_synctest_chare childproxy(children[i]);
    childValue = childproxy.test(new synctest_SyncSendMsg);
    sum = sum + childValue->value;
    delete childValue;
  }
  if (sum != (NUMCHILDREN * (NUMCHILDREN + 1))/2) {
    CkAbort("chare synctest failed!\n");
  }
  for (i=0;i<NUMCHILDREN;i++) {
    childValue = arr[i].test(new synctest_SyncSendMsg);
    if (childValue->value!=23) CkAbort("array synctest failed!\n");
    delete childValue;
  }
  delete this;
  megatest_finish();
}

synctest_chare::synctest_chare(synctest_InitMsg *s_im)
{
  value = s_im->initValue;
  myMain = s_im->myMain;
  delete s_im;
  CProxy_synctest_main mainproxy(myMain);
  synctest_ReplyMsg *myReply = new synctest_ReplyMsg;
  myReply->childID = thishandle;
  mainproxy.reply(myReply);
}

synctest_SyncRecvMsg *synctest_chare::test(synctest_SyncSendMsg *s_ssm)
{
  if (s_ssm->check() == 0)
    CkAbort("Message to chare sync method corrupted!");
  delete s_ssm;
  synctest_SyncRecvMsg *returnMsg = new synctest_SyncRecvMsg;
  returnMsg->value = value;
  return returnMsg;
}

synctest_arr::synctest_arr(synctest_InitMsg *s_im)
{
  value = s_im->initValue;
  myMain = s_im->myMain;
  delete s_im;
}

synctest_SyncRecvMsg *synctest_arr::test(synctest_SyncSendMsg *s_ssm)
{
  if (s_ssm->check() == 0)
    CkAbort("Message to array sync method corrupted!");
  delete s_ssm;
  synctest_SyncRecvMsg *returnMsg = new synctest_SyncRecvMsg;
  returnMsg->value = value;
  return returnMsg;
}

MEGATEST_REGISTER_TEST(synctest,"mjlang",1)  
#include "synctest.def.h"
