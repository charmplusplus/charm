#ifndef _SYNCTEST_H
#define _SYNCTEST_H

#include "megatest.h"
#include "synctest.decl.h"

#define NUMCHILDREN 100


class synctest_InitMsg : public CMessage_synctest_InitMsg {
public:
  CkChareID myMain;
  int initValue;
};

class synctest_ReplyMsg : public CMessage_synctest_ReplyMsg {
public:
  CkChareID childID;
};

class synctest_SyncSendMsg : public CMessage_synctest_SyncSendMsg {
private:
  char checkString[6];
public:
  synctest_SyncSendMsg(void) { sprintf(checkString, "Sync!");} 
  int check(void) { 
    if (strcmp("Sync!", checkString) == 0)
      return 1;
    else
      return 0;
  }
};

class synctest_SyncRecvMsg : public CMessage_synctest_SyncRecvMsg {
public:
  int value;
};

class synctest_main : public CBase_synctest_main {
private:
  int count;
  CkChareID children[NUMCHILDREN];
  CProxy_synctest_arr arr;
public:
  synctest_main(void);
  synctest_main(CkMigrateMessage *m) {}
  void reply(synctest_ReplyMsg *s_rm);
  void doTest(void);
};

class synctest_chare : public CBase_synctest_chare {
private:
  int value;
  CkChareID myMain;
public:
  synctest_chare(synctest_InitMsg *s_im);
  synctest_chare(CkMigrateMessage *m) {}
  synctest_SyncRecvMsg *test(synctest_SyncSendMsg *s_ssm);
};

class synctest_arr : public CBase_synctest_arr {
private:
  int value;
  CkChareID myMain;
public:
  synctest_arr(synctest_InitMsg *s_im);
  synctest_arr(CkMigrateMessage *m) {}
  synctest_SyncRecvMsg *test(synctest_SyncSendMsg *s_ssm);
};

#endif
