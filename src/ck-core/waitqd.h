#ifndef _WAITQD_H
#define _WAITQD_H

#include "charm++.h"
#include "conv-lists.h"
#include "waitqd.decl.h"

extern "C" void CkWaitQD(void);

class waitqd_QDChare : public Chare {
 private:
   bool waitStarted;
   void *threadList;
 public:
   waitqd_QDChare(CkArgMsg *ckam);
   waitqd_QDChare(CkMigrateMessage *m):Chare(m), waitStarted(false), threadList(0) {}
   void waitQD(void);
   void onQD(CkQdMsg *ckqm);
};

#endif
