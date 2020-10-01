#ifndef _LISTSEND_H
#define _LISTSEND_H

#include "megatest.h"
#include "listsend.decl.h"

class listsend_main : public CBase_listsend_main {
private:
  int count;
  std::vector<bool> checkedIn;
  CProxy_listsend_group groupProxy;
public:
  listsend_main(void);
  listsend_main(CkMigrateMessage *) {}
  void initDone(void);
  void check(int, unsigned int);
};

class listsend_group : public CBase_listsend_group {
private:
  CProxy_listsend_main mainProxy;
public:
  listsend_group(CProxy_listsend_main);
  listsend_group(CkMigrateMessage *) {}
  void multicast(unsigned int);
};

#endif
