#ifndef _IMMEDIATERING_H
#define _IMMEDIATERING_H

#include "immediatering.decl.h"
#include "megatest.h"

struct immediateMsg {
        char converseHeader[CmiMsgHeaderSizeBytes];
        int iter; /* number of times we've been around the ring. */
        char data[7];
};

// charm immeidate message
class immMessage : public CMessage_immMessage {
  char data[7];
public:
  int iter;
  immMessage(void) { sprintf(data, "Array!"); iter = 0; };
  int check(void) { return !strcmp(data, "Array!"); }
};

class immRing_nodegroup : public CBase_immRing_nodegroup
{
 public:
  immRing_nodegroup() {}
  immRing_nodegroup(CkMigrateMessage *msg) {}
  void start(immMessage *msg);
};

class immRing_group : public CBase_immRing_group
{
 public:
  immRing_group() {}
  immRing_group(CkMigrateMessage *msg) {}
  void start(immMessage *msg);
};

#endif
