#ifndef BLUE_TIMING_H
#define BLUE_TIMING_H

#include "cklists.h"

class bgMsgEntry {
public:
  int msgID;
  double time;
};

class bgTimingLog {
public:
  int ep;
  double time;
  int srcpe;
  int msgID;
  CkVec< bgMsgEntry > msgs;
};

typedef CkQ< bgTimingLog > BgTimeLine;


#endif
