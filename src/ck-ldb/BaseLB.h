#ifndef BASELB_H
#define BASELB_H

#include "LBDatabase.h"

/// Base class for all LB strategies.
class BaseLB: public Group
{
protected:
  LBDatabase *theLbdb;
  LDBarrierReceiver receiver;
public:
  BaseLB() ;
  void unregister(); 
};

#endif
